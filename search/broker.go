package search

import (
	"context"
	"runtime"
	"sort"
	"sync"

	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
)

// Broker fans a query across many shards, gathers their candidates, and runs one
// global rerank to produce a fleet-wide top-k. It owns the shards, the routing index
// that prunes the fan-out, the fleet-wide statistics, and the cascade that does the
// global rerank. The broker is read-only after construction and serves queries
// concurrently.
//
// The global rerank is where the broker's exactness lives. Each shard does only
// retrieval; the broker collects every candidate with its per-document feature row,
// merges the per-shard ranked lists into global ranked lists, and runs the cascade
// once over the union. Because a document's final score is the global L2 model over
// its own feature row, the score does not depend on which shard the document came
// from, so given a recall-complete candidate set the broker's top-k equals the top-k
// a single index over every shard would produce.
type Broker struct {
	shards  []*Shard
	routing *RoutingIndex
	stats   GlobalStats
	cascade *rank.Cascade

	maxConcurrency int
}

// NewBroker builds a broker over already-opened shards with a global cascade. It
// constructs the routing index and the fleet-wide statistics from the shards. The
// cascade carries the global L1 cut and L2 model the broker reranks the merged
// candidate set with. This is the scan path: it walks every shard's vocabulary to
// build the routing index, which is fine for a handful of shards but is what the
// persisted artifact replaces at fleet scale, through NewBrokerWith.
func NewBroker(shards []*Shard, cascade *rank.Cascade) *Broker {
	return NewBrokerWith(shards, cascade, BuildRoutingIndex(shards), computeGlobalStats(shards))
}

// NewBrokerWith builds a broker over already-opened shards with a routing index and
// fleet-wide statistics supplied from outside, the values a collection artifact loads.
// It is the scale path: serve reads the manifest, the statistics, and the routing index
// from one file rather than rescanning every shard, so startup is proportional to the
// query vocabulary, not the corpus. The routing index's shard ids must line up with the
// order of the shards slice.
func NewBrokerWith(shards []*Shard, cascade *rank.Cascade, routing *RoutingIndex, stats GlobalStats) *Broker {
	return &Broker{
		shards:         shards,
		routing:        routing,
		stats:          stats,
		cascade:        cascade,
		maxConcurrency: runtime.GOMAXPROCS(0),
	}
}

// Stats returns the fleet-wide collection statistics.
func (b *Broker) Stats() GlobalStats { return b.stats }

// NumShards is the number of shards the broker serves.
func (b *Broker) NumShards() int { return len(b.shards) }

// Close releases every shard's file mapping.
func (b *Broker) Close() error {
	var first error
	for _, s := range b.shards {
		if err := s.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// shardResult is one shard's contribution to a fan-out: its ranked lists already
// shifted into the global id space, the feature rows of its candidates, and the
// index of the shard that produced them so the global rerank can route a survivor's
// online feature extraction back to the shard that holds its text.
type shardResult struct {
	shard int
	lex   []scored
	dense []scored
	feats map[uint32][]float64
}

// Search fans the query out to the routed shards, gathers their candidates, merges
// the per-shard ranked lists into global ranked lists, and runs the global rerank,
// returning the fleet-wide top-k. The fan-out is concurrent and bounded by a
// semaphore; the context cancels it on deadline so a slow shard cannot hold the
// whole query past its budget.
func (b *Broker) Search(ctx context.Context, q Query) []Hit {
	// Analyze the query once at the broker and ship the term set to every shard, so the
	// analysis chain runs one time per query rather than once per shard the fan-out
	// visits. The shards score q.Terms directly; routing and the idf gather read the
	// same set. A query that already carries Terms (a pre-parsed ParsedQuery) keeps them.
	if q.Terms == nil {
		q.Terms = q.lexTerms()
	}
	targets := b.routing.RouteTerms(q.Terms)
	// Phase one of distributed exact idf: gather each query term's df across the routed
	// shards and turn it into one collection-wide idf the fan-out scores every shard
	// against. Without this each shard would use its local idf, and the merge would
	// favor whichever shard happens to hold a globally rare term densely. A query that
	// already carries idf overrides, or has no lexical text, skips the gather.
	if len(q.Terms) > 0 && q.TermIDF == nil {
		q.TermIDF = b.globalIDF(ctx, q.Terms, targets)
	}
	results := b.fanOut(ctx, q, targets)

	var allLex, allDense []scored
	feats := make(map[uint32][]float64)
	owner := make(map[uint32]int) // global id -> index of the shard that holds it
	for _, r := range results {
		if r == nil {
			continue
		}
		allLex = append(allLex, r.lex...)
		allDense = append(allDense, r.dense...)
		for id, fv := range r.feats {
			feats[id] = fv
			owner[id] = r.shard
		}
	}
	// Merge the per-shard ranked lists into one global ranked list per plane by score
	// descending, the order a single index over every shard would have produced.
	sortByScore(allLex)
	sortByScore(allDense)

	// L1 reads the matrix row already gathered from each shard. L2 needs the online
	// query-dependent features, which depend on the candidate's text and so must be
	// extracted at the shard that holds it: the rerank routes each survivor back to
	// its owning shard's per-query extractor, built once and reused across that
	// shard's survivors. Online extraction runs only over the L1 survivors, the
	// bounded set the spec's budget allots the per-candidate text work.
	exts := make([]*onlineExtractor, len(b.shards))
	l1feat := func(id uint32) []float64 { return feats[id] }
	l2feat := func(id uint32) []float64 {
		base := feats[id]
		si, ok := owner[id]
		if !ok {
			return base
		}
		s := b.shards[si]
		if exts[si] == nil {
			exts[si] = s.newOnline(q, q.TermIDF, b.stats.AvgDocLen)
		}
		return s.l2Row(base, exts[si], id-s.nodeBase)
	}
	cands := b.cascade.Rank(localIDs(allLex), localIDs(allDense), l1feat, l2feat, q.K)
	hits := make([]Hit, len(cands))
	for i, c := range cands {
		hits[i] = Hit{DocID: c.DocID, Score: c.Score}
	}
	return hits
}

// fanOut runs the routed shards' retrievals concurrently, bounded by the broker's
// concurrency limit, and returns each shard's result shifted into the global id
// space. A cancelled context stops dispatching further shards.
func (b *Broker) fanOut(ctx context.Context, q Query, targets []int) []*shardResult {
	results := make([]*shardResult, len(targets))
	sem := make(chan struct{}, b.maxConcurrency)
	var wg sync.WaitGroup
	for i, si := range targets {
		select {
		case <-ctx.Done():
			wg.Wait()
			return results
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(i, si int) {
			defer wg.Done()
			defer func() { <-sem }()
			s := b.shards[si]
			lex, dense, feats := s.retrieve(q)
			base := s.nodeBase
			gl := make([]scored, len(lex))
			for j, c := range lex {
				gl[j] = scored{docID: base + c.docID, score: c.score}
			}
			gd := make([]scored, len(dense))
			for j, c := range dense {
				gd[j] = scored{docID: base + c.docID, score: c.score}
			}
			gf := make(map[uint32][]float64, len(feats))
			for id, fv := range feats {
				gf[base+id] = fv
			}
			results[i] = &shardResult{shard: si, lex: gl, dense: gd, feats: gf}
		}(i, si)
	}
	wg.Wait()
	return results
}

// globalIDF gathers each query term's document frequency across the routed shards and
// returns the collection-wide idf per term. The document count is the fleet-wide N from
// the broker's statistics, the same N a single index over every shard would divide by,
// and the df is summed over exactly the shards routing selected, which for a lexical
// term are all the shards that hold it, so the sum is the term's true collection df. The
// gather reads only bloom filters and dictionaries, so it is cheap next to the retrieval
// it precedes; it runs concurrently under the broker's concurrency bound and a cancelled
// context returns whatever has been gathered, which only loosens an idf, never corrupts a
// score. An empty result lets the fan-out fall back to shard-local idf.
func (b *Broker) globalIDF(ctx context.Context, terms []string, targets []int) map[string]float64 {
	df := make(map[string]uint32)
	var mu sync.Mutex
	sem := make(chan struct{}, b.maxConcurrency)
	var wg sync.WaitGroup
	for _, si := range targets {
		select {
		case <-ctx.Done():
			wg.Wait()
			return idfFromDF(df, b.stats.DocCount)
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(si int) {
			defer wg.Done()
			defer func() { <-sem }()
			local := b.shards[si].LexDocFreqs(terms)
			if len(local) == 0 {
				return
			}
			mu.Lock()
			for t, f := range local {
				df[t] += f
			}
			mu.Unlock()
		}(si)
	}
	wg.Wait()
	return idfFromDF(df, b.stats.DocCount)
}

// idfFromDF turns gathered per-term document frequencies into per-term idf against the
// collection-wide document count. It returns nil for an empty gather so the caller
// leaves the query's idf override unset and the shards score with their local idf.
func idfFromDF(df map[string]uint32, n uint64) map[string]float64 {
	if len(df) == 0 {
		return nil
	}
	out := make(map[string]float64, len(df))
	for t, f := range df {
		out[t] = lexical.IDF(n, uint64(f))
	}
	return out
}

// sortByScore orders a candidate list by score descending, ties broken by ascending
// id so the merged order is deterministic across runs and across shard partitionings.
func sortByScore(ss []scored) {
	sort.Slice(ss, func(a, b int) bool {
		if ss[a].score != ss[b].score {
			return ss[a].score > ss[b].score
		}
		return ss[a].docID < ss[b].docID
	})
}
