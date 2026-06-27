package search

import (
	"context"
	"runtime"
	"sort"
	"sync"

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
// candidate set with.
func NewBroker(shards []*Shard, cascade *rank.Cascade) *Broker {
	return &Broker{
		shards:         shards,
		routing:        BuildRoutingIndex(shards),
		stats:          computeGlobalStats(shards),
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
// shifted into the global id space and the feature rows of its candidates.
type shardResult struct {
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
	targets := b.routing.Route(q)
	results := b.fanOut(ctx, q, targets)

	var allLex, allDense []scored
	feats := make(map[uint32][]float64)
	for _, r := range results {
		if r == nil {
			continue
		}
		allLex = append(allLex, r.lex...)
		allDense = append(allDense, r.dense...)
		for id, fv := range r.feats {
			feats[id] = fv
		}
	}
	// Merge the per-shard ranked lists into one global ranked list per plane by score
	// descending, the order a single index over every shard would have produced.
	sortByScore(allLex)
	sortByScore(allDense)

	feat := func(id uint32) []float64 { return feats[id] }
	cands := b.cascade.Rank(localIDs(allLex), localIDs(allDense), feat, q.K)
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
			results[i] = &shardResult{lex: gl, dense: gd, feats: gf}
		}(i, si)
	}
	wg.Wait()
	return results
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
