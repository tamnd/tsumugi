package search

import (
	"context"
	"runtime"
	"sort"
	"sync"
)

// Searcher is one node in the serving tree. A broker over shards and an aggregator over
// other nodes both answer a query with a completeness-carrying top-k and report how many
// shards sit beneath them, which is what lets an aggregator fan across a mix of brokers
// and sub-aggregators and roll their completeness up the same way a broker rolls its
// shards'. Both *Broker and *Aggregator satisfy it, so a tree of any depth is built by
// nesting aggregators.
type Searcher interface {
	// SearchComplete answers the query with the node's top-k and the completeness of the
	// fan-out beneath it.
	SearchComplete(ctx context.Context, q Query) Results
	// NumShards is the count of leaf shards beneath the node, known statically, so an
	// aggregator can account for a child that was dropped at the deadline without having
	// heard from it.
	NumShards() int
	// NumDocs is the fleet-wide document count beneath the node, the N idf divides by, so
	// a parent aggregator sums its children's NumDocs into the fleet N it computes one
	// shared idf against.
	NumDocs() uint64
	// DocFreqs sums each term's document frequency across the subtree beneath the node, so
	// a parent aggregator can add the per-child counts into the fleet-wide df no single
	// child can see. It is the cross-node half of the global idf the broker already
	// computes across its own shards.
	DocFreqs(ctx context.Context, terms []string) map[string]uint32
	// Stats is the node's fleet-wide normalization statistics, the document and field
	// average lengths and the counts they derive from, so a parent aggregator can fold its
	// children's stats into the deployment-wide averages and push them back down when the
	// children were built over separately-partitioned statistics. It is the
	// length-normalization counterpart to DocFreqs: DocFreqs lets the aggregator unify idf,
	// Stats lets it unify the BM25F length denominators (doc 11, the partitioned case).
	Stats() GlobalStats
}

// Aggregator is the top tier of the serving topology: it holds no shards, it holds a set
// of child nodes, and it fans a query across them and merges their top-k into a top-k over
// every shard beneath the whole subtree. One aggregator over a handful of brokers reaches
// past what one broker's shards can hold, and an aggregator over aggregators reaches
// internet scale by adding levels, each level fanning across the level below and merging,
// so the corpus a deployment serves grows with the depth and width of the tree rather than
// with one machine's memory (doc 11, "The serving topology").
//
// The merge composes exactness up the tree. Each broker returns its top candidates already
// scored by the global L2 model against the shared fleet-wide statistics, so the brokers'
// scores are on one scale and the aggregator takes the top across them by comparing those
// scores rather than re-running L2. This is exact: a document in the global top-k ranks no
// lower inside its own broker than it does globally, so it is in that broker's top-k, so
// the union of the brokers' top-ks contains the whole global top-k and the merge recovers
// it. A shard is exact over its documents, a broker over its shards, and an aggregator over
// its brokers by the same mechanism, so a top-k from the root of a tree of any depth is the
// single-index top-k over the whole corpus (doc 11, "Exactness up the tree").
type Aggregator struct {
	children []Searcher

	maxConcurrency int
}

// NewAggregator builds an aggregator over already-constructed child nodes, each a broker or
// a sub-aggregator. The children must share the collection's fleet-wide normalization
// statistics, the average document and field lengths and the static-rank and PageRank
// constants, so their L2 scores are comparable at the merge, the common single-GlobalStats
// case the cheap exact merge rests on. The one fleet-wide statistic the aggregator does not
// rely on the children already sharing is the per-term idf: a broker can only gather df over
// its own shards, so the aggregator gathers the df across every child and pushes one shared
// idf down at query time (see SearchComplete), which is what makes the merge exact even when
// the brokers' vocabularies diverge. A deployment that partitions the rest of the statistics
// across child groups, where each child's average field lengths describe only its own slice,
// is handled the same way: the aggregator folds the children's stats into the deployment-wide
// field averages and pushes those down too, so a broker normalizes BM25F against the unified
// denominators and its L2 scores land on the same scale as its siblings'. Because the L2 model
// is pointwise this is exact, the score a broker computes against the unified stats is the
// score the aggregator would have recomputed, so the merge stays a comparison of comparable
// scores rather than a re-run of L2 (see SearchComplete).
func NewAggregator(children []Searcher) *Aggregator {
	return &Aggregator{children: children, maxConcurrency: runtime.GOMAXPROCS(0)}
}

// NumShards is the total count of leaf shards beneath the aggregator, the sum over its
// children, so an aggregator nested under another reports the size of its whole subtree.
func (a *Aggregator) NumShards() int {
	n := 0
	for _, c := range a.children {
		n += c.NumShards()
	}
	return n
}

// NumChildren is the count of immediate child nodes the aggregator fans across.
func (a *Aggregator) NumChildren() int { return len(a.children) }

// NumDocs is the fleet-wide document count beneath the aggregator, the sum over its
// children, so the aggregator divides one shared idf by the same N a single index over the
// whole subtree would (the Searcher contract).
func (a *Aggregator) NumDocs() uint64 {
	var n uint64
	for _, c := range a.children {
		n += c.NumDocs()
	}
	return n
}

// Stats folds the children's fleet-wide statistics into one deployment-wide set of
// normalization averages. The fold runs over the raw additive accumulators, not the
// averages, so a token-weighted average comes out right no matter how unevenly the corpus is
// split across children: a child holding ten million documents pulls the average toward its
// lengths far harder than one holding ten thousand, the same average a single index over the
// whole subtree would compute. It is the length-normalization counterpart to DocFreqs and
// NumDocs, and it is what lets SearchComplete push one shared set of field averages down to
// children built over separately-partitioned statistics so their L2 scores stay comparable
// at the merge (doc 11, the partitioned-GlobalStats case).
func (a *Aggregator) Stats() GlobalStats {
	var sums statSums
	for _, c := range a.children {
		sums = sums.add(sumsFromStats(c.Stats()))
	}
	return sums.global()
}

// DocFreqs sums each term's document frequency across the whole subtree by gathering it
// from every child concurrently and adding the per-child counts. A child is a broker that
// sums over its shards or a sub-aggregator that recurses, so the gather composes down the
// tree the same way the search does, and the total is the term's true fleet-wide df. It
// reads only bloom filters and dictionaries, so it is cheap next to the search it precedes,
// and it honors the deadline: a cancelled context returns whatever has been gathered,
// which only loosens an idf, never corrupts a score.
func (a *Aggregator) DocFreqs(ctx context.Context, terms []string) map[string]uint32 {
	if len(terms) == 0 {
		return nil
	}
	df := make(map[string]uint32)
	var mu sync.Mutex
	sem := make(chan struct{}, a.maxConcurrency)
	var wg sync.WaitGroup
	for _, c := range a.children {
		select {
		case <-ctx.Done():
			wg.Wait()
			return df
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(c Searcher) {
			defer wg.Done()
			defer func() { <-sem }()
			local := c.DocFreqs(ctx, terms)
			if len(local) == 0 {
				return
			}
			mu.Lock()
			for t, f := range local {
				df[t] += f
			}
			mu.Unlock()
		}(c)
	}
	wg.Wait()
	return df
}

// Search fans the query across the children and returns the merged fleet-wide top-k,
// dropping the completeness indicator. SearchComplete returns the same top-k with the
// count of shards reached so a caller can tell a complete answer from a degraded one.
func (a *Aggregator) Search(ctx context.Context, q Query) []Hit {
	return a.SearchComplete(ctx, q).Hits
}

// SearchComplete fans the query across the children up to the deadline, merges the top-ks
// of those that responded into a fleet-wide top-k, and rolls their completeness up. A child
// that responded contributes its own shards_total and shards_ok, so a child that was itself
// partial stays partial in the sum; a child that was dropped at the deadline contributes its
// whole shard count to the total and nothing to the count reached, so the result is flagged
// partial whenever any shard anywhere beneath the tree was missed and is never passed off as
// complete (doc 11, "Failure modes and partial results", composed up a level).
func (a *Aggregator) SearchComplete(ctx context.Context, q Query) Results {
	// Phase one of cross-broker exact idf: a term's fleet-wide df is the sum across every
	// broker's shards, which no single broker can see, because a broker gathers df only over
	// its own shards and divides by its own N. So the aggregator gathers the df across all
	// the children, turns it into one fleet-wide idf against the fleet N, and pushes it down
	// into the query before fan-out. Each broker then scores against this shared idf instead
	// of its own partial one (the broker honors a pre-set TermIDF rather than recomputing),
	// so the brokers' L2 scores land on one scale and the merge below is a comparison of
	// comparable scores, the cheap exact merge (doc 11, "Exactness up the tree"). The
	// children must already share the fleet-wide GlobalStats for the rest of the
	// normalization (avg lengths, N), the NewAggregator contract; idf is the one statistic
	// the aggregator must gather at query time because it is specific to the query's terms.
	// A query that already carries idf overrides or has no lexical terms skips the gather.
	if len(q.Terms) > 0 && q.TermIDF == nil {
		q.TermIDF = idfFromDF(a.DocFreqs(ctx, q.Terms), a.NumDocs())
	}
	// Phase two, the length-normalization counterpart to the idf push-down above: a broker
	// normalizes BM25F by its own fleet average field lengths, which describe only the slice
	// of the collection its shards hold when the deployment partitions the statistics across
	// child groups. So the aggregator folds the children's stats into the deployment-wide
	// field averages and pushes them down, and each broker normalizes against those instead of
	// its own slice's, which puts every broker's BM25F on one denominator and so its L2 scores
	// on one scale. This is what makes the cheap comparable-score merge exact for a partitioned
	// deployment without re-running L2 at the aggregator: the L2 model is pointwise, a
	// candidate's score is a function of its own feature row and the shared stats alone, so a
	// broker scoring against the unified stats yields exactly the score the aggregator would
	// have recomputed (doc 11, "Exactness up the tree", the partitioned-GlobalStats case). A
	// single-broker deployment, or one whose children already share fleet stats, lands on the
	// same averages it already had, so the push-down is a no-op there and costs only the fold.
	if len(q.Terms) > 0 && q.AvgFieldLen == nil {
		afl := a.Stats().AvgFieldLen
		q.AvgFieldLen = &afl
	}
	results, _ := a.fanOut(ctx, q)

	var hits []Hit
	total, ok := 0, 0
	for i, c := range a.children {
		if r := results[i]; r != nil {
			hits = append(hits, r.Hits...)
			total += r.ShardsTotal
			ok += r.ShardsOK
			continue
		}
		// A dropped child never reported, so its query-routed shard total is unknown; its
		// full shard count is the honest upper bound on what was missed, which can only
		// move the result toward incomplete, never falsely toward complete.
		total += c.NumShards()
	}
	// The children already scored their candidates with the global L2 model against the
	// shared statistics, so the merge is a comparison of comparable scores: order by score
	// descending, ties broken by ascending id for a deterministic result, and keep the top
	// k. Each child returned at least its own top-k, so this union holds the whole global
	// top-k.
	sortHits(hits)
	if q.K > 0 && len(hits) > q.K {
		hits = hits[:q.K]
	}
	return Results{Hits: hits, ShardsTotal: total, ShardsOK: ok}
}

// fanOut runs the children concurrently, bounded by the aggregator's concurrency limit,
// and returns each responding child's result aligned with the children slice along with the
// count that responded. It is deadline-aware on both ends like the broker's fan-out: a
// cancelled context stops dispatching further children, and the collection stops at the
// deadline rather than waiting for a slow child, so a slow broker is dropped instead of
// holding the whole query past its budget. The result channel is buffered to the full child
// count so a child goroutine that finishes after the deadline can still send and exit rather
// than leaking.
func (a *Aggregator) fanOut(ctx context.Context, q Query) ([]*Results, int) {
	type childResult struct {
		idx int
		res Results
	}
	out := make(chan childResult, len(a.children))
	sem := make(chan struct{}, a.maxConcurrency)
	dispatched := 0
dispatch:
	for i := range a.children {
		select {
		case <-ctx.Done():
			break dispatch
		case sem <- struct{}{}:
		}
		dispatched++
		go func(i int) {
			defer func() { <-sem }()
			out <- childResult{idx: i, res: a.children[i].SearchComplete(ctx, q)}
		}(i)
	}
	results := make([]*Results, len(a.children))
	got := 0
	for got < dispatched {
		select {
		case r := <-out:
			rr := r.res
			results[r.idx] = &rr
			got++
		case <-ctx.Done():
			return results, got
		}
	}
	return results, got
}

// sortHits orders hits by score descending, ties broken by ascending id, the same order the
// broker's per-plane merge uses so the aggregator's merged order is deterministic across
// runs and across how the corpus is partitioned into brokers.
func sortHits(hh []Hit) {
	sort.Slice(hh, func(a, b int) bool {
		if hh[a].Score != hh[b].Score {
			return hh[a].Score > hh[b].Score
		}
		return hh[a].DocID < hh[b].DocID
	})
}
