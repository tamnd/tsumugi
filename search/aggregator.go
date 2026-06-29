package search

import (
	"context"
	"runtime"
	"sort"
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
// a sub-aggregator. The children must share the collection's fleet-wide statistics so their
// L2 scores are comparable at the merge, the common single-GlobalStats case the cheap exact
// merge rests on; a deployment that partitions statistics across child groups is the case
// the spec hands a re-run of L2 at the aggregator, which this merge does not yet do (see
// the package notes).
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
