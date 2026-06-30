package search

import (
	"context"
	"time"
)

// A leaf in the serving tree is one machine's shards, and at the 100k-shard target a query
// fans across a deep tree, so the slowest single leaf sets the tail of the whole query: a
// broker that is garbage-collecting, swapping, or sharing a noisy host answers in tens of
// milliseconds while its siblings answer in one, and the aggregator either waits for it past
// the budget or drops it and serves an incomplete top-k. Doc 11 ("retries and hedged
// requests") names the standard answer: run the leaf as a set of identical replicas and, when
// the one being asked is slow, ask a second replica without waiting for the first, taking
// whichever returns a complete answer first and cancelling the loser. A second replica's
// fast answer hides the first replica's slow tail, so the query's p99 tracks the replicas'
// p99-of-the-minimum rather than any single replica's p99.
//
// HedgedSearcher is that replica set as a Searcher. It holds a handful of equivalent replicas
// of one subtree (each a *RemoteSearcher pointing at a different host that serves the same
// shards, or any other Searcher) and a hedge delay. Because it is itself a Searcher, an
// aggregator fans across a HedgedSearcher exactly as it fans across a single broker or a
// RemoteSearcher, so hedging slots into the existing tree with no change to the aggregator:
// a child that was one RemoteSearcher becomes a HedgedSearcher over that peer's replicas and
// nothing above it knows the difference.
//
// The replicas must be equivalent: the same shards, the same fleet statistics, so any one of
// them answers a query the same way and the static methods (NumShards, NumDocs, Stats) can be
// answered from the first replica without polling them all. They are not load-balanced copies
// of different data; they are redundant copies of the same data, which is what makes taking
// the first complete answer correct rather than a partial.
type HedgedSearcher struct {
	replicas []Searcher

	// delay is how long to wait for the replica already in flight before sending the query to
	// the next replica too. It is the knob that trades duplicate work for tail latency: a delay
	// near the replicas' median latency hedges only the slow tail (most queries finish on the
	// first replica and never launch a second, so the duplicate-request rate is small), while a
	// delay at or below zero hedges immediately (the query goes to every replica at once and the
	// first complete answer wins, the lowest-latency, highest-cost extreme). A deployment sets it
	// from the measured median so the hedge rate stays a few percent.
	delay time.Duration
}

// NewHedgedSearcher builds a Searcher over a set of equivalent replicas of one subtree, each
// answering the same query the same way, with the given hedge delay. With a single replica it
// is a thin pass-through (there is nothing to hedge to), so a deployment can build every child
// as a HedgedSearcher and pay the hedging machinery only where a replica set actually exists.
// An empty replica set answers every query as a fully-missed subtree, the same honest empty a
// dropped child reports, rather than panicking.
func NewHedgedSearcher(replicas []Searcher, delay time.Duration) *HedgedSearcher {
	return &HedgedSearcher{replicas: replicas, delay: delay}
}

// NumReplicas is the count of equivalent replicas the hedger fans a slow query across.
func (h *HedgedSearcher) NumReplicas() int { return len(h.replicas) }

// NumShards reports the leaf shard count beneath the subtree, read from the first replica
// since the replicas are equivalent copies of the same shards.
func (h *HedgedSearcher) NumShards() int {
	if len(h.replicas) == 0 {
		return 0
	}
	return h.replicas[0].NumShards()
}

// NumDocs reports the document count beneath the subtree from the first replica, the N a
// parent sums into the fleet total it divides one shared idf by.
func (h *HedgedSearcher) NumDocs() uint64 {
	if len(h.replicas) == 0 {
		return 0
	}
	return h.replicas[0].NumDocs()
}

// Stats reports the subtree's fleet statistics from the first replica, which a parent folds
// into the deployment-wide field averages it pushes back down.
func (h *HedgedSearcher) Stats() GlobalStats {
	if len(h.replicas) == 0 {
		return GlobalStats{}
	}
	return h.replicas[0].Stats()
}

// VectorDim reports the dense input dimension the subtree agrees on, delegating to the first
// replica when it can report one (a broker or a RemoteSearcher), so a HedgedSearcher placed
// behind NewSearcherHandler still answers /meta's dense dimension. A replica that reports no
// dimension leaves the dense plane off.
func (h *HedgedSearcher) VectorDim() (int, bool) {
	if len(h.replicas) == 0 {
		return 0, false
	}
	if vd, ok := h.replicas[0].(vectorDimer); ok {
		return vd.VectorDim()
	}
	return 0, false
}

// ForEachTerm streams the subtree's vocabulary from the first replica when it can enumerate
// one, so a HedgedSearcher behind NewSearcherHandler serves /vocab the same way the replica
// it wraps does. A replica that holds no vocabulary of its own enumerates nothing.
func (h *HedgedSearcher) ForEachTerm(fn func(term string, df uint32)) {
	if len(h.replicas) == 0 {
		return
	}
	if vi, ok := h.replicas[0].(vocabIterator); ok {
		vi.ForEachTerm(fn)
	}
}

// DocFreqs gathers each term's document frequency from the replicas, trying them in order and
// returning the first non-empty answer. The replicas are equivalent, so one replica's df is
// the whole subtree's df; failing over to the next replica only matters when the first is
// unreachable, and a df gather that returns nothing only loosens the idf the parent computes
// rather than corrupting a score, so this stops at the first replica that answers rather than
// hedging in parallel the way the search does.
func (h *HedgedSearcher) DocFreqs(ctx context.Context, terms []string) map[string]uint32 {
	if len(terms) == 0 {
		return nil
	}
	for _, r := range h.replicas {
		if df := r.DocFreqs(ctx, terms); len(df) > 0 {
			return df
		}
		if ctx.Err() != nil {
			return nil
		}
	}
	return nil
}

// SearchComplete runs the query against the replica set, hedging against a slow replica. It
// sends the query to the first replica and starts a timer; if a complete answer comes back
// first it returns immediately, and if the timer fires first it sends the query to the next
// replica too without cancelling the first, so the query rides whichever replica is fast on
// this attempt. The first complete result wins and the still-running replicas are cancelled
// on return; if every replica answers but none is complete (each dropped some shard at the
// deadline) the best partial, the one that reached the most shards, is returned, so hedging
// never loses an answer it would otherwise have had. A replica that returns an incomplete
// answer also triggers the next replica early, the failover half of doc 11's "retries and
// hedged requests", so a replica whose subtree is down does not have to wait out the timer to
// hand off to a healthy one.
func (h *HedgedSearcher) SearchComplete(ctx context.Context, q Query) Results {
	switch len(h.replicas) {
	case 0:
		return Results{}
	case 1:
		return h.replicas[0].SearchComplete(ctx, q)
	}

	// Derive a cancellable context so returning the winner cancels every replica still in
	// flight, which propagates down a RemoteSearcher as a cancelled HTTP request so a loser
	// stops doing work the moment a winner is chosen.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// The channel is buffered to the full replica count so a replica that finishes after a
	// winner has been chosen still sends its result and its goroutine exits rather than
	// blocking forever on an unread channel.
	out := make(chan Results, len(h.replicas))
	next := 0
	launch := func() {
		i := next
		next++
		go func() { out <- h.replicas[i].SearchComplete(ctx, q) }()
	}
	launch() // the primary replica
	pending := 1

	timer := time.NewTimer(h.delay)
	defer timer.Stop()

	var best Results
	haveBest := false
	for {
		select {
		case r := <-out:
			pending--
			if r.Complete() {
				return r
			}
			if !haveBest || r.ShardsOK > best.ShardsOK {
				best, haveBest = r, true
			}
			// This replica came back incomplete; bring in the next one to try for a complete
			// answer rather than settling for the partial. When the replicas are exhausted and
			// nothing is still in flight, the best partial is the most this set could do.
			if next < len(h.replicas) {
				launch()
				pending++
			} else if pending == 0 {
				return best
			}
		case <-timer.C:
			// The replica in flight is slower than the hedge delay, so send the query to the
			// next replica too without waiting for the first; whichever returns a complete
			// answer first wins. The timer rearms only while replicas remain, so it escalates
			// one replica per delay and then stops.
			if next < len(h.replicas) {
				launch()
				pending++
			}
			if next < len(h.replicas) {
				timer.Reset(h.delay)
			}
		case <-ctx.Done():
			// The deadline passed: return the best partial gathered so far, or a fully-missed
			// subtree if nothing came back, the honest upper bound a parent rolls up as
			// incomplete rather than a hole passed off as complete.
			if haveBest {
				return best
			}
			return Results{ShardsTotal: h.replicas[0].NumShards()}
		}
	}
}
