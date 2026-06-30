package search

import (
	"context"
	"math"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// fakeSearcher is a controllable Searcher for the hedging unit tests: it answers SearchComplete
// after an optional delay (cancellable, so a cancelled context returns a fully-missed subtree and
// records that it saw the cancel), counts its calls, and reports static metadata from its fields,
// so a test can stand up replicas with engineered latency and completeness and assert exactly
// which replicas ran and which won.
type fakeSearcher struct {
	res    Results
	delay  time.Duration
	df     map[string]uint32
	shards int
	docs   uint64
	stats  GlobalStats
	dim    int
	hasDim bool

	calls     atomic.Int32
	sawCancel atomic.Bool
}

func (f *fakeSearcher) SearchComplete(ctx context.Context, _ Query) Results {
	f.calls.Add(1)
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			f.sawCancel.Store(true)
			return Results{ShardsTotal: f.shards}
		}
	}
	return f.res
}

func (f *fakeSearcher) NumShards() int     { return f.shards }
func (f *fakeSearcher) NumDocs() uint64    { return f.docs }
func (f *fakeSearcher) Stats() GlobalStats { return f.stats }
func (f *fakeSearcher) DocFreqs(_ context.Context, terms []string) map[string]uint32 {
	if len(terms) == 0 {
		return nil
	}
	return f.df
}
func (f *fakeSearcher) VectorDim() (int, bool) { return f.dim, f.hasDim }
func (f *fakeSearcher) ForEachTerm(fn func(term string, df uint32)) {
	for t, d := range f.df {
		fn(t, d)
	}
}

// complete builds a complete Results carrying one identifiable hit, so a test can tell which
// replica's answer was returned.
func complete(id uint32, shards int) Results {
	return Results{Hits: []Hit{{DocID: id, Score: 1}}, ShardsTotal: shards, ShardsOK: shards}
}

// partial builds an incomplete Results that reached ok of shards, carrying one identifiable hit.
func partial(id uint32, ok, shards int) Results {
	return Results{Hits: []Hit{{DocID: id, Score: 1}}, ShardsTotal: shards, ShardsOK: ok}
}

func firstHit(r Results) uint32 {
	if len(r.Hits) == 0 {
		return 0
	}
	return r.Hits[0].DocID
}

// TestHedgedIsSearcher pins the decorator to the Searcher contract so a HedgedSearcher slots
// into an aggregator exactly where a single broker or RemoteSearcher would.
func TestHedgedIsSearcher(t *testing.T) {
	var _ Searcher = NewHedgedSearcher(nil, 0)
}

// TestHedgedSingleReplicaPassThrough checks that with one replica the hedger is a thin
// pass-through: it runs the one replica once and returns its answer, paying none of the timer
// or goroutine machinery, so building every child as a HedgedSearcher costs nothing where no
// replica set exists.
func TestHedgedSingleReplicaPassThrough(t *testing.T) {
	r := &fakeSearcher{res: complete(7, 3), shards: 3}
	h := NewHedgedSearcher([]Searcher{r}, time.Millisecond)
	got := h.SearchComplete(context.Background(), Query{})
	if firstHit(got) != 7 || !got.Complete() {
		t.Fatalf("single-replica passthrough returned %+v", got)
	}
	if r.calls.Load() != 1 {
		t.Fatalf("single replica called %d times, want 1", r.calls.Load())
	}
}

// TestHedgedFastReplicaWins is the tail-latency proof: a slow primary and a fast secondary with
// a short hedge delay returns the fast secondary's answer well before the slow primary would have
// returned, and both replicas were dispatched (the timer fired and launched the second).
func TestHedgedFastReplicaWins(t *testing.T) {
	slow := &fakeSearcher{res: complete(1, 4), shards: 4, delay: 300 * time.Millisecond}
	fast := &fakeSearcher{res: complete(2, 4), shards: 4, delay: 2 * time.Millisecond}
	h := NewHedgedSearcher([]Searcher{slow, fast}, 5*time.Millisecond)

	start := time.Now()
	got := h.SearchComplete(context.Background(), Query{})
	elapsed := time.Since(start)

	if firstHit(got) != 2 {
		t.Fatalf("hedge returned %d, want the fast replica's 2", firstHit(got))
	}
	if elapsed > 100*time.Millisecond {
		t.Fatalf("hedge took %v, should have returned on the fast replica well under the slow 300ms", elapsed)
	}
	if slow.calls.Load() != 1 || fast.calls.Load() != 1 {
		t.Fatalf("dispatch counts slow=%d fast=%d, want both 1", slow.calls.Load(), fast.calls.Load())
	}
}

// TestHedgedNoHedgeWhenPrimaryFast checks the common case: the primary answers before the hedge
// delay, so the secondary is never launched. This is what keeps the duplicate-request rate small,
// the whole reason the delay is tuned to the median rather than hedging every query.
func TestHedgedNoHedgeWhenPrimaryFast(t *testing.T) {
	primary := &fakeSearcher{res: complete(1, 4), shards: 4, delay: 1 * time.Millisecond}
	backup := &fakeSearcher{res: complete(2, 4), shards: 4, delay: 1 * time.Millisecond}
	h := NewHedgedSearcher([]Searcher{primary, backup}, 100*time.Millisecond)

	got := h.SearchComplete(context.Background(), Query{})
	if firstHit(got) != 1 {
		t.Fatalf("hedge returned %d, want the primary's 1", firstHit(got))
	}
	if backup.calls.Load() != 0 {
		t.Fatalf("backup replica was launched %d times though the primary beat the hedge delay", backup.calls.Load())
	}
}

// TestHedgedFailoverOnIncomplete checks the failover half of "retries and hedged requests": a
// replica that returns an incomplete answer triggers the next replica early, without waiting out
// the hedge timer, so a replica whose subtree is down hands off to a healthy one immediately. The
// hedge delay is long here, so the only thing that can launch the second replica is the first's
// incomplete answer.
func TestHedgedFailoverOnIncomplete(t *testing.T) {
	down := &fakeSearcher{res: partial(1, 0, 4), shards: 4} // returns immediately, incomplete
	healthy := &fakeSearcher{res: complete(2, 4), shards: 4, delay: 2 * time.Millisecond}
	h := NewHedgedSearcher([]Searcher{down, healthy}, time.Hour)

	got := h.SearchComplete(context.Background(), Query{})
	if !got.Complete() || firstHit(got) != 2 {
		t.Fatalf("failover returned %+v, want the healthy replica's complete answer", got)
	}
	if healthy.calls.Load() != 1 {
		t.Fatalf("healthy replica was not failed over to (calls=%d)", healthy.calls.Load())
	}
}

// TestHedgedBestPartialWhenAllIncomplete checks that when every replica answers but none is
// complete, the hedger returns the best partial, the one that reached the most shards, rather than
// the first or an empty one, so hedging never loses an answer it would otherwise have had.
func TestHedgedBestPartialWhenAllIncomplete(t *testing.T) {
	worse := &fakeSearcher{res: partial(1, 1, 4), shards: 4}
	better := &fakeSearcher{res: partial(2, 3, 4), shards: 4}
	h := NewHedgedSearcher([]Searcher{worse, better}, time.Hour)

	got := h.SearchComplete(context.Background(), Query{})
	if got.Complete() {
		t.Fatalf("all-incomplete returned complete: %+v", got)
	}
	if got.ShardsOK != 3 || firstHit(got) != 2 {
		t.Fatalf("best-partial returned ok=%d id=%d, want the better partial 3/2", got.ShardsOK, firstHit(got))
	}
}

// TestHedgedDeadlineReturnsMissed checks that when the deadline passes with no replica having
// answered, the hedger returns a fully-missed subtree (its whole shard count charged to total,
// nothing reached), the honest upper bound a parent rolls up as incomplete rather than a hole
// passed off as complete, and that the in-flight replicas saw the cancellation.
func TestHedgedDeadlineReturnsMissed(t *testing.T) {
	a := &fakeSearcher{res: complete(1, 4), shards: 4, delay: time.Hour}
	b := &fakeSearcher{res: complete(2, 4), shards: 4, delay: time.Hour}
	h := NewHedgedSearcher([]Searcher{a, b}, 2*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	got := h.SearchComplete(ctx, Query{})
	if got.Complete() || got.ShardsOK != 0 || got.ShardsTotal != 4 {
		t.Fatalf("deadline returned %+v, want a fully-missed 0/4 subtree", got)
	}
	if !waitFor(func() bool { return a.sawCancel.Load() && b.sawCancel.Load() }, time.Second) {
		t.Fatal("a dispatched replica did not see the deadline cancellation")
	}
}

// TestHedgedCancelsLosers checks that once a winner is chosen the still-running replicas are
// cancelled, so a loser stops doing work (and, over a RemoteSearcher, cancels its HTTP request)
// the moment a winner returns rather than running to completion off the query's critical path.
func TestHedgedCancelsLosers(t *testing.T) {
	winner := &fakeSearcher{res: complete(2, 4), shards: 4, delay: 3 * time.Millisecond}
	loser := &fakeSearcher{res: complete(1, 4), shards: 4, delay: time.Hour}
	// Primary is the loser (slow); the hedge fires fast and the winner returns, after which the
	// loser must be cancelled.
	h := NewHedgedSearcher([]Searcher{loser, winner}, 1*time.Millisecond)

	got := h.SearchComplete(context.Background(), Query{})
	if firstHit(got) != 2 {
		t.Fatalf("winner should have been the fast replica, got %d", firstHit(got))
	}
	if !waitFor(loser.sawCancel.Load, time.Second) {
		t.Fatal("the losing replica was not cancelled after a winner was chosen")
	}
}

// TestHedgedDocFreqsFailover checks DocFreqs fails over to the next replica when the first returns
// nothing (an unreachable replica), since the replicas are equivalent so one replica's df is the
// whole subtree's, and a df gather that returns nothing only loosens an idf.
func TestHedgedDocFreqsFailover(t *testing.T) {
	dead := &fakeSearcher{df: nil, shards: 4}
	live := &fakeSearcher{df: map[string]uint32{"x": 5}, shards: 4}
	h := NewHedgedSearcher([]Searcher{dead, live}, time.Millisecond)

	df := h.DocFreqs(context.Background(), []string{"x"})
	if df["x"] != 5 {
		t.Fatalf("docfreqs failover returned %v, want x=5 from the live replica", df)
	}
	if df := h.DocFreqs(context.Background(), nil); df != nil {
		t.Fatalf("docfreqs of no terms returned %v, want nil", df)
	}
}

// TestHedgedStaticMethods checks the static metadata is answered from the first replica, since
// the replicas are equivalent copies of the same shards, including the optional VectorDim and
// ForEachTerm a HedgedSearcher delegates so it stays a drop-in behind NewSearcherHandler.
func TestHedgedStaticMethods(t *testing.T) {
	first := &fakeSearcher{
		shards: 5, docs: 100, dim: 8, hasDim: true,
		stats: GlobalStats{DocCount: 100, AvgDocLen: 12},
		df:    map[string]uint32{"a": 1, "b": 2},
	}
	second := &fakeSearcher{shards: 99, docs: 9999, dim: 64, hasDim: true}
	h := NewHedgedSearcher([]Searcher{first, second}, time.Millisecond)

	if h.NumShards() != 5 || h.NumDocs() != 100 || h.Stats().DocCount != 100 {
		t.Fatalf("static methods did not delegate to the first replica: shards=%d docs=%d stats=%+v",
			h.NumShards(), h.NumDocs(), h.Stats())
	}
	if dim, ok := h.VectorDim(); !ok || dim != 8 {
		t.Fatalf("VectorDim = (%d,%t), want (8,true) from the first replica", dim, ok)
	}
	if h.NumReplicas() != 2 {
		t.Fatalf("NumReplicas = %d, want 2", h.NumReplicas())
	}
	vocab := map[string]uint32{}
	h.ForEachTerm(func(term string, df uint32) { vocab[term] = df })
	if vocab["a"] != 1 || vocab["b"] != 2 || len(vocab) != 2 {
		t.Fatalf("ForEachTerm did not delegate the first replica's vocabulary: %v", vocab)
	}
}

// TestHedgedEmptyReplicaSet checks an empty set answers as a fully-missed subtree rather than
// panicking, so a misconfigured child degrades honestly.
func TestHedgedEmptyReplicaSet(t *testing.T) {
	h := NewHedgedSearcher(nil, time.Millisecond)
	if h.NumShards() != 0 || h.NumDocs() != 0 {
		t.Fatalf("empty set reported shards=%d docs=%d, want 0/0", h.NumShards(), h.NumDocs())
	}
	got := h.SearchComplete(context.Background(), Query{})
	if len(got.Hits) != 0 || got.ShardsTotal != 0 {
		t.Fatalf("empty set returned %+v, want an empty result", got)
	}
	if dim, ok := h.VectorDim(); ok || dim != 0 {
		t.Fatalf("empty set VectorDim = (%d,%t), want (0,false)", dim, ok)
	}
}

// delayedSearcher wraps a Searcher and adds a fixed latency to SearchComplete, respecting the
// context so a cancelled hedge loser stops waiting, the test stand-in for a replica on a slow or
// noisy host.
type delayedSearcher struct {
	Searcher
	delay time.Duration
}

func (d *delayedSearcher) SearchComplete(ctx context.Context, q Query) Results {
	select {
	case <-time.After(d.delay):
	case <-ctx.Done():
		return Results{ShardsTotal: d.NumShards()}
	}
	return d.Searcher.SearchComplete(ctx, q)
}

// TestHedgedOverRealBrokersMatchesAndHidesTail is the correctness-plus-latency proof on real
// brokers: a HedgedSearcher over two equivalent replicas of a broker, one of them wrapped to be
// slow, returns the exact answer the broker itself returns (so hedging changes latency, not
// results) and returns it well under the slow replica's latency (so the fast replica hid the
// tail). It runs over the wire, two httptest servers in front of the same broker.
func TestHedgedOverRealBrokersMatchesAndHidesTail(t *testing.T) {
	const n, parts = 120, 3
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	ctx := context.Background()
	q := Query{Terms: []string{"alpha", "common"}, K: 15}
	want := broker.SearchComplete(ctx, q)

	// Two equivalent replicas dialed over HTTP at the same broker; the primary is wrapped slow so
	// the hedge must fall to the fast secondary.
	slow := &delayedSearcher{Searcher: serveSearcher(t, broker), delay: 300 * time.Millisecond}
	fast := serveSearcher(t, broker)
	h := NewHedgedSearcher([]Searcher{slow, fast}, 5*time.Millisecond)

	start := time.Now()
	got := h.SearchComplete(ctx, q)
	elapsed := time.Since(start)

	if elapsed > 150*time.Millisecond {
		t.Fatalf("hedge took %v, the fast replica should have hidden the slow 300ms tail", elapsed)
	}
	if !got.Complete() {
		t.Fatalf("hedge returned incomplete: %d/%d", got.ShardsOK, got.ShardsTotal)
	}
	if len(got.Hits) != len(want.Hits) {
		t.Fatalf("hedge returned %d hits, broker %d", len(got.Hits), len(want.Hits))
	}
	for i := range want.Hits {
		if got.Hits[i].DocID != want.Hits[i].DocID || math.Abs(got.Hits[i].Score-want.Hits[i].Score) > 1e-9 {
			t.Fatalf("hit %d: hedge {%d,%v}, broker {%d,%v}", i,
				got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].DocID, want.Hits[i].Score)
		}
	}
	t.Logf("hedged over a 300ms-slow and a fast replica: returned the broker's exact %d-hit top-k in %v",
		len(got.Hits), elapsed)
}

// TestAggregatorOverHedgedMatchesMonolith proves hedging composes into the serving tree: an
// aggregator whose children are HedgedSearchers, each over two equivalent remote replicas of a
// broker, reproduces a single broker over every shard exactly, so wrapping a child in a replica
// set for tail tolerance does not change the fleet-wide top-k. One replica per child is slowed so
// the hedge path is exercised on the way to the exact answer.
func TestAggregatorOverHedgedMatchesMonolith(t *testing.T) {
	const n = 200
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	monoBroker := NewBroker([]*Shard{ms}, newTestCascade(model))
	defer func() { _ = monoBroker.Close() }()

	const parts = 4
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, "s"+string(rune('0'+p))+".tsumugi")
		lo := p * size
		buildShardFile(t, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()
	b0 := NewBroker(shards[:2], newTestCascade(model))
	b1 := NewBroker(shards[2:], newTestCascade(model))

	// Each child is a replica set: a slow primary and a fast secondary over the same broker, so the
	// aggregator fans across two hedged children and the hedge falls to the fast replica each time.
	child0 := NewHedgedSearcher([]Searcher{
		&delayedSearcher{Searcher: serveSearcher(t, b0), delay: 200 * time.Millisecond},
		serveSearcher(t, b0),
	}, 3*time.Millisecond)
	child1 := NewHedgedSearcher([]Searcher{
		&delayedSearcher{Searcher: serveSearcher(t, b1), delay: 200 * time.Millisecond},
		serveSearcher(t, b1),
	}, 3*time.Millisecond)
	agg := NewAggregator([]Searcher{child0, child1})

	ctx := context.Background()
	if agg.NumShards() != parts || agg.NumDocs() != monoBroker.NumDocs() {
		t.Fatalf("tree shape shards=%d docs=%d, want %d/%d", agg.NumShards(), agg.NumDocs(), parts, monoBroker.NumDocs())
	}

	nontrivial := 0
	for _, q := range []Query{
		{Terms: []string{"common"}, K: 10},
		{Terms: []string{"alpha", "common"}, K: 20},
		{Terms: []string{"beta", "common"}, K: 20},
		{Terms: []string{"alpha", "beta", "common"}, K: 25},
	} {
		want := monoBroker.SearchComplete(ctx, q)
		got := agg.SearchComplete(ctx, q)
		if !got.Complete() {
			t.Fatalf("query %v over hedged tree was not complete: %d/%d", q.Terms, got.ShardsOK, got.ShardsTotal)
		}
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("query %v: hedged tree returned %d hits, monolith %d", q.Terms, len(got.Hits), len(want.Hits))
		}
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID || math.Abs(got.Hits[i].Score-want.Hits[i].Score) > 1e-6 {
				t.Fatalf("query %v hit %d: hedged tree {%d,%v}, monolith {%d,%v}", q.Terms, i,
					got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].DocID, want.Hits[i].Score)
			}
		}
		if len(got.Hits) > 1 && got.Hits[0].Score != got.Hits[len(got.Hits)-1].Score {
			nontrivial++
		}
	}
	if nontrivial == 0 {
		t.Fatal("every query produced a flat ranking; the corpus or model is not exercising the merge")
	}
	t.Logf("aggregator over hedged replica sets reproduced the monolith on %d queries (%d non-trivial)", 4, nontrivial)
}

// waitFor polls cond until it is true or the timeout elapses, returning whether it became true. It
// is the test's way to wait on a background replica goroutine observing a cancellation without
// racing on a fixed sleep.
func waitFor(cond func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(time.Millisecond)
	}
	return cond()
}
