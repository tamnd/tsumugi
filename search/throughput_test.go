package search

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// The throughput benchmark (doc 14, "Throughput under concurrency") measures the second
// axis the latency benchmark does not: how many queries per second a box sustains while
// holding the p99 budget. The qps that matters is the qps at which the p99 still fits the
// budget, not peak qps at any latency, because a box that does a hundred thousand queries
// per second at a fifty-millisecond p99 is useless for a ten-millisecond engine. The pieces
// the doc pins are an open-loop load driver that exposes queueing, a throughput-versus-latency
// curve whose knee is the sustained qps at budget, a memory bound that holds under load, a
// deadline that keeps the tail bounded under overload, and a concurrent answer that is exact.
// This file implements all five.

// openLoopResult is the measurement from one open-loop run at a fixed offered rate.
type openLoopResult struct {
	offeredQPS  float64
	issued      int64
	completed   int64
	achievedQPS float64
	p50, p99    time.Duration
	dishonest   int64 // results that violated a Results invariant, must stay zero
}

// driveOpenLoop runs the open-loop load model doc 14 pins: it issues queries from the fixed
// set at the target rate regardless of whether prior queries have returned, rather than a
// closed-loop model that waits for a response before issuing the next. The open-loop model is
// what real traffic looks like and it exposes the queueing a closed-loop driver hides, because
// a closed-loop driver throttles itself to the box's speed and makes a saturated box look fine.
//
// It paces in one-millisecond bursts, launching the fractional per-tick share of the offered
// rate each tick so a high target rate is actually offered rather than capped by the sleep
// granularity. Every query runs under its own per-request deadline, the lever that keeps the
// tail bounded under overload, and each result is checked against the Results invariants so a
// concurrency bug that corrupts the completeness accounting shows as a nonzero dishonest count.
func driveOpenLoop(s Searcher, queries []Query, perReq time.Duration, offeredQPS float64, dur time.Duration) openLoopResult {
	const tick = time.Millisecond
	perTick := offeredQPS * tick.Seconds()

	var (
		issued, completed, dishonest int64
		qi                           int64
		mu                           sync.Mutex
		lat                          []time.Duration
		wg                           sync.WaitGroup
		acc                          float64
	)
	launch := func() {
		wg.Add(1)
		atomic.AddInt64(&issued, 1)
		idx := int(atomic.AddInt64(&qi, 1)-1) % len(queries)
		q := queries[idx]
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), perReq)
			defer cancel()
			t0 := time.Now()
			res := s.SearchComplete(ctx, q)
			d := time.Since(t0)
			// A result must never claim more responding shards than it fanned to, and the
			// Complete predicate must agree with the shard counts. These hold by construction,
			// so a violation under load is a real concurrency bug in the accounting.
			if res.ShardsOK > res.ShardsTotal || res.Complete() != (res.ShardsOK == res.ShardsTotal) {
				atomic.AddInt64(&dishonest, 1)
			}
			mu.Lock()
			lat = append(lat, d)
			mu.Unlock()
			atomic.AddInt64(&completed, 1)
		}()
	}

	start := time.Now()
	for time.Since(start) < dur {
		acc += perTick
		for acc >= 1 {
			launch()
			acc--
		}
		time.Sleep(tick)
	}
	wg.Wait()
	elapsed := time.Since(start)

	return openLoopResult{
		offeredQPS:  offeredQPS,
		issued:      issued,
		completed:   completed,
		achievedQPS: float64(completed) / elapsed.Seconds(),
		p50:         durPercentile(lat, 50),
		p99:         durPercentile(lat, 99),
		dishonest:   dishonest,
	}
}

// durPercentile returns the nearest-rank pth percentile of a duration sample.
func durPercentile(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	c := append([]time.Duration(nil), ds...)
	sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	rank := int(math.Ceil(p/100*float64(len(c)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(c) {
		rank = len(c) - 1
	}
	return c[rank]
}

// hitsIdentical reports whether two hit lists are byte-for-byte equal: same length, same
// docIDs in the same order, same score bits. It is the non-fatal form of assertHitsIdentical,
// for the concurrent comparisons that collect failures rather than failing inline.
func hitsIdentical(a, b []Hit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].DocID != b[i].DocID || math.Float64bits(a[i].Score) != math.Float64bits(b[i].Score) {
			return false
		}
	}
	return true
}

// TestConcurrentAnswerExact is doc 14's exactness-under-load gate: the cascade is
// deterministic given the model and the shard with a docID tie-break, so a query run under
// heavy concurrency returns the same top-k in the same order as the same query run alone.
// It computes a single-threaded reference per query, then hammers the same broker from many
// goroutines and asserts every concurrent result is identical to its reference. Run under
// the race detector this is the load gate on the shared read-only state the query path
// touches: the pooled QuickScorer leaf buffer, the parallel L2 feature extraction, and the
// shard mapping, all of which a corrupting concurrency bug would surface here.
func TestConcurrentAnswerExact(t *testing.T) {
	b, cleanup := buildTextBroker(t)
	defer cleanup()

	queries := []Query{
		{Text: "common document", K: 20},
		{Text: "common", K: 50},
		{Text: "document number term0", K: 30},
		{Text: "common document number", K: 10},
	}
	ref := make([][]Hit, len(queries))
	for i, q := range queries {
		ref[i] = b.Search(context.Background(), q)
		if len(ref[i]) == 0 {
			t.Fatalf("reference query %q returned no hits", q.Text)
		}
	}

	const goroutines, reps = 32, 25
	var wg sync.WaitGroup
	fail := make(chan string, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < reps; r++ {
				for i, q := range queries {
					got := b.Search(context.Background(), q)
					if !hitsIdentical(got, ref[i]) {
						fail <- fmt.Sprintf("query %q diverged under concurrency: serial %d hits, concurrent %d hits", q.Text, len(ref[i]), len(got))
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(fail)
	for msg := range fail {
		t.Fatal(msg)
	}
}

// TestThroughputMemoryBounded is doc 14's memory gate: the resident set under load is
// bounded by the cache cap rather than growing with the query load, because an engine whose
// memory grows unbounded at peak traffic does not serve. It sizes a small result cache, then
// drives far more distinct queries through it concurrently than the cap holds, and asserts the
// cache never exceeds its cap during or after the load: the LRU evicts rather than growing, so
// the bound the cache promises holds under a concurrent miss storm. The heap delta is logged
// for observability, but the deterministic gate is the cache bound itself.
func TestThroughputMemoryBounded(t *testing.T) {
	b, cleanup := buildTextBroker(t)
	defer cleanup()

	const capEntries = 64
	b.SetResultCache(NewResultCache(capEntries))

	// Distinct K values give distinct cache keys (cacheKey folds k in), so this is a stream
	// of distinct queries far exceeding the cap, the miss storm the bound has to survive.
	const distinct = 500
	queries := make([]Query, distinct)
	for k := 0; k < distinct; k++ {
		queries[k] = Query{Text: "common document", K: k + 1}
	}

	var ms0 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms0)

	const goroutines = 24
	var wg sync.WaitGroup
	var over int64
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < distinct; i++ {
				q := queries[(seed+i)%distinct]
				if _, _ = b.SearchCached(context.Background(), q); b.ResultCache().Len() > capEntries {
					atomic.StoreInt64(&over, int64(b.ResultCache().Len()))
				}
			}
		}(g * 7)
	}
	wg.Wait()

	if over > 0 {
		t.Fatalf("cache grew past its cap under load: saw %d entries, cap %d", over, capEntries)
	}
	if got := b.ResultCache().Len(); got > capEntries {
		t.Fatalf("cache settled past its cap: %d entries, cap %d", got, capEntries)
	}

	var ms1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms1)
	t.Logf("cache held at %d/%d entries under %d goroutines x %d distinct queries; heap inuse %.1f -> %.1f MiB",
		b.ResultCache().Len(), capEntries, goroutines, distinct,
		float64(ms0.HeapInuse)/(1<<20), float64(ms1.HeapInuse)/(1<<20))
}

// TestDeadlineUnderOverload is doc 14's graceful-degradation gate: the benchmark drives the
// box past its sustainable rate and checks that under overload it sheds load and stays honest
// rather than collapsing. It offers a rate far above what the tiny synthetic broker can absorb
// under a tight per-request deadline, then asserts the structural invariants that must hold no
// matter how hard the box is pushed: every issued query completes (the box does not hang or
// livelock), no result is dishonest (the completeness accounting stays correct even as shards
// fall off the deadline), and the tail stays under a loose runaway backstop. The point doc 14
// pins is that open-loop overload makes latency climb as the queue grows, the honest picture a
// closed-loop driver would hide, so the per-rung tail is a logged measurement rather than a
// tight threshold; the deterministic proof that the deadline returns a best-effort partial is
// the completeness suite's job (a dropped shard yields an honest incomplete result), and this
// test confirms that property survives concurrent overload.
func TestDeadlineUnderOverload(t *testing.T) {
	b, cleanup := buildTextBroker(t)
	defer cleanup()

	queries := []Query{
		{Text: "common document", K: 20},
		{Text: "common document number", K: 20},
		{Text: "document term0", K: 20},
	}
	const perReq = 5 * time.Millisecond
	const backstop = 5 * time.Second // catches a true livelock, not the expected queueing climb

	r := driveOpenLoop(b, queries, perReq, 20000, 150*time.Millisecond)
	if r.completed == 0 {
		t.Fatal("overload run completed no queries")
	}
	if r.completed != r.issued {
		t.Fatalf("box lost queries under overload: issued %d, completed %d", r.issued, r.completed)
	}
	if r.dishonest != 0 {
		t.Fatalf("%d of %d results violated the completeness invariant under overload", r.dishonest, r.completed)
	}
	if r.p99 > backstop {
		t.Fatalf("p99 %v past the runaway backstop %v (box collapsed rather than queued)", r.p99, backstop)
	}
	t.Logf("overload: offered %.0f qps, achieved %.0f qps over %d queries, p50 %v p99 %v, deadline %v",
		r.offeredQPS, r.achievedQPS, r.completed, r.p50, r.p99, perReq)
}

// TestBrokerThroughputCurveCCrawl drives the real-data throughput-versus-latency curve doc 14
// describes: it ramps the open-loop offered rate across runs, and at each rate it records the
// achieved qps and the p50/p99 latency, so the output is the curve and the operating point is
// the highest offered rate whose p99 still fits the ten-millisecond budget. Like the stage and
// scale benchmarks it gates structural invariants rather than a machine-dependent qps floor:
// every rate must keep its results honest and stay under the catastrophic ceiling, and the
// sustained qps at budget is reported for the record. It skips when the ccrawl sample is
// absent, so it runs on the box that has the data and is a no-op in CI.
func TestBrokerThroughputCurveCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl sample not present: %v", err)
	}
	docs := readRealDocs(t, 2000)
	if len(docs) < 200 {
		t.Skipf("too few real docs: %d", len(docs))
	}
	b, cleanup := buildRealBroker(t, docs, 5)
	defer cleanup()

	queries := realTitleQueries(docs, 25, 32)
	if len(queries) == 0 {
		t.Skip("no usable real-title queries in the sample")
	}

	// Warm the broker so the curve measures steady-state service rather than the cold-start
	// zstd frame decompression a first-touch query pays: the throughput-versus-latency curve
	// is a steady-state property, and an unwarmed run's tail at a low rate is dominated by the
	// one cold query in a small sample, not by queueing.
	for w := 0; w < 3; w++ {
		for _, q := range queries {
			_ = b.Search(context.Background(), q)
		}
	}

	const budget = 10 * time.Millisecond
	const backstop = 5 * time.Second // catches a true livelock, not the expected queueing climb
	const perReq = 10 * time.Millisecond
	rates := []float64{100, 250, 500, 1000, 2000, 4000, 8000, 16000}

	var p99AtBudget, p50AtBudget, peak float64
	for _, rate := range rates {
		r := driveOpenLoop(b, queries, perReq, rate, 250*time.Millisecond)
		if r.completed == 0 {
			continue
		}
		// Honesty and completion are the hard structural gates at every rate; the latency tail
		// climbing past budget at the high rates is the expected open-loop queueing picture, so
		// the per-rung tail is logged and the operating point is the knee where p99 still fits.
		if r.completed != r.issued {
			t.Fatalf("offered %.0f qps: box lost queries, issued %d completed %d", rate, r.issued, r.completed)
		}
		if r.dishonest != 0 {
			t.Fatalf("offered %.0f qps: %d results violated the completeness invariant", rate, r.dishonest)
		}
		if r.p99 > backstop {
			t.Fatalf("offered %.0f qps: p99 %v past the runaway backstop %v", rate, r.p99, backstop)
		}
		if r.achievedQPS > peak {
			peak = r.achievedQPS
		}
		if r.p99 <= budget && r.achievedQPS > p99AtBudget {
			p99AtBudget = r.achievedQPS
		}
		if r.p50 <= budget && r.achievedQPS > p50AtBudget {
			p50AtBudget = r.achievedQPS
		}
		t.Logf("offered %6.0f qps -> achieved %7.1f qps, p50 %10v, p99 %10v%s",
			rate, r.achievedQPS, r.p50, r.p99, map[bool]string{true: " (p99 within budget)", false: ""}[r.p99 <= budget])
	}
	// Three operating points over the same curve: the p99-at-budget knee is the strict gate the
	// spec names; the p50-at-budget rate is the typical-query sustainable rate; the peak is where
	// the achieved rate saturates. On this body-heavy corpus the p99 sits just above the budget,
	// the L2 body-decompression tail slices 75 and 76 measured, so the strict knee is low and the
	// body codec is the lever that would move it; the p50 rate and peak show the box is otherwise
	// fast, with the typical query well under budget.
	t.Logf("over %d shards, %d real-title queries, %v budget: p99-at-budget %.0f qps, p50-at-budget %.0f qps, peak %.0f qps",
		b.NumShards(), len(queries), budget, p99AtBudget, p50AtBudget, peak)
}

// BenchmarkBrokerThroughput measures the achieved queries per second at full concurrency over
// the real crawl, the saturation operating point that complements the curve test's ramp. It
// fans the fixed query set across the box's cores with RunParallel and reports the qps custom
// metric alongside the per-query latency, so a run shows both the per-query cost and the
// aggregate rate the box sustains when every core is busy.
func BenchmarkBrokerThroughput(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl sample not present: %v", err)
	}
	docs := readRealDocs(b, 2000)
	if len(docs) < 200 {
		b.Skipf("too few real docs: %d", len(docs))
	}
	br, cleanup := buildRealBroker(b, docs, 5)
	defer cleanup()

	queries := realTitleQueries(docs, 20, 64)
	if len(queries) == 0 {
		b.Skip("no usable real-title queries in the sample")
	}

	var qi int64
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			q := queries[int(atomic.AddInt64(&qi, 1)-1)%len(queries)]
			if hits := br.Search(context.Background(), q); len(hits) == 0 {
				// A real title should always retrieve its own page; an empty result is a build bug.
				b.Error("real-title query returned no hits")
				return
			}
		}
	})
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
}

// buildRealBroker builds a parts-way broker over the real ccrawl docs, the shared fixture the
// throughput curve test and the throughput benchmark use. It mirrors the parallel-L2 build:
// each shard carries the url/title/body forward store the online L2 features decode, stitched
// into one contiguous docID space by node base.
func buildRealBroker(t testing.TB, docs []textDoc, parts int) (*Broker, func()) {
	t.Helper()
	dir := t.TempDir()
	model := trainExactMatchModel(t)
	size := (len(docs) + parts - 1) / parts
	shards := make([]*Shard, 0, parts)
	for p := 0; p < parts; p++ {
		lo := p * size
		if lo >= len(docs) {
			break
		}
		hi := lo + size
		if hi > len(docs) {
			hi = len(docs)
		}
		path := filepath.Join(dir, fmt.Sprintf("tput%d.tsumugi", p))
		buildTextShardBase(t, path, docs[lo:hi], uint32(lo))
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards = append(shards, sh)
	}
	b := NewBroker(shards, newTestCascade(model))
	return b, func() { _ = b.Close() }
}

// realTitleQueries draws up to max queries from real document titles, so the survivors carry
// real text into the L2 decode the throughput path exercises. It keeps titles of a few tokens,
// the shape a keyword query takes, at the given k.
func realTitleQueries(docs []textDoc, k, max int) []Query {
	var queries []Query
	for i := 0; i < len(docs) && len(queries) < max; i++ {
		toks := uniqueTokens(docs[i].title)
		if len(toks) >= 2 && len(toks) <= 6 {
			queries = append(queries, Query{Text: docs[i].title, K: k})
		}
	}
	return queries
}
