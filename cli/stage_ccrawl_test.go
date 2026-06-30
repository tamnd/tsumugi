package cli

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/search"
)

// durPercentile is the nearest-rank percentile of an already-sorted duration slice,
// p in [0,1]. It is the same reading the scale benchmark reports latency with: p99 is
// the tail the latency budget is set against, not the mean a slow tail hides behind.
func durPercentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// knownItemQueries builds a sample of known-item queries from a built collection's
// distinctive titles, the same selection the quality harness uses: a title that
// parses to at least a few content terms, taken at most once, is the query a searcher
// who remembered that page would type. It returns the parsed search queries ready to
// run, the shared front-end work the breakdown does not want to time.
func knownItemQueries(t *testing.T, dir string, pl *pipeline, docs, want, minTerms, k int) []search.Query {
	t.Helper()
	titles := readTitles(t, dir)
	queries := make([]search.Query, 0, want)
	seen := map[string]struct{}{}
	for id := uint32(0); id < uint32(docs) && len(queries) < want; id++ {
		title := titles[id]
		if title == "" {
			continue
		}
		if _, dup := seen[title]; dup {
			continue
		}
		pq := pl.parse(title)
		if pq.Empty() {
			continue
		}
		q := toQuery(pq, k)
		if len(q.Terms) < minTerms {
			continue
		}
		seen[title] = struct{}{}
		queries = append(queries, q)
	}
	return queries
}

// TestStageBreakdownCCrawl is the per-stage latency breakdown doc 14 pins (L233-252):
// it decomposes a query's cascade time into the stages of doc 14's budget table,
// lexical retrieval, dense recall, fusion, the linear cut, the L2 feature extraction,
// and the L2 model scoring, and reports each stage's p50/p95/p99 alongside the
// end-to-end p99 over the real ccrawl corpus. The end-to-end number is the sum of the
// stages, so the breakdown says which stage a query's budget is spent in, the reading
// an optimization is aimed by. The decomposition is of the production pipeline, not a
// parallel one: SearchStaged returns the same hits Search does, which the test asserts
// before it trusts the timings.
func TestStageBreakdownCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards to spread the breakdown, got %d", res.Shards)
	}

	modelPath := filepath.Join(tmp, "model.bin")
	writeModel(t, modelPath)
	model, err := loadModel(modelPath)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	shards, _, broker, err := openShards(out, model)
	if err != nil {
		t.Fatalf("open shards: %v", err)
	}
	defer func() { _ = broker.Close() }()

	const (
		wantQueries = 80
		minTerms    = 3
		k           = 100
		reps        = 3
	)
	queries := knownItemQueries(t, out, buildPipeline(shards), res.Docs, wantQueries, minTerms, k)
	if len(queries) < 20 {
		t.Fatalf("only %d known-item queries built from %d docs; need a richer corpus to measure", len(queries), res.Docs)
	}

	// Faithfulness: the staged decomposition must produce exactly what the production
	// Search produces, or the per-stage times describe a different pipeline. Check it on
	// the first query against the first shard before timing anything.
	staged, _ := shards[0].SearchStaged(queries[0])
	plain := shards[0].Search(queries[0])
	if len(staged) != len(plain) {
		t.Fatalf("staged returned %d hits, Search %d", len(staged), len(plain))
	}
	for i := range staged {
		if staged[i] != plain[i] {
			t.Fatalf("staged hit %d = %+v, Search %+v; the breakdown is not of the production path", i, staged[i], plain[i])
		}
	}

	// Each (query, shard) cascade is one sample. The broker's global rerank reranks
	// L1Keep survivors once over the merged candidate set, the same survivor count a
	// single shard's cascade reranks, so a per-shard cascade's stage shape stands in for
	// the broker rerank's stage shape while the L0 planes run per shard in parallel. The
	// distribution over every query against every shard is what the percentiles read,
	// and a few repetitions warm the file cache and fill the tail. Alongside each
	// staged decomposition the loop records the externally-timed wall clock of the same
	// call, the ground truth the instrumented Total is checked against so an
	// uninstrumented gap cannot hide between the stages.
	type sample struct {
		t    search.StageTimings
		wall time.Duration
	}
	samples := make([]sample, 0, len(queries)*len(shards)*reps)
	for r := 0; r < reps; r++ {
		for _, q := range queries {
			for _, sh := range shards {
				start := time.Now()
				_, st := sh.SearchStaged(q)
				wall := time.Since(start)
				samples = append(samples, sample{t: st, wall: wall})
			}
		}
	}

	// Pull each stage into its own sorted slice and read p50/p95/p99 off it, plus the
	// end-to-end total per sample.
	collect := func(get func(search.StageTimings) time.Duration) []time.Duration {
		ds := make([]time.Duration, len(samples))
		for i, s := range samples {
			ds[i] = get(s.t)
		}
		sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
		return ds
	}
	stages := []struct {
		name string
		get  func(search.StageTimings) time.Duration
	}{
		{"L0 lexical+sparse", func(s search.StageTimings) time.Duration { return s.L0Lexical }},
		{"L0 dense", func(s search.StageTimings) time.Duration { return s.L0Dense }},
		{"fuse", func(s search.StageTimings) time.Duration { return s.Fuse }},
		{"L1 cut", func(s search.StageTimings) time.Duration { return s.L1Cut }},
		{"L2 feature", func(s search.StageTimings) time.Duration { return s.L2Feature }},
		{"L2 score", func(s search.StageTimings) time.Duration { return s.L2Score }},
	}

	t.Logf("per-stage latency over %d samples (%d queries x %d shards x %d reps), %d docs:",
		len(samples), len(queries), len(shards), reps, res.Docs)
	for _, st := range stages {
		ds := collect(st.get)
		t.Logf("  %-18s p50 %8v  p95 %8v  p99 %8v", st.name, durPercentile(ds, 0.50), durPercentile(ds, 0.95), durPercentile(ds, 0.99))
	}
	total := collect(search.StageTimings.Total)
	p50, p95, p99 := durPercentile(total, 0.50), durPercentile(total, 0.95), durPercentile(total, 0.99)
	t.Logf("  %-18s p50 %8v  p95 %8v  p99 %8v", "end-to-end (sum)", p50, p95, p99)

	// Identify the stage the budget is spent in, the reading an optimization is aimed by.
	type stageP50 struct {
		name string
		p50  time.Duration
	}
	dominant := stageP50{}
	for _, st := range stages {
		ds := collect(st.get)
		if p := durPercentile(ds, 0.50); p > dominant.p50 {
			dominant = stageP50{st.name, p}
		}
	}
	t.Logf("dominant stage at p50: %s (%v of %v end-to-end)", dominant.name, dominant.p50, p50)

	// Absolute latency is machine-dependent, so like the scale test this reports the
	// numbers and gates structural invariants rather than a wall-clock threshold a slow
	// CI box would flake on. A catastrophic-regression ceiling well above any real run
	// still catches a stage that blows up by an order of magnitude.
	const ceiling = 250 * time.Millisecond
	if p99 > ceiling {
		t.Fatalf("end-to-end p99 %v exceeds the catastrophic-regression ceiling %v; a stage has blown up", p99, ceiling)
	}
	// The decomposition must account for real work: the end-to-end p50 is nonzero, so the
	// per-stage timers are measuring the cascade rather than reporting empty timers.
	if p50 <= 0 {
		t.Fatalf("end-to-end p50 is zero; the per-stage timers captured no work")
	}

	// Instrumentation completeness: the instrumented stage sum must account for nearly
	// all of the externally-timed wall clock, or time is being spent in a stage the
	// breakdown does not name. Compare the two p50s and require the stage sum to cover at
	// least 80% of the wall clock; the small remainder is the slice and map bookkeeping
	// between stages, which is real but not a stage. This is what makes the breakdown
	// trustworthy: every meaningful millisecond is attributed to a named stage.
	wallTotal := make([]time.Duration, len(samples))
	for i, s := range samples {
		wallTotal[i] = s.wall
	}
	sort.Slice(wallTotal, func(i, j int) bool { return wallTotal[i] < wallTotal[j] })
	wallP50 := durPercentile(wallTotal, 0.50)
	if wallP50 > 0 && p50 < wallP50*8/10 {
		t.Fatalf("instrumented stage sum p50 %v accounts for less than 80%% of the wall-clock p50 %v; an uninstrumented stage is hiding", p50, wallP50)
	}
}
