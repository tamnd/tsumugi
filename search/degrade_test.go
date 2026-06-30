package search

import (
	"context"
	"sort"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/rank"
)

// TestDegradeLevelString checks each rung names itself for the metadata the serve path
// reports, so an operator sees which degradation fired.
func TestDegradeLevelString(t *testing.T) {
	cases := map[DegradeLevel]string{
		DegradeNone:   "none",
		DegradeL0:     "shrink-l0",
		DegradeDense:  "drop-dense",
		DegradeShards: "drop-shards",
		DegradeL2:     "trim-l2",
	}
	for level, want := range cases {
		if got := level.String(); got != want {
			t.Fatalf("DegradeLevel(%d).String() = %q, want %q", level, got, want)
		}
	}
}

// TestDegradationForLadder checks the ladder is cumulative in the fixed order: each
// rung turns on its own lever and keeps every lever of the rungs below it, so a higher
// rung is strictly more degraded than a lower one, the spec's least-cost-first order.
func TestDegradationForLadder(t *testing.T) {
	none := degradationFor(DegradeNone)
	if none.l0 != 0 || none.dropDense || none.dropShardFrac != 0 || none.l1Keep != 0 {
		t.Fatalf("DegradeNone turned on a lever: %+v", none)
	}

	l0 := degradationFor(DegradeL0)
	if l0.l0 != DefaultL0/4 {
		t.Fatalf("DegradeL0 l0 = %d, want %d", l0.l0, DefaultL0/4)
	}
	if l0.dropDense || l0.dropShardFrac != 0 || l0.l1Keep != 0 {
		t.Fatalf("DegradeL0 turned on a higher lever: %+v", l0)
	}

	dense := degradationFor(DegradeDense)
	if dense.l0 != DefaultL0/4 || !dense.dropDense {
		t.Fatalf("DegradeDense did not keep L0 and add dense drop: %+v", dense)
	}
	if dense.dropShardFrac != 0 || dense.l1Keep != 0 {
		t.Fatalf("DegradeDense turned on a higher lever: %+v", dense)
	}

	shards := degradationFor(DegradeShards)
	if shards.l0 != DefaultL0/4 || !shards.dropDense || shards.dropShardFrac != 0.25 {
		t.Fatalf("DegradeShards did not keep lower levers and add shard drop: %+v", shards)
	}
	if shards.l1Keep != 0 {
		t.Fatalf("DegradeShards turned on the L2 trim: %+v", shards)
	}

	l2 := degradationFor(DegradeL2)
	if l2.l0 != DefaultL0/4 || !l2.dropDense || l2.dropShardFrac != 0.25 || l2.l1Keep != rank.DefaultL1Keep/2 {
		t.Fatalf("DegradeL2 is not the full cumulative ladder: %+v", l2)
	}
}

// TestDegradeForBudget checks the budget-to-rung policy: full quality with most of the
// budget left, climbing the ladder as the remaining budget falls, and the cheapest rung
// once the budget is gone, so a query never overruns the deadline.
func TestDegradeForBudget(t *testing.T) {
	cases := []struct {
		remaining time.Duration
		want      DegradeLevel
	}{
		{10 * time.Millisecond, DegradeNone},
		{8 * time.Millisecond, DegradeNone},
		{7 * time.Millisecond, DegradeL0},
		{6 * time.Millisecond, DegradeL0},
		{5 * time.Millisecond, DegradeDense},
		{4 * time.Millisecond, DegradeDense},
		{3 * time.Millisecond, DegradeShards},
		{2 * time.Millisecond, DegradeShards},
		{1 * time.Millisecond, DegradeL2},
		{0, DegradeL2},
		{-1 * time.Millisecond, DegradeL2},
	}
	for _, c := range cases {
		if got := DegradeForBudget(c.remaining); got != c.want {
			t.Fatalf("DegradeForBudget(%v) = %v, want %v", c.remaining, got, c.want)
		}
	}
}

// TestQueryL0OverridesRetrieval checks the first rung's lever actually flows to the
// shard: a query carrying a small L0 makes the shard retrieve that many candidates per
// plane instead of its built-in width, so the broker can shrink the candidate set under
// pressure rather than skipping retrieval.
func TestQueryL0OverridesRetrieval(t *testing.T) {
	const n = 80
	docs := makeCorpus(n)
	dir := t.TempDir()
	path := dir + "/shard.tsumugi"
	buildShardFile(t, path, docs, 0, n, 0, false)
	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Every document carries the shared term, so the lexical plane can return up to n
	// candidates. With the default L0 (1000) and a small K, retrieval returns all n.
	full, _, _, _ := s.retrieve(context.Background(), Query{Text: "common document", K: 5})
	if len(full) != n {
		t.Fatalf("default L0 retrieved %d candidates, want %d (every doc matches)", len(full), n)
	}
	// The same query with a small L0 override caps retrieval to that width.
	small, _, _, _ := s.retrieve(context.Background(), Query{Text: "common document", K: 5, L0: 10})
	if len(small) != 10 {
		t.Fatalf("L0=10 retrieved %d candidates, want 10", len(small))
	}
}

// TestSearchDegradedReportsLevel checks the no-degradation rung equals the plain
// completeness path and that the chosen level is reported back, so the reduction is
// observable in the result.
func TestSearchDegradedReportsLevel(t *testing.T) {
	b, parts := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	q := Query{Text: "common document", K: 20}
	plain := b.SearchComplete(context.Background(), q)
	none := b.SearchDegraded(context.Background(), q, DegradeNone)
	if none.Degraded != DegradeNone {
		t.Fatalf("DegradeNone reported level %v", none.Degraded)
	}
	if len(plain.Hits) != len(none.Hits) {
		t.Fatalf("SearchComplete returned %d hits, SearchDegraded(none) %d", len(plain.Hits), len(none.Hits))
	}
	for i := range plain.Hits {
		if plain.Hits[i] != none.Hits[i] {
			t.Fatalf("rank %d: SearchComplete %+v, SearchDegraded(none) %+v", i, plain.Hits[i], none.Hits[i])
		}
	}

	// A degraded query still answers over the shards it queries and reports its rung.
	deg := b.SearchDegraded(context.Background(), q, DegradeShards)
	if deg.Degraded != DegradeShards {
		t.Fatalf("DegradeShards reported level %v", deg.Degraded)
	}
	if deg.ShardsTotal >= parts {
		t.Fatalf("DegradeShards queried %d of %d shards, want fewer (it drops the lowest static rank)", deg.ShardsTotal, parts)
	}
	if !deg.Complete() {
		t.Fatalf("every queried shard responded yet the result was flagged incomplete")
	}
}

// TestDropDenseServesLexicalOnly checks the dense-drop rung: serving at DegradeDense
// produces the same top-k as a query that never carried a dense vector, the proof the
// lever clears the dense plane rather than something else.
func TestDropDenseServesLexicalOnly(t *testing.T) {
	const n, parts = 160, 4
	docs := makeCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := dir + "/shard" + string(rune('0'+p)) + ".tsumugi"
		lo := p * size
		buildShardFile(t, path, docs, lo, lo+size, uint32(lo), true) // with vectors
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	vec := docs[0].vec
	// Lexical-only ground truth: the same text query with no dense vector at all.
	lexOnly := b.SearchComplete(context.Background(), Query{Text: "common document", K: 20})
	// The dense-drop rung over a query that does carry a vector must match it, because the
	// rung clears the vector before the fan-out.
	dropped := b.SearchDegraded(context.Background(), Query{Text: "common document", Vector: vec, K: 20}, DegradeDense)
	if dropped.Degraded != DegradeDense {
		t.Fatalf("reported level %v, want drop-dense", dropped.Degraded)
	}
	if len(lexOnly.Hits) != len(dropped.Hits) {
		t.Fatalf("lexical-only returned %d hits, drop-dense %d", len(lexOnly.Hits), len(dropped.Hits))
	}
	for i := range lexOnly.Hits {
		if lexOnly.Hits[i] != dropped.Hits[i] {
			t.Fatalf("rank %d: lexical-only %+v, drop-dense %+v", i, lexOnly.Hits[i], dropped.Hits[i])
		}
	}
}

// TestDropLowStatic checks the shard-dropping lever orders shards by their per-shard
// static rank and drops the lowest, keeps the result deterministic in shard order, and
// never drops every shard.
func TestDropLowStatic(t *testing.T) {
	b, parts := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	targets := make([]int, parts)
	for i := range targets {
		targets[i] = i
	}
	st := b.loadState()
	static := st.shardStatics()
	if len(static) != parts {
		t.Fatalf("shardStatics returned %d entries, want %d", len(static), parts)
	}

	// A zero fraction and a single target are no-ops.
	if got := st.dropLowStatic(targets, 0); len(got) != parts {
		t.Fatalf("frac 0 dropped shards: %v", got)
	}
	if got := st.dropLowStatic([]int{2}, 0.5); len(got) != 1 || got[0] != 2 {
		t.Fatalf("single target was dropped: %v", got)
	}

	// A quarter of four shards drops exactly the one with the lowest static rank.
	lowest := 0
	for i := 1; i < parts; i++ {
		if static[i] < static[lowest] {
			lowest = i
		}
	}
	kept := st.dropLowStatic(targets, 0.25)
	if len(kept) != parts-1 {
		t.Fatalf("dropLowStatic(0.25) kept %d shards, want %d", len(kept), parts-1)
	}
	if !sort.IntsAreSorted(kept) {
		t.Fatalf("kept set is not in shard order: %v", kept)
	}
	for _, s := range kept {
		if s == lowest {
			t.Fatalf("kept the lowest-static-rank shard %d (static %v): %v", lowest, static[lowest], kept)
		}
	}

	// Dropping the whole fraction still keeps at least one shard, so a query never
	// degrades to no shards at all.
	all := st.dropLowStatic(targets, 1.0)
	if len(all) != 1 {
		t.Fatalf("frac 1.0 kept %d shards, want 1 (never drop them all)", len(all))
	}
}

// TestTrimL2DoesNotMutateCascade checks the L2-trim rung shrinks the L1 cut for the
// query through a per-query cascade copy, leaving the broker's shared cascade untouched
// so concurrent queries are unaffected, and still returns a ranked top-k.
func TestTrimL2DoesNotMutateCascade(t *testing.T) {
	b, _ := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	before := b.cascade.L1Keep
	res := b.SearchDegraded(context.Background(), Query{Text: "common document", K: 20}, DegradeL2)
	if b.cascade.L1Keep != before {
		t.Fatalf("DegradeL2 mutated the shared cascade L1Keep: %d -> %d", before, b.cascade.L1Keep)
	}
	if res.Degraded != DegradeL2 {
		t.Fatalf("reported level %v, want trim-l2", res.Degraded)
	}
	if len(res.Hits) == 0 {
		t.Fatalf("trim-l2 returned no hits")
	}
	// The hits are ranked by score descending.
	for i := 1; i < len(res.Hits); i++ {
		if res.Hits[i-1].Score < res.Hits[i].Score {
			t.Fatalf("hits not score-ordered at rank %d: %v < %v", i, res.Hits[i-1].Score, res.Hits[i].Score)
		}
	}
}

// TestSearchWithinBudget checks the deadline-driven policy: a context with a tight
// deadline serves a degraded result, a generous one serves full quality, and no
// deadline serves full quality.
func TestSearchWithinBudget(t *testing.T) {
	b, _ := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	q := Query{Text: "common document", K: 20}

	// No deadline: full quality.
	none := b.SearchWithinBudget(context.Background(), q)
	if none.Degraded != DegradeNone {
		t.Fatalf("no-deadline query degraded to %v", none.Degraded)
	}

	// A generous deadline still runs full quality.
	genCtx, genCancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer genCancel()
	gen := b.SearchWithinBudget(genCtx, q)
	if gen.Degraded != DegradeNone {
		t.Fatalf("generous-deadline query degraded to %v", gen.Degraded)
	}

	// A tight deadline degrades. The remaining budget at entry is under the full-quality
	// threshold, so the policy climbs the ladder above none.
	tightCtx, tightCancel := context.WithTimeout(context.Background(), 3*time.Millisecond)
	defer tightCancel()
	tight := b.SearchWithinBudget(tightCtx, q)
	if tight.Degraded == DegradeNone {
		t.Fatalf("tight-deadline query ran at full quality, want a degraded rung")
	}
}
