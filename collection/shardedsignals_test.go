package collection

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/graph"
)

// rankTol is the float tolerance the sharded signals are held to against the merged
// oracle: the cross-shard rank loops match the in-core ranks to float32 precision, far
// finer than the byte-quantized feature column keeps, so 1e-5 is a tight gate that still
// absorbs the float32 rounding inside the iteration.
const rankTol = 1e-5

// maxRankErr is the largest absolute difference between two float64 vectors, the gauge
// the rank gates report.
func maxRankErr(got, want []float64) float64 {
	var m float64
	for i := range want {
		if d := math.Abs(got[i] - want[i]); d > m {
			m = d
		}
	}
	return m
}

// signalsClose asserts the assembled signal set equals the merged oracle field for
// field: the float columns within rankTol, the integer counts and the categorical
// language id exactly. A field-level failure names the field and the worst index so a
// regression points straight at the form that drifted.
func signalsClose(t testing.TB, got, want graphSignals) {
	t.Helper()
	floatField := func(name string, g, w []float64) {
		if len(g) != len(w) {
			t.Fatalf("%s: length %d, want %d", name, len(g), len(w))
		}
		if e := maxRankErr(g, w); e > rankTol {
			t.Fatalf("%s: max error %g exceeds %g", name, e, rankTol)
		}
	}
	intField := func(name string, g, w []int) {
		if len(g) != len(w) {
			t.Fatalf("%s: length %d, want %d", name, len(g), len(w))
		}
		for i := range w {
			if g[i] != w[i] {
				t.Fatalf("%s[%d] = %d, want %d", name, i, g[i], w[i])
			}
		}
	}
	floatField("pageRank", got.pageRank, want.pageRank)
	floatField("hostRank", got.hostRank, want.hostRank)
	floatField("domainRank", got.domainRank, want.domainRank)
	floatField("trust", got.trust, want.trust)
	floatField("spamMass", got.spamMass, want.spamMass)
	floatField("reciprocity", got.reciprocity, want.reciprocity)
	floatField("hostLinkDiv", got.hostLinkDiv, want.hostLinkDiv)
	floatField("nearDup", got.nearDup, want.nearDup)
	floatField("outboundSpam", got.outboundSpam, want.outboundSpam)
	floatField("langConsist", got.langConsist, want.langConsist)
	floatField("staticRank", got.staticRank, want.staticRank)
	intField("inDegree", got.inDegree, want.inDegree)
	intField("linkingDomains", got.linkingDomains, want.linkingDomains)
	intField("linkingHosts", got.linkingHosts, want.linkingHosts)
	if len(got.langID) != len(want.langID) {
		t.Fatalf("langID: length %d, want %d", len(got.langID), len(want.langID))
	}
	for i := range want.langID {
		if got.langID[i] != want.langID[i] {
			t.Fatalf("langID[%d] = %d, want %d", i, got.langID[i], want.langID[i])
		}
	}
}

// TestShardedSignalsMatchMergedFromBuild proves the aggregator equals globalSignals
// over shards a real build wrote. It builds a multi-shard collection over a corpus
// whose links cross hosts, loads the per-shard graph regions, assembles the signals off
// them, and compares the whole set field for field against the merged in-core pass over
// the same documents. It asserts the build produced cross-shard edges and a real spread
// of rank and in-degree first, so a degenerate build cannot pass it.
func TestShardedSignalsMatchMergedFromBuild(t *testing.T) {
	tmp := t.TempDir()
	src := writeCrossLinkedJSONL(t, tmp, "web.jsonl")
	out := filepath.Join(tmp, "col")
	const shardSize = 40
	res, err := Build(Options{Source: src, Out: out, ShardSize: shardSize})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("want a multi-shard collection, got %d shards", res.Shards)
	}

	docs, gids, _ := buildLayout(t, src)
	dir := buildDir(docs)
	regions, closeAll := loadShardGraphs(t, out)
	defer closeAll()

	var crossEdges int
	for _, e := range graph.RouteCrossEdges(regions) {
		crossEdges += len(e)
	}
	if crossEdges == 0 {
		t.Fatal("the build produced no cross-shard edges, so the gate is vacuous")
	}

	got := shardedSignals(regions, docs, gids, nil, nil, dir, DefaultPartitionParams())
	want, _, _ := globalSignals(docs, nil, nil)

	var nonzeroRank, nonzeroDeg int
	for i := range want.pageRank {
		if want.pageRank[i] > 0 {
			nonzeroRank++
		}
		if want.inDegree[i] > 0 {
			nonzeroDeg++
		}
	}
	if nonzeroRank == 0 || nonzeroDeg == 0 {
		t.Fatalf("merged signals are degenerate: nonzeroRank=%d nonzeroDeg=%d", nonzeroRank, nonzeroDeg)
	}

	signalsClose(t, got, want)
	t.Logf("docs=%d shards=%d crossEdges=%d nonzeroRank=%d prErr=%.2e",
		len(docs), len(regions), crossEdges, nonzeroRank, maxRankErr(got.pageRank, want.pageRank))
}

// BenchmarkShardedSignals measures one full signal assembly off the loaded regions, the
// per-build cost the reorder pays once in place of the merged in-core rank. The build and
// the region load are setup, outside the timed loop.
func BenchmarkShardedSignals(b *testing.B) {
	tmp := b.TempDir()
	src := writeCrossLinkedJSONL(b, tmp, "web.jsonl")
	out := filepath.Join(tmp, "col")
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 40}); err != nil {
		b.Fatalf("build: %v", err)
	}
	docs, gids, _ := buildLayout(b, src)
	dir := buildDir(docs)
	regions, closeAll := loadShardGraphs(b, out)
	defer closeAll()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = shardedSignals(regions, docs, gids, nil, nil, dir, DefaultPartitionParams())
	}
}
