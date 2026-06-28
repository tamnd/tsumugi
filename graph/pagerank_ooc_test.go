package graph

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// topRanked returns the indices of the k highest ranks, ties broken by index, the
// ordering the static-rank column and the posting order care about.
func topRanked(r []float64, k int) []int {
	idx := make([]int, len(r))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if r[idx[a]] != r[idx[b]] {
			return r[idx[a]] > r[idx[b]]
		}
		return idx[a] < idx[b]
	})
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

func f32to64(s []float32) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = float64(v)
	}
	return out
}

// TestStreamPageRankMatchesInCore is the gate: the out-of-core streaming pass, which
// never materializes the adjacency, must compute the same ranks as the in-core CSR
// pass. On a random graph the two agree to float32 precision and the streamed ranks
// still sum to one, so the streaming structure is exact, not an approximation of the
// algorithm.
func TestStreamPageRankMatchesInCore(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const n = 500
	var edges [][2]int
	for i := 0; i < 1500; i++ {
		edges = append(edges, [2]int{rng.Intn(n), rng.Intn(n)})
	}
	g := buildGraph(n, edges)

	want := PageRank(g, DefaultPRConfig())
	got := StreamPageRank(g, OutDegrees(g), DefaultPRConfig())
	if len(got) != n {
		t.Fatalf("stream pagerank length %d, want %d", len(got), n)
	}

	var maxDiff, sum float64
	for i := 0; i < n; i++ {
		d := math.Abs(want[i] - float64(got[i]))
		if d > maxDiff {
			maxDiff = d
		}
		sum += float64(got[i])
	}
	if maxDiff > 1e-5 {
		t.Fatalf("stream vs in-core max abs diff %g, want < 1e-5", maxDiff)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("streamed ranks sum to %g, want ~1", sum)
	}
}

// maxAbsDiff returns the largest absolute gap between an in-core float64 rank vector
// and a streamed float32 one, the float32-precision agreement the streaming ranks
// must hold against their in-core references.
func maxAbsDiff(want []float64, got []float32) float64 {
	var m float64
	for i := range want {
		d := math.Abs(want[i] - float64(got[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// TestStreamTrustRankMatchesInCore gates the seed-biased forward streaming pass: with
// the surfer teleporting onto a trust seed set, the personalized out-of-core iteration
// must reproduce the in-core TrustRank, which restarts on the same seed vector. The
// two agree to float32 precision, so the streaming structure carries the
// personalization exactly, not just the uniform case.
func TestStreamTrustRankMatchesInCore(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const n = 400
	var edges [][2]int
	for i := 0; i < 1800; i++ {
		edges = append(edges, [2]int{rng.Intn(n), rng.Intn(n)})
	}
	g := buildGraph(n, edges)
	seeds := []int{3, 17, 42, 88, 200, 311}

	want := TrustRank(g, seeds, DefaultPRConfig())
	got := StreamPageRankP(g, OutDegrees(g), SeedTeleport(n, seeds), DefaultPRConfig())
	if d := maxAbsDiff(want, got); d > 1e-5 {
		t.Fatalf("stream TrustRank vs in-core max abs diff %g, want < 1e-5", d)
	}
	var sum float64
	for _, v := range got {
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("streamed TrustRank sums to %g, want ~1", sum)
	}
}

// TestStreamInversePageRankMatchesInCore gates the reversed-graph streaming pass: a
// uniform iteration over ReverseSource, dividing by the original in-degree, must
// reproduce the in-core InversePageRank, which runs the same iteration over the
// materialized reverse CSR. The reversed view never copies the adjacency, so the gate
// proves the on-demand out-list decode is a faithful transpose.
func TestStreamInversePageRankMatchesInCore(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	const n = 450
	var edges [][2]int
	for i := 0; i < 2000; i++ {
		edges = append(edges, [2]int{rng.Intn(n), rng.Intn(n)})
	}
	g := buildGraph(n, edges)

	want := InversePageRank(g, DefaultPRConfig())
	rev := ReverseSource(g)
	outdeg := OutDegreesFromSource(n, g.InDegree)
	got := StreamPageRankP(rev, outdeg, UniformTeleport(n), DefaultPRConfig())
	if d := maxAbsDiff(want, got); d > 1e-5 {
		t.Fatalf("stream InversePageRank vs in-core max abs diff %g, want < 1e-5", d)
	}
}

// TestStreamAntiTrustRankMatchesInCore gates the seed-biased reversed pass: distrust
// teleporting onto a spam seed set and flowing backward along links, the streamed
// reverse iteration against the in-core AntiTrustRank. It combines both extensions,
// the reversed source and the seed teleport, and must match to float32 precision.
func TestStreamAntiTrustRankMatchesInCore(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	const n = 380
	var edges [][2]int
	for i := 0; i < 1600; i++ {
		edges = append(edges, [2]int{rng.Intn(n), rng.Intn(n)})
	}
	g := buildGraph(n, edges)
	spam := []int{5, 50, 150, 300}

	want := AntiTrustRank(g, spam, DefaultPRConfig())
	rev := ReverseSource(g)
	outdeg := OutDegreesFromSource(n, g.InDegree)
	got := StreamPageRankP(rev, outdeg, SeedTeleport(n, spam), DefaultPRConfig())
	if d := maxAbsDiff(want, got); d > 1e-5 {
		t.Fatalf("stream AntiTrustRank vs in-core max abs diff %g, want < 1e-5", d)
	}
}

// TestSeedTeleportEmptyIsUniform proves an empty seed set degenerates to the uniform
// teleport, so a seed-biased streamed rank with no seeds is plain PageRank, the same
// fallback the in-core seedVec makes.
func TestSeedTeleportEmptyIsUniform(t *testing.T) {
	const n = 32
	u := UniformTeleport(n)
	s := SeedTeleport(n, nil)
	if len(s) != n {
		t.Fatalf("seed teleport length %d, want %d", len(s), n)
	}
	for i := range u {
		if s[i] != u[i] {
			t.Fatalf("empty seed teleport differs from uniform at %d: %g vs %g", i, s[i], u[i])
		}
	}
}

// TestStreamPageRankOrdering checks the streaming pass orders documents identically
// to the in-core pass where the ranks are well separated: on a hub graph the two
// agree on the full top-of-the-list order, which is what the posting order depends
// on.
func TestStreamPageRankOrdering(t *testing.T) {
	// A few hubs of different in-degree give a clearly separated rank order.
	n := 60
	var edges [][2]int
	for i := 10; i < n; i++ {
		edges = append(edges, [2]int{i, 0}) // hub 0: 50 in-links
	}
	for i := 10; i < 40; i++ {
		edges = append(edges, [2]int{i, 1}) // hub 1: 30 in-links
	}
	for i := 10; i < 25; i++ {
		edges = append(edges, [2]int{i, 2}) // hub 2: 15 in-links
	}
	g := buildGraph(n, edges)

	want := topRanked(PageRank(g, DefaultPRConfig()), 5)
	got := topRanked(f32to64(StreamPageRank(g, OutDegrees(g), DefaultPRConfig())), 5)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("top-5 order differs at %d: in-core %v, stream %v", i, want, got)
		}
	}
	if want[0] != 0 {
		t.Fatalf("expected hub 0 ranked first, got %d", want[0])
	}
}

// benchGraph is the shared fixture for the rank benchmarks: a moderate random graph
// large enough that the in-core CSR materialization is a measurable share of the
// in-core path's cost.
func benchGraph(n, m int) *Region {
	rng := rand.New(rand.NewSource(99))
	edges := make([][2]int, m)
	for i := range edges {
		edges[i] = [2]int{rng.Intn(n), rng.Intn(n)}
	}
	return buildGraph(n, edges)
}

// BenchmarkInCorePageRank measures the in-core path, which decodes the whole transpose
// into a flat CSR once and then iterates over flat arrays.
func BenchmarkInCorePageRank(b *testing.B) {
	g := benchGraph(20000, 200000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = PageRank(g, DefaultPRConfig())
	}
}

// BenchmarkStreamPageRank measures the out-of-core path on the same graph: it
// re-streams the transpose each iteration from the region and never materializes the
// CSR, so it allocates the rank vectors and the out-degree array, not the adjacency.
func BenchmarkStreamPageRank(b *testing.B) {
	g := benchGraph(20000, 200000)
	outdeg := OutDegrees(g)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StreamPageRank(g, outdeg, DefaultPRConfig())
	}
}

// BenchmarkStreamTrustRank measures the seed-biased forward streaming path.
func BenchmarkStreamTrustRank(b *testing.B) {
	g := benchGraph(20000, 200000)
	outdeg := OutDegrees(g)
	p := SeedTeleport(g.NodeCount(), []int{1, 100, 5000, 19999})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StreamPageRankP(g, outdeg, p, DefaultPRConfig())
	}
}

// BenchmarkStreamInversePageRank measures the reversed-source streaming path, where
// each in-list is decoded as an out-list from the region on demand.
func BenchmarkStreamInversePageRank(b *testing.B) {
	g := benchGraph(20000, 200000)
	rev := ReverseSource(g)
	outdeg := OutDegreesFromSource(g.NodeCount(), g.InDegree)
	p := UniformTeleport(g.NodeCount())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StreamPageRankP(rev, outdeg, p, DefaultPRConfig())
	}
}
