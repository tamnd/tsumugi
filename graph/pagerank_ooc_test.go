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
