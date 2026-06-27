package graph

import (
	"math"
	"math/rand"
	"testing"
)

// clusteredEdges builds the edges of a host-local web graph: pages are grouped
// into hosts of hostSize contiguous ids, every page links to its host's shared
// nav block (a contiguous run at the host start) so the lists across a host are
// nearly identical, plus a few intra-host article links and the odd cross-host
// link. host(u) is u/hostSize. This is the shape host grouping is meant to
// recover, returned as raw edges so a test can scramble and relabel the ids.
func clusteredEdges(n, hostSize int, seed int64) (edges [][2]int, hostOf []int) {
	rng := rand.New(rand.NewSource(seed))
	const navSize = 8
	hostOf = make([]int, n)
	for u := 0; u < n; u++ {
		host := u / hostSize
		hostOf[u] = host
		lo := host * hostSize
		hi := lo + hostSize
		if hi > n {
			hi = n
		}
		for k := 0; k < navSize && lo+k < hi; k++ {
			if lo+k != u {
				edges = append(edges, [2]int{u, lo + k})
			}
		}
		extra := 3 + rng.Intn(10)
		for k := 0; k < extra; k++ {
			var t int
			if rng.Intn(10) < 9 && hi > lo {
				t = lo + rng.Intn(hi-lo)
			} else {
				t = rng.Intn(n)
			}
			if t != u {
				edges = append(edges, [2]int{u, t})
			}
		}
	}
	return edges, hostOf
}

// relabel rewrites every edge's endpoints through a map old -> new.
func relabel(edges [][2]int, newID []int) [][2]int {
	out := make([][2]int, len(edges))
	for i, e := range edges {
		out[i] = [2]int{newID[e[0]], newID[e[1]]}
	}
	return out
}

// outListsFromEdges groups edges into per-source adjacency slices.
func outListsFromEdges(n int, edges [][2]int) [][]int32 {
	out := make([][]int32, n)
	for _, e := range edges {
		out[e[0]] = append(out[e[0]], int32(e[1]))
	}
	return out
}

// regionBits builds a region from edges and returns its total size in bits and
// the directed edge count, the bits/edge measure the codec is judged on.
func regionBits(n int, edges [][2]int) (bits float64, edges64 uint64) {
	region := func() []byte {
		b := NewBuilder(n)
		for _, e := range edges {
			b.AddEdge(e[0], e[1])
		}
		return b.Build()
	}()
	g, err := Open(region)
	if err != nil {
		panic(err)
	}
	return float64(len(region) * 8), g.EdgeCount()
}

// TestReorderReducesBitsPerEdge is the mechanism gate: a scrambled web graph,
// where the good order is hidden so the codec spends many bits an edge, is
// reordered by host grouping plus Recursive Graph Bisection, and the codec then
// spends far fewer. The real graph can never exercise this (a breadth-first
// crawl resolves almost no edges, so it is too flat to compress), so the gate is
// synthetic by the spec's own rule that synthetic data proves a mechanism the
// real data cannot reach. The compression win is what the whole graph plane rests
// on, so it is gated, not merely reported.
func TestReorderReducesBitsPerEdge(t *testing.T) {
	const n = 4000
	const hostSize = 50
	base, hostOf := clusteredEdges(n, hostSize, 7)

	// Scramble the ids so the host-local structure is hidden: a random
	// permutation scatters each host's pages across the id space.
	rng := rand.New(rand.NewSource(99))
	scramble := rng.Perm(n) // old id -> scrambled id
	scrambled := relabel(base, scramble)

	// The grouping hint: each scrambled node's original host, the key the build
	// gets from the document's host before it ever sees the graph.
	group := make([]int, n)
	for old, s := range scramble {
		group[s] = hostOf[old]
	}

	scrBits, edges := regionBits(n, scrambled)
	scrPerEdge := scrBits / float64(edges)

	out := outListsFromEdges(n, scrambled)
	order := Reorder(out, group, DefaultBPConfig())
	inv := Invert(order)
	reordered := relabel(scrambled, inv)
	reBits, edges2 := regionBits(n, reordered)
	if edges2 != edges {
		t.Fatalf("relabel changed edge count: %d vs %d", edges2, edges)
	}
	rePerEdge := reBits / float64(edges)

	t.Logf("scrambled %.2f bits/edge, reordered %.2f bits/edge (%.0f%% smaller)",
		scrPerEdge, rePerEdge, 100*(1-rePerEdge/scrPerEdge))
	if rePerEdge >= scrPerEdge {
		t.Fatalf("reorder did not reduce bits/edge: %.2f -> %.2f", scrPerEdge, rePerEdge)
	}
	// Host grouping alone gets a web graph into the single digits; with the BP
	// refinement on top the result should be a large fraction smaller than the
	// scrambled order, not a marginal trim.
	if rePerEdge > 0.6*scrPerEdge {
		t.Fatalf("reorder win too small: %.2f -> %.2f bits/edge", scrPerEdge, rePerEdge)
	}
}

// TestReorderPreservesPageRank is the invariance gate: PageRank is a property of
// the graph, not of its labeling, so relabeling every node by the reorder must
// carry each node's rank to its new id unchanged. This is what lets the build use
// the reorder as the dense docID assignment without disturbing the signals baked
// against those ids.
func TestReorderPreservesPageRank(t *testing.T) {
	const n = 2000
	const hostSize = 40
	base, hostOf := clusteredEdges(n, hostSize, 3)
	rng := rand.New(rand.NewSource(5))
	scramble := rng.Perm(n)
	scrambled := relabel(base, scramble)
	group := make([]int, n)
	for old, s := range scramble {
		group[s] = hostOf[old]
	}

	g1 := buildGraph(n, scrambled)
	pr1 := PageRank(g1, DefaultPRConfig())

	out := outListsFromEdges(n, scrambled)
	order := Reorder(out, group, DefaultBPConfig())
	inv := Invert(order)
	g2 := buildGraph(n, relabel(scrambled, inv))
	pr2 := PageRank(g2, DefaultPRConfig())

	var maxDiff float64
	for v := 0; v < n; v++ {
		if d := math.Abs(pr1[v] - pr2[inv[v]]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-9 {
		t.Fatalf("pagerank not invariant under reorder: max diff %g", maxDiff)
	}
}

// TestReorderIsPermutation checks the order is a bijection over [0, n): every old
// id appears exactly once, so no document is dropped or duplicated when the build
// relabels by it.
func TestReorderIsPermutation(t *testing.T) {
	const n = 500
	edges, hostOf := clusteredEdges(n, 25, 1)
	out := outListsFromEdges(n, edges)
	order := Reorder(out, hostOf, DefaultBPConfig())
	if len(order) != n {
		t.Fatalf("order length %d, want %d", len(order), n)
	}
	seen := make([]bool, n)
	for _, old := range order {
		if old < 0 || old >= n || seen[old] {
			t.Fatalf("order is not a permutation: bad or repeated id %d", old)
		}
		seen[old] = true
	}
	inv := Invert(order)
	for i := 0; i < n; i++ {
		if order[inv[i]] != i {
			t.Fatalf("Invert is not the inverse at %d", i)
		}
	}
}

// TestReorderBPOnlyReducesBits checks the bisection pass earns its place on its
// own, with no grouping hint: even from the identity order it clusters nodes that
// share out-neighbors and lowers the bits an edge below the unordered scrambled
// baseline.
func TestReorderBPOnlyReducesBits(t *testing.T) {
	const n = 2000
	base, _ := clusteredEdges(n, 40, 11)
	rng := rand.New(rand.NewSource(21))
	scramble := rng.Perm(n)
	scrambled := relabel(base, scramble)

	scrBits, edges := regionBits(n, scrambled)
	out := outListsFromEdges(n, scrambled)
	order := Reorder(out, nil, DefaultBPConfig()) // no grouping, pure BP
	inv := Invert(order)
	reBits, _ := regionBits(n, relabel(scrambled, inv))

	scrPerEdge := scrBits / float64(edges)
	rePerEdge := reBits / float64(edges)
	t.Logf("BP only: scrambled %.2f -> reordered %.2f bits/edge", scrPerEdge, rePerEdge)
	if rePerEdge >= scrPerEdge {
		t.Fatalf("pure BP did not reduce bits/edge: %.2f -> %.2f", scrPerEdge, rePerEdge)
	}
}
