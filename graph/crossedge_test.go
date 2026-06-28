package graph

import (
	"bytes"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// TestCrossEdgeEmpty checks that a graph with no cross-shard edges frames no
// cross-shard list and reads back as no far neighbors, the case every region built
// before this slice is in.
func TestCrossEdgeEmpty(t *testing.T) {
	b := NewBuilder(10)
	for x := 0; x < 10; x++ {
		b.AddEdge(x, (x+1)%10)
	}
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if g.xs != nil {
		t.Fatal("region with no cross edges carries a cross-shard list")
	}
	if got := g.CrossNeighbors(3); got != nil {
		t.Fatalf("CrossNeighbors on an empty list = %v, want nil", got)
	}
	calls := 0
	g.ForEachCrossEdge(func(int, []uint64) { calls++ })
	if calls != 0 {
		t.Fatalf("ForEachCrossEdge called fn %d times on an empty list", calls)
	}
}

// TestCrossEdgeRoundTrip builds cross edges with duplicates and out-of-order
// targets and checks they read back sorted, deduped, and grouped by source, both by
// random access (CrossNeighbors) and by the sequential walk (ForEachCrossEdge).
func TestCrossEdgeRoundTrip(t *testing.T) {
	const n = 200
	want := map[int][]uint64{
		0:   {1 << 40, 5, 5, 1<<40 + 1, 9}, // unsorted with a duplicate
		7:   {1<<50 + 3, 1<<50 + 2, 1<<50 + 1},
		199: {42},
	}
	b := NewBuilder(n)
	for src, tgts := range want {
		for _, g := range tgts {
			b.AddCrossEdge(src, g)
		}
	}
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if g.xs == nil {
		t.Fatal("cross edges did not produce a cross-shard list")
	}

	expect := func(src int) []uint64 {
		out := append([]uint64(nil), want[src]...)
		return sortDedupU64(out)
	}
	for src := range want {
		got := g.CrossNeighbors(src)
		if !reflect.DeepEqual(got, expect(src)) {
			t.Fatalf("CrossNeighbors(%d) = %v, want %v", src, got, expect(src))
		}
	}
	// A node without cross edges has none.
	if got := g.CrossNeighbors(1); got != nil {
		t.Fatalf("CrossNeighbors(1) = %v, want nil", got)
	}

	// The sequential walk yields the same records in ascending source order.
	var seenSrc []int
	g.ForEachCrossEdge(func(src int, tgts []uint64) {
		seenSrc = append(seenSrc, src)
		if !reflect.DeepEqual(tgts, expect(src)) {
			t.Fatalf("ForEachCrossEdge(%d) = %v, want %v", src, tgts, expect(src))
		}
	})
	wantSrc := []int{0, 7, 199}
	if !reflect.DeepEqual(seenSrc, wantSrc) {
		t.Fatalf("walk visited sources %v, want %v", seenSrc, wantSrc)
	}
}

// TestCrossEdgeOOCMatchesResident checks the out-of-core builder frames the
// cross-shard list byte-for-byte the same as the resident builder, the equivalence
// slice 38 holds for the adjacency and this slice extends to part four.
func TestCrossEdgeOOCMatchesResident(t *testing.T) {
	const n = 5000
	rng := rand.New(rand.NewSource(13))
	type ce struct {
		from int
		to   uint64
	}
	var edges []ce
	for i := 0; i < 20000; i++ {
		edges = append(edges, ce{from: rng.Intn(n), to: rng.Uint64() >> 4})
	}

	rb := NewBuilder(n)
	ob := NewOOCBuilder(n).WithSpillThreshold(1024) // force the disk path
	for i := 0; i < n; i++ {
		rb.AddEdge(i, (i+1)%n)
		ob.AddEdge(i, (i+1)%n)
	}
	for _, e := range edges {
		rb.AddCrossEdge(e.from, e.to)
		ob.AddCrossEdge(e.from, e.to)
	}
	resident := rb.Build()
	ooc, err := ob.Build()
	if err != nil {
		t.Fatalf("ooc build: %v", err)
	}
	if !bytes.Equal(resident, ooc) {
		t.Fatalf("resident and ooc regions differ: %d vs %d bytes", len(resident), len(ooc))
	}
}

// TestCrossEdgeRoute exercises the resolution join: two shards whose cross edges
// point at each other's nodes, routed against the shards' id tables into per-shard
// inbound edges.
func TestCrossEdgeRoute(t *testing.T) {
	// Shard 0 holds global ids 100..103, shard 1 holds 200..203, both as explicit
	// permuted tables so Dense goes through the id table, not the fast path.
	ids0 := []uint64{102, 100, 103, 101}
	ids1 := []uint64{201, 203, 200, 202}

	b0 := NewBuilder(4).WithNodeIDs(ids0)
	b0.AddCrossEdge(0, 200) // dense 0 (global 102) -> shard1 dense 2
	b0.AddCrossEdge(0, 203) // dense 0 (global 102) -> shard1 dense 1
	b0.AddCrossEdge(3, 201) // dense 3 (global 101) -> shard1 dense 0
	b0.AddCrossEdge(1, 999) // dense 1 -> a node no shard here holds, dropped
	g0, err := Open(b0.Build())
	if err != nil {
		t.Fatalf("open shard0: %v", err)
	}

	b1 := NewBuilder(4).WithNodeIDs(ids1)
	b1.AddCrossEdge(2, 100) // dense 2 (global 200) -> shard0 dense 1
	g1, err := Open(b1.Build())
	if err != nil {
		t.Fatalf("open shard1: %v", err)
	}

	inbound := RouteCrossEdges([]*Region{g0, g1})

	// Into shard 0: shard1 dense 2 (global 200) -> shard0 dense 1 (global 100).
	want0 := []InboundEdge{{Source: 200, Target: 1}}
	if !reflect.DeepEqual(inbound[0], want0) {
		t.Fatalf("inbound[0] = %v, want %v", inbound[0], want0)
	}

	// Into shard 1: three resolved edges from shard 0, the global-999 one dropped.
	got1 := append([]InboundEdge(nil), inbound[1]...)
	sort.Slice(got1, func(i, j int) bool {
		if got1[i].Source != got1[j].Source {
			return got1[i].Source < got1[j].Source
		}
		return got1[i].Target < got1[j].Target
	})
	want1 := []InboundEdge{
		{Source: 101, Target: 0}, // dense3 global101 -> shard1 dense0 (global200)
		{Source: 102, Target: 1}, // dense0 global102 -> shard1 dense1 (global203)
		{Source: 102, Target: 2}, // dense0 global102 -> shard1 dense2 (global200)
	}
	if !reflect.DeepEqual(got1, want1) {
		t.Fatalf("inbound[1] = %v, want %v", got1, want1)
	}
}

// TestCrossEdgeCorrupt checks the decoder rejects truncations rather than reading
// out of bounds.
func TestCrossEdgeCorrupt(t *testing.T) {
	recs := []crossRecord{
		{source: 1, targets: []uint64{10, 20, 30}},
		{source: 5, targets: []uint64{1 << 40}},
	}
	blob := encodeCrossShard(recs, DefaultParams())
	if blob == nil {
		t.Fatal("expected a cross-shard blob")
	}
	for _, cut := range []int{0, 2, 6, len(blob) - 1} {
		if _, err := decodeCrossList(blob[:cut], DefaultParams()); err == nil {
			t.Fatalf("decodeCrossList accepted a %d-byte truncation", cut)
		}
	}
	if _, err := decodeCrossList(blob, DefaultParams()); err != nil {
		t.Fatalf("decodeCrossList rejected the whole blob: %v", err)
	}
}

// TestCrossEdgeSize records the compressed size of a realistic cross-shard list:
// power-law out-degrees of far edges with globally spread 40-bit target ids.
func TestCrossEdgeSize(t *testing.T) {
	const n = 100000
	rng := rand.New(rand.NewSource(5))
	b := NewBuilder(n)
	var edges int
	for x := 0; x < n; x++ {
		// Most nodes have a few far edges, a tail has many.
		deg := 1 + rng.Intn(8)
		if rng.Intn(50) == 0 {
			deg += rng.Intn(200)
		}
		for k := 0; k < deg; k++ {
			b.AddCrossEdge(x, rng.Uint64()&((1<<40)-1))
			edges++
		}
	}
	region := b.Build()
	g, err := Open(region)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	// Re-measure just the cross blob by reconstructing its length from the header.
	h, _ := decodeHeader(region)
	bitsPerEdge := float64(h.xsLen) * 8 / float64(edges)
	t.Logf("cross-shard list: %d nodes, %d edges, %d bytes, %.2f bits an edge",
		n, edges, h.xsLen, bitsPerEdge)
	if g.xs == nil {
		t.Fatal("expected a cross-shard list")
	}
}

func benchCrossList(n, edges int) *Region {
	rng := rand.New(rand.NewSource(1))
	b := NewBuilder(n)
	for i := 0; i < edges; i++ {
		b.AddCrossEdge(rng.Intn(n), rng.Uint64()&((1<<40)-1))
	}
	g, err := Open(b.Build())
	if err != nil {
		panic(err)
	}
	return g
}

func BenchmarkCrossNeighbors(b *testing.B) {
	const n = 1 << 20
	g := benchCrossList(n, n*4)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		sink += len(g.CrossNeighbors(i & (n - 1)))
	}
	_ = sink
}
