package graph

import (
	"math/rand"
	"sort"
	"testing"
)

// TestIDTableContiguousFastPath checks that a contiguous run of global ids that the
// dense order follows takes the no-table fast path: nodeBase is set, no id table
// blob is framed, and the mapping both ways is the add and subtract.
func TestIDTableContiguousFastPath(t *testing.T) {
	const n = 1000
	const base = 1 << 40
	ids := make([]uint64, n)
	for d := range ids {
		ids[d] = base + uint64(d)
	}
	nodeBase, blob := computeIDTable(n, ids)
	if nodeBase != base {
		t.Fatalf("nodeBase = %d, want %d", nodeBase, base)
	}
	if blob != nil {
		t.Fatalf("contiguous run framed a %d-byte id table, want none", len(blob))
	}

	g := buildRegionWithIDs(t, n, ids)
	if g.idt != nil {
		t.Fatal("region carries an id table on the contiguous fast path")
	}
	for d := 0; d < n; d++ {
		if got := g.Global(d); got != ids[d] {
			t.Fatalf("Global(%d) = %d, want %d", d, got, ids[d])
		}
		dense, ok := g.Dense(ids[d])
		if !ok || dense != d {
			t.Fatalf("Dense(%d) = (%d,%v), want (%d,true)", ids[d], dense, ok, d)
		}
	}
	if _, ok := g.Dense(base - 1); ok {
		t.Fatal("Dense accepted an id below the run")
	}
	if _, ok := g.Dense(base + n); ok {
		t.Fatal("Dense accepted an id above the run")
	}
}

// TestIDTableIdentityDefault checks the degenerate default: no node ids supplied,
// so the region maps dense docID to itself with no table, which is what a
// self-contained collection graph relies on.
func TestIDTableIdentityDefault(t *testing.T) {
	const n = 50
	g, err := Open(NewBuilder(n).Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if g.idt != nil || g.nodeBase != 0 {
		t.Fatalf("default region has idt=%v nodeBase=%d, want identity", g.idt != nil, g.nodeBase)
	}
	for d := 0; d < n; d++ {
		if g.Global(d) != uint64(d) {
			t.Fatalf("Global(%d) = %d, want identity", d, g.Global(d))
		}
		dense, ok := g.Dense(uint64(d))
		if !ok || dense != d {
			t.Fatalf("Dense(%d) = (%d,%v), want (%d,true)", d, dense, ok, d)
		}
	}
}

// TestIDTablePermutedRoundTrip exercises the explicit table: the global ids are a
// permutation of a base range so the dense order does not follow them, forcing the
// sorted Elias-Fano array plus the rank permutation, and every dense docID round
// trips to its id and back.
func TestIDTablePermutedRoundTrip(t *testing.T) {
	const n = 2000
	rng := rand.New(rand.NewSource(7))
	ids := make([]uint64, n)
	for d := range ids {
		ids[d] = 100 + uint64(d)
	}
	rng.Shuffle(n, func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	if _, ok := contiguousBase(ids); ok {
		t.Fatal("shuffled ids unexpectedly looked contiguous")
	}

	g := buildRegionWithIDs(t, n, ids)
	if g.idt == nil {
		t.Fatal("permuted ids did not produce an id table")
	}
	for d := 0; d < n; d++ {
		if got := g.Global(d); got != ids[d] {
			t.Fatalf("Global(%d) = %d, want %d", d, got, ids[d])
		}
		dense, ok := g.Dense(ids[d])
		if !ok || dense != d {
			t.Fatalf("Dense(%d) = (%d,%v), want (%d,true)", ids[d], dense, ok, d)
		}
	}
	// An id the shard does not hold is rejected.
	if _, ok := g.Dense(99); ok {
		t.Fatal("Dense accepted an id below the set")
	}
	if _, ok := g.Dense(100 + uint64(n)); ok {
		t.Fatal("Dense accepted an id above the set")
	}
}

// TestIDTableSparse64 builds the table over a sparse 64-bit id set, the multi-build
// headroom case doc 02 sizes the id for, and checks the round trip plus that the
// Elias-Fano array keeps it well under eight bytes a node.
func TestIDTableSparse64(t *testing.T) {
	const n = 5000
	rng := rand.New(rand.NewSource(99))
	set := make(map[uint64]struct{}, n)
	ids := make([]uint64, 0, n)
	for len(ids) < n {
		v := rng.Uint64() >> 8 // a 56-bit spread, sparse over the node space
		if _, dup := set[v]; dup {
			continue
		}
		set[v] = struct{}{}
		ids = append(ids, v)
	}

	g := buildRegionWithIDs(t, n, ids)
	if g.idt == nil {
		t.Fatal("sparse ids did not produce an id table")
	}
	for d := 0; d < n; d++ {
		if got := g.Global(d); got != ids[d] {
			t.Fatalf("Global(%d) = %d, want %d", d, got, ids[d])
		}
		dense, ok := g.Dense(ids[d])
		if !ok || dense != d {
			t.Fatalf("Dense(%d) = (%d,%v), want (%d,true)", ids[d], dense, ok, d)
		}
	}

	blob := g.idt.encode()
	perNode := float64(len(blob)) / float64(n)
	// The permutation alone is four bytes a node; the Elias-Fano array of a 56-bit
	// sparse set is a handful of bits a node, so the whole table stays well under the
	// twelve bytes a naive uint64-plus-uint32 pair would cost.
	if perNode >= 12 {
		t.Fatalf("id table is %.1f bytes a node, want < 12", perNode)
	}
	t.Logf("sparse 56-bit id table: %d bytes over %d nodes, %.2f bytes a node", len(blob), n, perNode)
}

// TestIDTableCorruptBlob checks the decoder rejects a truncated table rather than
// reading out of bounds.
func TestIDTableCorruptBlob(t *testing.T) {
	ids := []uint64{40, 10, 30, 20}
	_, blob := computeIDTable(len(ids), ids)
	if blob == nil {
		t.Fatal("expected an id table blob")
	}
	for _, cut := range []int{0, 2, len(blob) - 1} {
		if _, err := decodeIDTable(blob[:cut]); err == nil {
			t.Fatalf("decodeIDTable accepted a %d-byte truncation", cut)
		}
	}
	if _, err := decodeIDTable(blob); err != nil {
		t.Fatalf("decodeIDTable rejected the whole blob: %v", err)
	}
}

// buildRegionWithIDs frames a region over a tiny ring graph with the given global
// ids and opens it, so a test sees the full encode-decode path the reader walks.
func buildRegionWithIDs(t *testing.T, n int, ids []uint64) *Region {
	t.Helper()
	b := NewBuilder(n).WithNodeIDs(ids)
	for x := 0; x < n; x++ {
		b.AddEdge(x, (x+1)%n)
	}
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open region: %v", err)
	}
	return g
}

func benchIDTable(n int) *idTable {
	rng := rand.New(rand.NewSource(1))
	ids := make([]uint64, n)
	for d := range ids {
		ids[d] = 1000 + uint64(d)
	}
	rng.Shuffle(n, func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	return buildIDTable(ids)
}

func BenchmarkIDTableGlobal(b *testing.B) {
	const n = 1 << 20
	t := benchIDTable(n)
	b.ResetTimer()
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink += t.global(i & (n - 1))
	}
	_ = sink
}

func BenchmarkIDTableDense(b *testing.B) {
	const n = 1 << 20
	t := benchIDTable(n)
	// Query the ids in sorted order so the benchmark is the binary search, not a
	// cache-friendly stride; collect them first.
	sorted := make([]uint64, n)
	for rank := 0; rank < n; rank++ {
		sorted[rank] = t.sorted.get(rank)
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		d, _ := t.dense(sorted[i&(n-1)])
		sink += d
	}
	_ = sink
}
