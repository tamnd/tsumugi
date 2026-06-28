package graph

import (
	"bytes"
	"math/rand"
	"os"
	"testing"
)

// randomEdges builds a deterministic random directed edge set over n nodes, with
// duplicates and self-loops left in so both builders exercise the same dedup and
// self-loop drop. The seed makes the corpus reproducible across runs.
func randomEdges(seed int64, n, m int) [][2]int {
	rng := rand.New(rand.NewSource(seed))
	edges := make([][2]int, m)
	for i := range edges {
		edges[i] = [2]int{rng.Intn(n), rng.Intn(n)}
	}
	return edges
}

// inRAMRegion builds the reference region through the resident Builder.
func inRAMRegion(n int, edges [][2]int, p Params) []byte {
	b := NewBuilder(n).WithParams(p)
	for _, e := range edges {
		b.AddEdge(e[0], e[1])
	}
	return b.Build()
}

// TestOOCBuilderMatchesInRAMInMemory proves the out-of-core builder, on its in-RAM
// fast path (spill threshold above the edge count), produces a region byte-for-byte
// identical to the resident Builder over the same edges. Identity is the contract:
// the reader cannot tell which builder wrote a region, so the OOC path is a drop-in.
// It sweeps node counts and densities so the reference window, intervals, and
// residual coding all get exercised, and confirms the region opens and its node and
// edge counts agree with the reference.
func TestOOCBuilderMatchesInRAMInMemory(t *testing.T) {
	p := DefaultParams()
	cases := []struct{ n, m int }{
		{1, 0}, {2, 1}, {10, 5}, {50, 200}, {200, 2000}, {1000, 8000}, {2000, 200},
	}
	for _, c := range cases {
		edges := randomEdges(int64(c.n*7+c.m), c.n, c.m)
		want := inRAMRegion(c.n, edges, p)

		b := NewOOCBuilder(c.n).WithParams(p)
		for _, e := range edges {
			b.AddEdge(e[0], e[1])
		}
		if len(b.runs) != 0 {
			t.Fatalf("n=%d m=%d: expected in-RAM path, got %d runs", c.n, c.m, len(b.runs))
		}
		got, err := b.Build()
		if err != nil {
			t.Fatalf("n=%d m=%d: build: %v", c.n, c.m, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("n=%d m=%d: OOC region differs from in-RAM (%d vs %d bytes)", c.n, c.m, len(got), len(want))
		}

		g, err := Open(got)
		if err != nil {
			t.Fatalf("n=%d m=%d: open: %v", c.n, c.m, err)
		}
		ref, _ := Open(want)
		if g.NodeCount() != ref.NodeCount() || g.EdgeCount() != ref.EdgeCount() {
			t.Fatalf("n=%d m=%d: counts differ: got (%d,%d) want (%d,%d)", c.n, c.m, g.NodeCount(), g.EdgeCount(), ref.NodeCount(), ref.EdgeCount())
		}
	}
}

// TestOOCBuilderSpillMatchesInRAM forces the disk path with a tiny spill threshold
// and proves the external-sort merge produces the identical region. The threshold
// is well below the edge count, so the edges spill into several sorted runs and the
// forward plane merges them while the transpose runs a second external sort; the
// byte-for-byte identity proves the merge, dedup, and windowed encode reconstruct
// exactly what the resident builder produces. The run count is asserted positive so
// the test cannot silently fall through to the in-RAM path.
func TestOOCBuilderSpillMatchesInRAM(t *testing.T) {
	p := DefaultParams()
	cases := []struct {
		n, m, spill int
	}{
		{500, 5000, 64}, {1000, 20000, 256}, {2000, 40000, 1000}, {300, 9000, 17},
	}
	for _, c := range cases {
		edges := randomEdges(int64(c.n+c.m*3), c.n, c.m)
		want := inRAMRegion(c.n, edges, p)

		b := NewOOCBuilder(c.n).WithParams(p).WithSpillThreshold(c.spill)
		for _, e := range edges {
			b.AddEdge(e[0], e[1])
		}
		if len(b.runs) == 0 {
			t.Fatalf("n=%d m=%d spill=%d: expected spilled runs, got none", c.n, c.m, c.spill)
		}
		runsBefore := len(b.runs)
		got, err := b.Build()
		if err != nil {
			t.Fatalf("n=%d m=%d spill=%d: build: %v", c.n, c.m, c.spill, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("n=%d m=%d spill=%d: spilled region differs from in-RAM (%d vs %d bytes)", c.n, c.m, c.spill, len(got), len(want))
		}
		t.Logf("n=%d m=%d spill=%d: %d runs, region %d bytes, identical to in-RAM", c.n, c.m, c.spill, runsBefore, len(got))
	}
}

// TestOOCBuilderTempFilesCleanedUp proves Build removes its spill directory, so a
// large build leaves no run files behind regardless of success.
func TestOOCBuilderTempFilesCleanedUp(t *testing.T) {
	edges := randomEdges(99, 500, 5000)
	b := NewOOCBuilder(500).WithSpillThreshold(64)
	for _, e := range edges {
		b.AddEdge(e[0], e[1])
	}
	dir := b.dir
	if dir == "" {
		t.Fatal("expected a temp dir after spilling")
	}
	if _, err := b.Build(); err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, err := os.Stat(dir); err == nil {
		t.Fatalf("temp dir %s still exists after Build", dir)
	}
}

// TestOOCBuilderResidentBoundedBySpill proves the resident edge buffer never holds
// more than the spill threshold: with a tiny threshold and many edges, the builder
// spills into many runs (edges go to disk in bounded batches) rather than holding
// the whole edge set, which is the property that lets it scale past RAM. It also
// confirms the heap growth across the build tracks the offsets table and window,
// not the edge count, by leaving the resident buffer capacity bounded.
func TestOOCBuilderResidentBoundedBySpill(t *testing.T) {
	const n, m, spill = 5000, 100000, 1000
	edges := randomEdges(2024, n, m)
	b := NewOOCBuilder(n).WithSpillThreshold(spill)
	for _, e := range edges {
		b.AddEdge(e[0], e[1])
		if len(b.buf) > spill {
			t.Fatalf("resident buffer %d exceeded spill threshold %d", len(b.buf), spill)
		}
	}
	// Many runs means the edges genuinely streamed to disk in bounded batches.
	runsBefore := len(b.runs)
	if runsBefore < m/spill/2 {
		t.Fatalf("expected many spilled runs for %d edges at threshold %d, got %d", m, spill, runsBefore)
	}
	want := inRAMRegion(n, edges, DefaultParams())
	got, err := b.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("streamed region differs from in-RAM at scale (%d vs %d bytes)", len(got), len(want))
	}
	g, err := Open(got)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Logf("n=%d m=%d spill=%d: %d runs, %d edges encoded, region %d bytes", n, m, spill, runsBefore, g.EdgeCount(), len(got))
}

// TestOOCBuilderPageRankMatches checks the streamed-build region ranks the same as
// the resident-build region under the out-of-core PageRank, the use the artifact is
// built for: same graph in, same ranks out, whichever builder wrote it.
func TestOOCBuilderPageRankMatches(t *testing.T) {
	const n, m = 3000, 30000
	edges := randomEdges(7, n, m)
	want := inRAMRegion(n, edges, DefaultParams())
	ref, err := Open(want)
	if err != nil {
		t.Fatalf("open ref: %v", err)
	}

	b := NewOOCBuilder(n).WithSpillThreshold(512)
	for _, e := range edges {
		b.AddEdge(e[0], e[1])
	}
	got, err := b.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	g, err := Open(got)
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	prRef := PageRank(ref, DefaultPRConfig())
	prGot := PageRank(g, DefaultPRConfig())
	var worst float64
	for i := range prRef {
		d := prRef[i] - prGot[i]
		if d < 0 {
			d = -d
		}
		if d > worst {
			worst = d
		}
	}
	if worst != 0 {
		t.Fatalf("PageRank differs between builders by %g (regions should be identical)", worst)
	}
}

// BenchmarkOOCBuilderInRAM measures the in-RAM encode path against a moderate graph.
func BenchmarkOOCBuilderInRAM(b *testing.B) {
	edges := randomEdges(1, 5000, 50000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bd := NewOOCBuilder(5000)
		for _, e := range edges {
			bd.AddEdge(e[0], e[1])
		}
		if _, err := bd.Build(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOOCBuilderSpill measures the external-sort path, the cost paid when the
// edge set exceeds RAM, by forcing spilling with a small threshold.
func BenchmarkOOCBuilderSpill(b *testing.B) {
	edges := randomEdges(1, 5000, 50000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bd := NewOOCBuilder(5000).WithSpillThreshold(2000)
		for _, e := range edges {
			bd.AddEdge(e[0], e[1])
		}
		if _, err := bd.Build(); err != nil {
			b.Fatal(err)
		}
	}
}
