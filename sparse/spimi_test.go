package sparse_test

import (
	"bytes"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/sparse"
)

// The M14 claim for the sparse plane matches the lexical one: the SPIMI external-merge
// build produces exactly the region the in-memory builder produces, so it scales an impact
// shard build past RAM without changing a byte. These tests prove it the strongest way,
// building the same impact records both ways and comparing the encoded regions byte for
// byte. They lean on duplicate (term, docID) records so the merge's keep-the-strongest
// dedup is exercised against the in-memory dedupByDoc, the one place the two paths could
// diverge.

// addOp is one impact posting fed to a builder, replayed identically into both so a
// byte difference is the build path's alone.
type addOp struct {
	term   string
	docID  uint32
	weight float64
}

// buildInMemory builds a region with the in-memory builder over the given adds.
func buildInMemory(docCount uint32, ops []addOp) []byte {
	b := sparse.NewBuilder(docCount)
	for _, op := range ops {
		b.Add(op.term, op.docID, op.weight)
	}
	return b.Build()
}

// buildSpimi builds the same adds with the external-merge builder.
func buildSpimi(t testing.TB, docCount uint32, ops []addOp, dir string, maxBytes int) ([]byte, int) {
	t.Helper()
	b := sparse.NewSpimiBuilder(docCount, dir, maxBytes)
	for _, op := range ops {
		b.Add(op.term, op.docID, op.weight)
	}
	out, err := b.Build()
	if err != nil {
		t.Fatalf("spimi build: %v", err)
	}
	return out, b.Spills()
}

// syntheticOps generates impact records over n documents with a skewed vocabulary and, for
// every document, a deliberate duplicate of one term at two different weights so the
// dedup-keeps-strongest path is always exercised.
func syntheticOps(n int) []addOp {
	var ops []addOp
	for d := 0; d < n; d++ {
		id := uint32(d)
		ops = append(ops,
			addOp{"common", id, 1.0 + float64(d%5)},
			addOp{fmt.Sprintf("mid%d", d%97), id, 2.5 + float64(d%3)},
			addOp{fmt.Sprintf("rare%d", d), id, 0.5 + float64(d%7)},
		)
		// Duplicate the common term in the same doc at a different weight; the build must
		// keep the stronger of the two, the same as the in-memory dedup.
		ops = append(ops, addOp{"common", id, 3.0 + float64(d%2)})
	}
	return ops
}

// TestSpimiMatchesInMemorySingleRun is the base case: a large budget means one run, and
// the output must still equal the in-memory build.
func TestSpimiMatchesInMemorySingleRun(t *testing.T) {
	ops := syntheticOps(400)
	want := buildInMemory(400, ops)
	got, spills := buildSpimi(t, 400, ops, t.TempDir(), 1<<30)
	if spills != 1 {
		t.Fatalf("expected a single spill with a large budget, got %d", spills)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("spimi region differs from in-memory region: got %d bytes, want %d", len(got), len(want))
	}
}

// TestSpimiMatchesInMemoryManySpills is the keystone: a tiny budget forces many spills, so
// the region is assembled from an external merge across many run files, with the
// dedup-keeps-strongest having to reconcile duplicate postings split across runs, and it
// must still be byte-identical.
func TestSpimiMatchesInMemoryManySpills(t *testing.T) {
	ops := syntheticOps(4000)
	want := buildInMemory(4000, ops)
	got, spills := buildSpimi(t, 4000, ops, t.TempDir(), 8<<10)
	if spills < 4 {
		t.Fatalf("expected several spills under a tiny budget, got %d", spills)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("spimi region differs from in-memory region after %d spills: got %d bytes, want %d",
			spills, len(got), len(want))
	}
	t.Logf("byte-identical region over %d spills, %d bytes", spills, len(got))
}

// TestSpimiEmpty checks the degenerate corpus: no postings builds the same empty region
// both ways, including the zero weight range the quantizer falls back to.
func TestSpimiEmpty(t *testing.T) {
	want := buildInMemory(100, nil)
	got, _ := buildSpimi(t, 100, nil, t.TempDir(), 1<<20)
	if !bytes.Equal(got, want) {
		t.Fatalf("empty spimi region differs from empty in-memory region")
	}
}

// ccrawlSparseParquet is the real Common Crawl markdown shard the scale tests build from.
// Tests skip when it is absent so the suite still runs without the data.
const ccrawlSparseParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// ccrawlOps reads up to limit documents from the real crawl export and turns each into
// impact records by counting body term frequencies, a stand-in for a learned impact model
// that still exercises the build over the real vocabulary and document-length distribution.
// Repeated terms in a document produce one record per occurrence with the running count as
// the weight, so the strongest per (term, docID) is the final count, which the dedup must
// recover.
func ccrawlOps(t testing.TB, limit int) ([]addOp, uint32) {
	t.Helper()
	src, err := convert.OpenSource(ccrawlSparseParquet)
	if err != nil {
		t.Skipf("ccrawl export not available: %v", err)
	}
	defer func() { _ = src.Close() }()
	var ops []addOp
	var docID uint32
	for docID < uint32(limit) {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read ccrawl: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		counts := map[string]int{}
		for _, tok := range lexical.Analyze(d.Body) {
			counts[tok]++
			ops = append(ops, addOp{term: tok, docID: docID, weight: float64(counts[tok])})
		}
		docID++
	}
	return ops, docID
}

// TestSpimiMatchesInMemoryCCrawl proves the byte-identical guarantee on real crawl
// vocabulary, with a small budget so the build runs the external merge over many spills.
func TestSpimiMatchesInMemoryCCrawl(t *testing.T) {
	ops, docCount := ccrawlOps(t, 4000)
	if len(ops) == 0 {
		t.Skip("no ccrawl documents")
	}
	want := buildInMemory(docCount, ops)
	got, spills := buildSpimi(t, docCount, ops, t.TempDir(), 1<<20)
	if spills < 2 {
		t.Logf("warning: only %d spill(s); corpus fit the budget", spills)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("ccrawl spimi region differs from in-memory region: got %d bytes, want %d (%d docs, %d records, %d spills)",
			len(got), len(want), docCount, len(ops), spills)
	}
	t.Logf("ccrawl: byte-identical region over %d docs, %d records, %d spills, %d bytes",
		docCount, len(ops), spills, len(got))
}

// TestSpimiBoundedMemory shows the external merge holds peak live memory below the
// in-memory builder, which retains every posting. It samples HeapInuse during each build
// and compares the peaks on a corpus whose postings dwarf the buffer.
func TestSpimiBoundedMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("memory sampling test skipped in -short")
	}
	const n = 50000
	ops := syntheticOps(n)

	inMemPeak := sampleHeapPeak(func() {
		_ = buildInMemory(n, ops)
	})
	spimiPeak := sampleHeapPeak(func() {
		b := sparse.NewSpimiBuilder(n, t.TempDir(), 256<<10)
		for _, op := range ops {
			b.Add(op.term, op.docID, op.weight)
		}
		if _, err := b.Build(); err != nil {
			t.Fatalf("spimi build: %v", err)
		}
	})

	t.Logf("peak HeapInuse: in-memory %.1f MiB, spimi %.1f MiB",
		float64(inMemPeak)/(1<<20), float64(spimiPeak)/(1<<20))
	if spimiPeak >= inMemPeak {
		t.Fatalf("expected spimi peak below in-memory peak, got spimi %d >= in-memory %d", spimiPeak, inMemPeak)
	}
}

// sampleHeapPeak runs f while polling HeapInuse on a ticker and returns the highest
// reading. The peak lives inside the sampler goroutine and comes back over a channel, so
// nothing is shared between the sampler and the caller to race on.
func sampleHeapPeak(f func()) uint64 {
	runtime.GC()
	stop := make(chan struct{})
	result := make(chan uint64, 1)
	go func() {
		var ms runtime.MemStats
		var peak uint64
		tick := time.NewTicker(time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				result <- peak
				return
			case <-tick.C:
				runtime.ReadMemStats(&ms)
				if ms.HeapInuse > peak {
					peak = ms.HeapInuse
				}
			}
		}
	}()
	f()
	close(stop)
	return <-result
}

// BenchmarkBuildCCrawl compares the two builders on real crawl vocabulary, reporting
// allocation volume per build.
func BenchmarkBuildCCrawl(b *testing.B) {
	ops, docCount := ccrawlOps(b, 4000)
	if len(ops) == 0 {
		b.Skip("no ccrawl documents")
	}
	b.Run("in-memory", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = buildInMemory(docCount, ops)
		}
	})
	b.Run("spimi", func(b *testing.B) {
		dir := b.TempDir()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			bd := sparse.NewSpimiBuilder(docCount, dir, 1<<20)
			for _, op := range ops {
				bd.Add(op.term, op.docID, op.weight)
			}
			if _, err := bd.Build(); err != nil {
				b.Fatalf("spimi build: %v", err)
			}
		}
	})
}
