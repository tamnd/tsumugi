package lexical_test

import (
	"bytes"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/lexical"
)

// The M14 claim is that the SPIMI external-merge build produces exactly the region the
// in-memory builder produces, so it scales a shard build past RAM without changing a
// single output byte. These tests prove it the strongest way available: build the same
// documents both ways and compare the encoded regions byte for byte. A byte-identical
// region means every downstream guarantee already tested against the in-memory builder,
// scoring, block-max pruning, the dictionary and bloom layout, holds for the SPIMI build
// too, with no separate re-proof needed.

// spimiDoc is one document with the three fields the build indexes, the title, the body,
// and the url, mirroring the collection build's field mapping.
type spimiDoc struct {
	title, body, url string
}

func (d spimiDoc) fields() map[lexical.Field]string {
	return map[lexical.Field]string{
		lexical.FieldTitle: d.title,
		lexical.FieldBody:  d.body,
		lexical.FieldURL:   d.url,
	}
}

// buildInMemory builds a region with the in-memory builder.
func buildInMemory(docs []spimiDoc) []byte {
	b := lexical.NewBuilder(lexical.DefaultParams())
	for i, d := range docs {
		b.AddDoc(uint32(i), d.fields())
	}
	return b.Build()
}

// buildSpimi builds the same documents with the external-merge builder, spilling under
// dir and holding at most maxBytes of records in memory before each spill.
func buildSpimi(t testing.TB, docs []spimiDoc, dir string, maxBytes int) ([]byte, int) {
	t.Helper()
	b := lexical.NewSpimiBuilder(lexical.DefaultParams(), dir, maxBytes)
	for i, d := range docs {
		b.AddDoc(uint32(i), d.fields())
	}
	out, err := b.Build()
	if err != nil {
		t.Fatalf("spimi build: %v", err)
	}
	return out, b.Spills()
}

// syntheticCorpus generates n documents with a Zipf-ish vocabulary: a few common terms in
// almost every document and a long tail of rare terms, the shape that makes posting lists
// of very different lengths and so exercises both multi-block lists and singleton lists.
func syntheticCorpus(n int) []spimiDoc {
	docs := make([]spimiDoc, n)
	for i := 0; i < n; i++ {
		// Common terms appear everywhere; mid terms cycle; a rare term is unique per doc.
		body := fmt.Sprintf("common alpha beta common gamma term%d term%d rare%d",
			i%7, i%131, i)
		docs[i] = spimiDoc{
			title: fmt.Sprintf("title common heading%d", i%11),
			body:  body,
			url:   fmt.Sprintf("https://host%d.example.com/page/%d", i%97, i),
		}
	}
	return docs
}

// TestSpimiMatchesInMemorySingleRun checks the base case: when the buffer never fills,
// the merge is over a single run and the output must still equal the in-memory builder's.
func TestSpimiMatchesInMemorySingleRun(t *testing.T) {
	docs := syntheticCorpus(500)
	want := buildInMemory(docs)
	got, spills := buildSpimi(t, docs, t.TempDir(), 1<<30)
	if spills != 1 {
		t.Fatalf("expected a single spill with a large budget, got %d", spills)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("spimi region differs from in-memory region: got %d bytes, want %d", len(got), len(want))
	}
}

// TestSpimiMatchesInMemoryManySpills is the keystone: a tiny buffer budget forces many
// spills, so the output is assembled entirely from an external k-way merge across many
// run files, and it must still be byte-identical to the in-memory build.
func TestSpimiMatchesInMemoryManySpills(t *testing.T) {
	docs := syntheticCorpus(5000)
	want := buildInMemory(docs)
	got, spills := buildSpimi(t, docs, t.TempDir(), 16<<10)
	if spills < 4 {
		t.Fatalf("expected several spills under a tiny budget, got %d", spills)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("spimi region differs from in-memory region after %d spills: got %d bytes, want %d",
			spills, len(got), len(want))
	}
	t.Logf("byte-identical region over %d spills, %d bytes", spills, len(got))
}

// TestSpimiEmpty checks the degenerate corpus: no documents at all builds the same empty
// region both ways.
func TestSpimiEmpty(t *testing.T) {
	want := buildInMemory(nil)
	got, _ := buildSpimi(t, nil, t.TempDir(), 1<<20)
	if !bytes.Equal(got, want) {
		t.Fatalf("empty spimi region differs from empty in-memory region")
	}
}

// ccrawlSpimiDocs reads up to limit documents from the real crawl export, mapping each to
// the three indexed fields the same way the collection build does.
func ccrawlSpimiDocs(t testing.TB, limit int) []spimiDoc {
	t.Helper()
	src, err := convert.OpenSource(ccrawlSpimiParquet)
	if err != nil {
		t.Skipf("ccrawl export not available: %v", err)
	}
	defer func() { _ = src.Close() }()
	var docs []spimiDoc
	for len(docs) < limit {
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
		a := analyze.Document(d)
		docs = append(docs, spimiDoc{title: a.Title, body: d.Body, url: d.URL})
	}
	return docs
}

// ccrawlSpimiParquet is the real Common Crawl markdown shard the scale tests build from,
// the language and content distribution the engine is meant to serve. Tests skip when it
// is absent so the suite still runs without the data.
const ccrawlSpimiParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// TestSpimiMatchesInMemoryCCrawl proves the byte-identical guarantee on real crawl data,
// the content distribution the spec targets, with a small budget so the build runs the
// external merge over many spills of real postings.
func TestSpimiMatchesInMemoryCCrawl(t *testing.T) {
	docs := ccrawlSpimiDocs(t, 8000)
	if len(docs) == 0 {
		t.Skip("no ccrawl documents")
	}
	want := buildInMemory(docs)
	got, spills := buildSpimi(t, docs, t.TempDir(), 1<<20)
	if spills < 2 {
		t.Logf("warning: only %d spill(s); corpus fit the budget", spills)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("ccrawl spimi region differs from in-memory region: got %d bytes, want %d (%d docs, %d spills)",
			len(got), len(want), len(docs), spills)
	}
	t.Logf("ccrawl: byte-identical region over %d docs, %d spills, %d bytes", len(docs), spills, len(got))
}

// TestSpimiBoundedMemory shows the point of the external merge: building a corpus whose
// postings dwarf the buffer keeps peak live memory far below the in-memory builder, which
// retains the whole inverted index. It samples HeapInuse during each build and compares
// the peaks. The corpus is sized so the gap is large enough to clear runtime noise.
func TestSpimiBoundedMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("memory sampling test skipped in -short")
	}
	docs := syntheticCorpus(60000)

	inMemPeak := sampleHeapPeak(func() {
		_ = buildInMemory(docs)
	})
	spimiPeak := sampleHeapPeak(func() {
		b := lexical.NewSpimiBuilder(lexical.DefaultParams(), t.TempDir(), 256<<10)
		for i, d := range docs {
			b.AddDoc(uint32(i), d.fields())
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
// reading seen. It is a coarse peak, enough to show the order-of-magnitude difference the
// external merge buys, not a precise allocator measurement. The peak lives inside the
// sampler goroutine and comes back over a channel, so there is no variable shared between
// the sampler and the caller to race on.
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

// BenchmarkBuildCCrawl compares the two builders on real crawl data, reporting allocation
// volume per build. The in-memory builder allocates the whole index live; the SPIMI build
// trades that for streamed run files, which is the cost that keeps peak memory flat.
func BenchmarkBuildCCrawl(b *testing.B) {
	docs := ccrawlSpimiDocs(b, 8000)
	if len(docs) == 0 {
		b.Skip("no ccrawl documents")
	}
	b.Run("in-memory", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = buildInMemory(docs)
		}
	})
	b.Run("spimi", func(b *testing.B) {
		dir := b.TempDir()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			bd := lexical.NewSpimiBuilder(lexical.DefaultParams(), dir, 1<<20)
			for j, d := range docs {
				bd.AddDoc(uint32(j), d.fields())
			}
			if _, err := bd.Build(); err != nil {
				b.Fatalf("spimi build: %v", err)
			}
		}
	})
}
