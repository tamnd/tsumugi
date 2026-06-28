package collection

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
)

// writeDocsJSONL writes real documents back to a .jsonl crawl export so the build
// reads them through the same source path as a fresh crawl, preserving the markdown
// body the link extractor recovers edges from.
func writeDocsJSONL(t testing.TB, path string, docs []convert.Document) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// Real crawl bodies carry arbitrary bytes, so the record is JSON-encoded rather
	// than fmt-quoted: Go's %q escapes (\x..) are not valid JSON string escapes.
	enc := json.NewEncoder(f)
	for _, d := range docs {
		rec := struct {
			URL      string `json:"url"`
			Host     string `json:"host"`
			Markdown string `json:"markdown"`
		}{d.URL, d.Host, d.Body}
		if err := enc.Encode(rec); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// maxAbsDiffF compares an in-core float64 rank vector against the streamed float32
// one element by element and returns the largest absolute gap.
func maxAbsDiffF(incore []float64, streamed []float32) float64 {
	var worst float64
	for i := range incore {
		d := math.Abs(incore[i] - float64(streamed[i]))
		if d > worst {
			worst = d
		}
	}
	return worst
}

// streamRanks opens the persisted collection graph artifact and runs the
// out-of-core StreamPageRank over the memory-mapped transpose, the disk-sourced
// path under test. It returns the streamed ranks and the open source so the caller
// can read the same region in core for the parity reference and close the mapping.
func streamRanks(t *testing.T, dir string) ([]float32, *GraphSource) {
	t.Helper()
	src, err := OpenGraphSource(dir)
	if err != nil {
		t.Fatalf("open graph source: %v", err)
	}
	outdeg := graph.OutDegreesFromSource(src.NodeCount(), src.OutDegree)
	got := graph.StreamPageRank(src, outdeg, graph.DefaultPRConfig())
	return got, src
}

// TestCollectionGraphStreamMatchesInCoreSynthetic is the streaming-correctness
// gate over a graph that actually has edges. A real broad crawl resolves almost no
// links (the web graph is cross-shard and one-page-per-host), so its persisted
// graph is near-empty and exercises only the trivial in-lists; the hub-and-spoke
// corpus links every spoke to one hub, so the persisted transpose carries a real
// in-list (the hub's, of length spokes) the streaming scan must decode from the
// mapping. The gate proves the disk-sourced StreamPageRank reproduces the in-core
// PageRank over the very same persisted region to float32 precision, and that the
// recovered graph is non-trivial (the hub outranks every spoke), so the parity is
// over a graph with structure, not an empty one.
func TestCollectionGraphStreamMatchesInCoreSynthetic(t *testing.T) {
	const spokes = 2000
	tmp := t.TempDir()
	in := writeLinkedJSONL(t, tmp, "in.jsonl", spokes)
	out := filepath.Join(tmp, "col")
	if _, err := Build(Options{Source: in, Out: out, ShardSize: 1 << 20}); err != nil {
		t.Fatalf("build: %v", err)
	}

	got, src := streamRanks(t, out)
	defer func() { _ = src.Close() }()

	// In-core PageRank over the identical persisted region: node ids and edges are
	// the same bytes, so the only difference from the stream is the working set.
	want := graph.PageRank(src.region, graph.DefaultPRConfig())
	if len(got) != len(want) {
		t.Fatalf("streamed %d ranks, in-core %d", len(got), len(want))
	}
	if d := maxAbsDiffF(want, got); d >= 1e-6 {
		t.Fatalf("streamed PageRank diverged from in-core by %g (>= 1e-6)", d)
	}

	// The graph is non-trivial: exactly one node (the hub) is the in-target of every
	// spoke, so it must hold the single highest rank.
	hub, hubRank := -1, float32(-1)
	for i, r := range got {
		if r > hubRank {
			hub, hubRank = i, r
		}
	}
	indeg := len(src.region.InNeighbors(hub))
	if indeg < spokes/2 {
		t.Fatalf("top-ranked node %d has in-degree %d, not the hub (want ~%d)", hub, indeg, spokes)
	}
	t.Logf("spokes=%d N=%d hub=%d hubInDeg=%d maxAbsDiff(stream,incore) below 1e-6", spokes, len(got), hub, indeg)
}

// TestCollectionGraphStreamMemoryBudgetSynthetic proves the streaming run's
// resident memory tracks the node count, not the edge count: it keeps the two
// float32 rank vectors and the uint32 out-degree array (12 bytes a node) plus one
// decoded in-list at a time, never the adjacency. It builds a graph with many more
// edges than the resident budget would allow if the adjacency were held, runs the
// stream, and measures the live heap the run leaves behind, not its transient churn.
//
// The measurement collects after the call with the result still reachable, so the
// per-iteration in-lists the scan decodes and discards are freed before the snapshot
// and the delta counts only what survives: the returned rank vector. A naive
// delta taken without that collection would scale with edges times iterations,
// because every decoded in-list the run threw away would still be counted as growth,
// which measures churn rather than residency and is exactly what this gate must not do.
func TestCollectionGraphStreamMemoryBudgetSynthetic(t *testing.T) {
	const spokes = 4000 // N ~ 4001 nodes, ~8000 directed edges
	tmp := t.TempDir()
	in := writeLinkedJSONL(t, tmp, "in.jsonl", spokes)
	out := filepath.Join(tmp, "col")
	if _, err := Build(Options{Source: in, Out: out, ShardSize: 1 << 20}); err != nil {
		t.Fatalf("build: %v", err)
	}

	src, err := OpenGraphSource(out)
	if err != nil {
		t.Fatalf("open graph source: %v", err)
	}
	defer func() { _ = src.Close() }()

	// The region must be stored uncompressed so the mapped bytes alias the heap
	// nowhere: a zstd region would decompress the whole adjacency into a buffer,
	// defeating the streaming budget.
	desc, ok := src.r.RegionDesc(tsumugi.RegionGraph)
	if !ok {
		t.Fatal("graph artifact has no graph region")
	}
	if desc.Codec != tsumugi.CodecNone {
		t.Fatalf("graph region codec %v, want CodecNone for zero-copy streaming", desc.Codec)
	}

	n := src.NodeCount()
	edges := int(src.region.EdgeCount())
	// The out-degree array is allocated before the baseline so it is not counted as
	// growth; it is part of the resident 12 bytes a node either way.
	outdeg := graph.OutDegreesFromSource(n, src.OutDegree)

	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)
	ranks := graph.StreamPageRank(src, outdeg, graph.DefaultPRConfig())
	// Collect with the result reachable: the transient in-lists are freed, the rank
	// vector survives, so the delta is the live residue, not the per-iteration churn.
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	runtime.KeepAlive(ranks)
	runtime.KeepAlive(outdeg)

	var delta int64
	if m1.HeapAlloc > m0.HeapAlloc {
		delta = int64(m1.HeapAlloc - m0.HeapAlloc)
	}
	// What survives the call is the returned float32 rank vector (4 bytes a node);
	// the second vector and every decoded in-list are unreachable and collected. The
	// bound is a few multiples of that to absorb allocator slack, and is proportional
	// to N, not edges: at this edge count a retained adjacency would dwarf it.
	budget := int64(32 * n)
	if delta > budget {
		t.Fatalf("stream resident growth %d bytes for N=%d edges=%d exceeds N-proportional budget %d", delta, n, edges, budget)
	}
	t.Logf("N=%d edges=%d streamResidentDelta=%dB (%.1f B/node), budget=%dB, codec=None", n, edges, delta, float64(delta)/float64(n), budget)
}

// TestCollectionGraphStreamMatchesInCoreCCrawl runs the parity gate on the real
// crawl export: the build persists the collection graph artifact, and the
// disk-sourced StreamPageRank over the memory-mapped transpose reproduces the
// in-core PageRank over the same region to float32 precision on the real
// distribution, with the rank vector normalizing. It confirms the format and the
// streaming reader on real data even though that broad sample's graph is sparse.
func TestCollectionGraphStreamMatchesInCoreCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
		if len(docs) >= 8000 {
			break
		}
	}
	_ = src.Close()
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	tmp := t.TempDir()
	in := filepath.Join(tmp, "in.jsonl")
	writeDocsJSONL(t, in, docs)
	out := filepath.Join(tmp, "col")
	res, err := Build(Options{Source: in, Out: out, ShardSize: 100000})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	got, gsrc := streamRanks(t, out)
	defer func() { _ = gsrc.Close() }()

	if gsrc.NodeCount() != res.Docs {
		t.Fatalf("graph node count %d != collection docs %d", gsrc.NodeCount(), res.Docs)
	}
	want := graph.PageRank(gsrc.region, graph.DefaultPRConfig())
	if d := maxAbsDiffF(want, got); d >= 1e-6 {
		t.Fatalf("streamed PageRank diverged from in-core by %g (>= 1e-6) on ccrawl", d)
	}
	var sum float64
	for _, r := range got {
		sum += float64(r)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("streamed rank sum %g not ~1", sum)
	}
	t.Logf("docs=%d edges=%d streamed PageRank matches in-core below 1e-6, sum=%.6f", res.Docs, gsrc.region.EdgeCount(), sum)
}

// BenchmarkCollectionGraphStream measures one out-of-core PageRank pass over the
// memory-mapped collection graph, the rank-time cost the scale path pays. The build
// is excluded from the timer; only the streaming iteration is measured.
func BenchmarkCollectionGraphStream(b *testing.B) {
	const spokes = 4000
	tmp := b.TempDir()
	in := writeLinkedJSONL(b, tmp, "in.jsonl", spokes)
	out := filepath.Join(tmp, "col")
	if _, err := Build(Options{Source: in, Out: out, ShardSize: 1 << 20}); err != nil {
		b.Fatalf("build: %v", err)
	}
	src, err := OpenGraphSource(out)
	if err != nil {
		b.Fatalf("open graph source: %v", err)
	}
	defer func() { _ = src.Close() }()
	outdeg := graph.OutDegreesFromSource(src.NodeCount(), src.OutDegree)
	b.ReportMetric(8, "residentB/node")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = graph.StreamPageRank(src, outdeg, graph.DefaultPRConfig())
	}
}
