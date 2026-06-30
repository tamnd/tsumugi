package collection

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
)

// signalColumns lists every link-signal feature column the build bakes off the
// collection-wide signals, paired with the field of graphSignals it comes from. The
// build-output equivalence gate reads each column back from the shards and checks it
// against the merged-graph oracle, so the list is the contract the reorder must hold:
// every signal globalSignals computes must reach the same feature column whether the
// build ranks over one merged in-core graph (the old path) or the per-shard regions
// (the reorder).
func signalColumns(s graphSignals) []struct {
	id  feature.FeatureID
	val []float64
} {
	ints := func(v []int) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out
	}
	u32 := func(v []uint32) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out
	}
	return []struct {
		id  feature.FeatureID
		val []float64
	}{
		{feature.FeatPageRank, s.pageRank},
		{feature.FeatHostRank, s.hostRank},
		{feature.FeatDomainRank, s.domainRank},
		{feature.FeatTrust, s.trust},
		{feature.FeatSpamMass, s.spamMass},
		{feature.FeatInDegree, ints(s.inDegree)},
		{feature.FeatLinkingDomains, ints(s.linkingDomains)},
		{feature.FeatLinkingHosts, ints(s.linkingHosts)},
		{feature.FeatReciprocity, s.reciprocity},
		{feature.FeatHostLinkDiv, s.hostLinkDiv},
		{feature.FeatNearDup, s.nearDup},
		{feature.FeatOutboundSpam, s.outboundSpam},
		{feature.FeatLanguage, u32(s.langID)},
		{feature.FeatStaticRank, s.staticRank},
	}
}

// columnTol is the per-column tolerance the baked feature column is allowed to differ
// from the merged-graph oracle by: one and a half byte-quantization steps over the
// column's observed range, plus a floor for a column that is flat. The feature matrix
// stores each value as one byte across the column's min..max, so the inherent
// representation error is a step of range/255; the reorder's cross-shard ranks match the
// merged ranks to float32 precision, far finer than that step, so the only material
// difference the gate can see is a value landing on the other side of a quantization
// boundary, which the one-and-a-half-step band absorbs.
func columnTol(oracle []float64) float64 {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, v := range oracle {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if !(hi > lo) {
		return 1e-6
	}
	return 1.5*(hi-lo)/255 + 1e-6
}

// openFeature returns a shard's feature reader and a closer.
func openFeature(t testing.TB, path string) (*feature.Region, func()) {
	t.Helper()
	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	b, err := r.Region(tsumugi.RegionFeature)
	if err != nil {
		_ = r.Close()
		t.Fatalf("feature region: %v", err)
	}
	fr, err := feature.Open(b)
	if err != nil {
		_ = r.Close()
		t.Fatalf("open feature: %v", err)
	}
	return fr, func() { _ = r.Close() }
}

// assertBuildMatchesMergedOracle builds a collection at the given shard size, then
// proves the reorder's build output equals the merged-graph oracle end to end: every
// baked link-signal column read back from the shards matches globalSignals over the same
// documents within the byte-quantization band, and every shard's stored graph region is
// byte-identical to the region buildShardGraph produces from the same slice. The first
// is the signal equivalence the reorder rests on; the second proves the graph plane the
// signals were computed from is the very plane the shard persists.
func assertBuildMatchesMergedOracle(t *testing.T, src, out string, shardSize int) (docs int, shards int) {
	t.Helper()
	res, err := Build(Options{Source: src, Out: out, ShardSize: shardSize})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	layoutDocs, gids, _ := buildLayout(t, src)
	dir := buildDir(layoutDocs)
	oracle, _, _ := globalSignals(layoutDocs, nil, nil)
	cols := signalColumns(oracle)

	infos, err := List(out)
	if err != nil {
		t.Fatalf("list: %v", err)
	}

	base := 0
	for s, in := range infos {
		fr, closeShard := openFeature(t, in.Path)
		n := int(fr.DocCount())
		for _, c := range cols {
			tol := columnTol(c.val)
			for d := 0; d < n; d++ {
				gi := base + d
				got, ok := fr.Value(uint32(d), c.id)
				if !ok {
					closeShard()
					t.Fatalf("shard %d doc %d: feature %d not set", s, d, c.id)
				}
				if diff := math.Abs(got - c.val[gi]); diff > tol {
					closeShard()
					t.Fatalf("shard %d doc %d feature %d: baked %g, oracle %g, diff %g > tol %g",
						s, d, c.id, got, c.val[gi], diff, tol)
				}
			}
		}
		closeShard()

		// The stored graph region must be the very bytes the signals were computed from:
		// buildShardGraph over this shard's slice reproduces them, and the container hands
		// back the uncompressed region, so the two are byte-identical.
		r, err := tsumugi.Open(in.Path)
		if err != nil {
			t.Fatalf("reopen shard %d: %v", s, err)
		}
		stored, err := r.Region(tsumugi.RegionGraph)
		if err != nil {
			_ = r.Close()
			t.Fatalf("graph region shard %d: %v", s, err)
		}
		want := buildShardGraph(layoutDocs[base:base+n], base, gids, dir)
		if !bytes.Equal(stored, want) {
			_ = r.Close()
			t.Fatalf("shard %d: stored graph region %d bytes != rebuilt %d bytes", s, len(stored), len(want))
		}
		_ = r.Close()

		base += n
	}
	if base != res.Docs {
		t.Fatalf("read back %d docs across shards, build reported %d", base, res.Docs)
	}
	return res.Docs, res.Shards
}

// TestBuildMatchesMergedOracleSynthetic is the build-output equivalence gate on the deep
// cross-linked corpus: the reorder builds the per-shard graphs first and ranks over them,
// and the resulting shards must carry the same signal columns and the same graph regions
// the merged-graph path produced. The corpus links across hosts so the shards genuinely
// carry cross-shard edges, the case the reorder's cross-shard rank loops exist for.
func TestBuildMatchesMergedOracleSynthetic(t *testing.T) {
	tmp := t.TempDir()
	src := writeCrossLinkedJSONL(t, tmp, "web.jsonl")
	out := filepath.Join(tmp, "col")
	docs, shards := assertBuildMatchesMergedOracle(t, src, out, 40)
	if shards < 2 {
		t.Fatalf("want a multi-shard collection, got %d shards", shards)
	}
	t.Logf("docs=%d shards=%d build output matches merged oracle", docs, shards)
}

// TestBuildMatchesMergedOracleCCrawl is the same gate on the real ccrawl collection: the
// reorder's build output must match the merged-graph oracle on the actual corpus, not
// just the synthetic one. The breadth-first crawl resolves few edges so the graph is
// flat, which is why the synthetic gate carries the cross-shard ranks; this proves the
// plumbing and the content signals over real, skewed language and link distributions.
func TestBuildMatchesMergedOracleCCrawl(t *testing.T) {
	if testing.Short() {
		t.Skip("ccrawl gate skipped in -short")
	}
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "col")
	docs, shards := assertBuildMatchesMergedOracle(t, ccrawlGraphParquet, out, 3000)
	if shards < 2 {
		t.Fatalf("want a multi-shard collection from ccrawl, got %d shards", shards)
	}
	t.Logf("docs=%d shards=%d ccrawl build output matches merged oracle", docs, shards)
}

// loadCCrawlDocs reads the ccrawl parquet into the build's final document order, the
// shared input both signal-path measurements rank over.
func loadCCrawlDocs(tb testing.TB) []convert.Document {
	tb.Helper()
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		tb.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		tb.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			tb.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		tb.Skip("no documents in parquet")
	}
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Host != docs[j].Host {
			return docs[i].Host < docs[j].Host
		}
		return docs[i].URL < docs[j].URL
	})
	order := collectionOrder(docs)
	reordered := make([]convert.Document, len(docs))
	for newID, oldID := range order {
		reordered[newID] = docs[oldID]
	}
	return reordered
}

// peakHeap runs f while a background sampler polls the heap, returning the maximum
// HeapInuse observed during the call. Cumulative allocation (the benchmark's B/op) counts
// transient garbage both paths churn through equally; the scalability constraint is the
// peak resident set, the bytes live at once, which is what this samples. It forces a GC
// before starting so the baseline is the live set, not leftover garbage.
func peakHeap(f func()) uint64 {
	runtime.GC()
	stop := make(chan struct{})
	done := make(chan uint64)
	go func() {
		var peak uint64
		var ms runtime.MemStats
		sample := func() {
			runtime.ReadMemStats(&ms)
			if ms.HeapInuse > peak {
				peak = ms.HeapInuse
			}
		}
		// ReadMemStats stops the world, so poll on an interval rather than in a busy loop:
		// a tight loop would serialize the work behind millions of stop-the-world reads.
		for {
			select {
			case <-stop:
				sample()
				done <- peak
				return
			case <-time.After(2 * time.Millisecond):
				sample()
			}
		}
	}()
	f()
	close(stop)
	return <-done
}

// TestSignalPathMemoryDense records what the reorder does and does not do for memory on a
// dense graph, measured rather than asserted. Two facts come out of it.
//
// First, peak transient heap is comparable between the two paths, sometimes higher for the
// sharded path: it holds every per-shard region resident at once because the in-core
// cross-shard rank loops rank all shards jointly, and the cross-shard routing materializes
// the inbound-edge join across shards. So the reorder is not a moderate-scale peak-heap
// win, and this test does not pretend it is.
//
// Second, the structural change is real and is what the reorder exists for: the merged path
// ranks over graph.PageRank's in-core CSR, which materializes the full forward and transpose
// adjacency as flat int arrays, about (2*edges + 2*nodes) eight-byte words. At two billion
// documents and tens of billions of edges that array is hundreds of gigabytes and cannot be
// built on one box at all, which is why the merged path has no 100k-shard form. The sharded
// path never builds it: the same graph lives as the gap-coded per-shard regions, which are
// far smaller per edge and which a disk-backed build can map and page rather than hold, the
// shape an out-of-core cross-shard routing milestone needs. This logs the CSR estimate
// against the summed region bytes so the gap, and how it widens per edge, is on the record.
func TestSignalPathMemoryDense(t *testing.T) {
	if testing.Short() {
		t.Skip("memory measurement skipped in -short")
	}
	tmp := t.TempDir()
	src := writeDenseLinkedJSONL(t, tmp, 200, 60, 24)
	docs, _, err := readSource(src, 0)
	if err != nil {
		t.Fatalf("read source: %v", err)
	}
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Host != docs[j].Host {
			return docs[i].Host < docs[j].Host
		}
		return docs[i].URL < docs[j].URL
	})
	order := collectionOrder(docs)
	reordered := make([]convert.Document, len(docs))
	for newID, oldID := range order {
		reordered[newID] = docs[oldID]
	}
	docs = reordered
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())
	dir := buildDir(docs)
	const shardSize = 3000

	layouts, regions, err := buildShardGraphs(docs, gids, dir, shardSize)
	if err != nil {
		t.Fatalf("build shard graphs: %v", err)
	}
	var regionBytes, edges int
	for i, sl := range layouts {
		regionBytes += len(sl.gregion)
		edges += int(regions[i].EdgeCount())
	}
	// The in-core CSR the merged path builds holds the forward and transpose target arrays
	// (one int per edge each) plus the two offset arrays (one int per node each).
	csrBytes := (2*edges + 2*len(docs)) * 8

	mergedPeak := peakHeap(func() { _, _, _ = globalSignals(docs, nil, nil) })
	shardedPeak := peakHeap(func() {
		ls, rs, err := buildShardGraphs(docs, gids, dir, shardSize)
		if err != nil {
			t.Fatal(err)
		}
		_ = shardedSignals(rs, docs, gids, nil, nil, dir, DefaultPartitionParams())
		runtime.KeepAlive(ls)
	})
	t.Logf("docs=%d edges=%d", len(docs), edges)
	t.Logf("graph representation: merged in-core CSR ~%.1f MiB, sharded per-shard regions %.1f MiB (%.2f bytes/edge vs %.2f)",
		float64(csrBytes)/(1<<20), float64(regionBytes)/(1<<20),
		float64(regionBytes)/float64(edges), float64(csrBytes)/float64(edges))
	t.Logf("transient peak heap: merged %.1f MiB, sharded %.1f MiB (comparable; the reorder is structural, not a moderate-scale peak win)",
		float64(mergedPeak)/(1<<20), float64(shardedPeak)/(1<<20))
}

// writeDenseLinkedJSONL writes a corpus with a real, resolvable link graph: hosts hosts
// of pagesPer pages each, every page linking to linksPer other pages that all exist, so
// the edges resolve (unlike a breadth-first crawl sample, which resolves almost none) and
// the graph genuinely materializes. It is the corpus the memory measurement needs: the
// merged path's in-core ranks scale with the edge count, so a dense graph is where the
// reorder's streaming saves anything; the flat ccrawl sample cannot show it.
func writeDenseLinkedJSONL(tb testing.TB, dir string, hosts, pagesPer, linksPer int) string {
	tb.Helper()
	path := filepath.Join(dir, "dense.jsonl")
	f, err := os.Create(path)
	if err != nil {
		tb.Fatal(err)
	}
	type pg struct{ host, url string }
	var pages []pg
	for h := 0; h < hosts; h++ {
		host := hostName(h)
		for p := 0; p < pagesPer; p++ {
			pages = append(pages, pg{host: host, url: "https://" + host + "/p" + itoa(p)})
		}
	}
	for i, p := range pages {
		body := "# " + p.url + "\n"
		for k := 1; k <= linksPer; k++ {
			tgt := pages[(i+k*7919)%len(pages)]
			if tgt.url != p.url {
				body += "see <" + tgt.url + "> "
			}
		}
		writeRec(tb, f, p.url, p.host, body)
	}
	if err := f.Close(); err != nil {
		tb.Fatal(err)
	}
	return path
}

func hostName(h int) string { return "h" + itoa(h%97) + ".dom" + itoa(h/97) + ".com" }

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [12]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// BenchmarkSignalPathsDense measures the memory the reorder saves on a graph with real
// edge density: the merged path materializes the in-core CSR the rank functions read, the
// sharded path streams the cross-shard loops over the per-shard regions. Run with
// -benchtime=1x.
func BenchmarkSignalPathsDense(b *testing.B) {
	tmp := b.TempDir()
	src := writeDenseLinkedJSONL(b, tmp, 200, 60, 24)
	docs, _, err := readSource(src, 0)
	if err != nil {
		b.Fatalf("read source: %v", err)
	}
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Host != docs[j].Host {
			return docs[i].Host < docs[j].Host
		}
		return docs[i].URL < docs[j].URL
	})
	order := collectionOrder(docs)
	reordered := make([]convert.Document, len(docs))
	for newID, oldID := range order {
		reordered[newID] = docs[oldID]
	}
	docs = reordered
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())
	dir := buildDir(docs)
	const shardSize = 3000

	b.Run("merged", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _, _ = globalSignals(docs, nil, nil)
		}
	})
	b.Run("sharded", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			layouts, regions, err := buildShardGraphs(docs, gids, dir, shardSize)
			if err != nil {
				b.Fatal(err)
			}
			_ = shardedSignals(regions, docs, gids, nil, nil, dir, DefaultPartitionParams())
			runtime.KeepAlive(layouts)
		}
	})
}

// BenchmarkSignalPathsCCrawl measures the memory the reorder saves on the real corpus:
// the merged path builds one in-core graph and ranks over it, materializing the full
// adjacency the in-core rank functions need, while the sharded path builds the per-shard
// regions and streams the cross-shard rank loops over them, holding no expanded
// adjacency resident. Run with -benchtime=1x to measure a single build's allocation.
func BenchmarkSignalPathsCCrawl(b *testing.B) {
	docs := loadCCrawlDocs(b)
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())
	dir := buildDir(docs)
	const shardSize = 3000

	b.Run("merged", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _, _ = globalSignals(docs, nil, nil)
		}
	})
	b.Run("sharded", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			layouts, regions, err := buildShardGraphs(docs, gids, dir, shardSize)
			if err != nil {
				b.Fatal(err)
			}
			_ = shardedSignals(regions, docs, gids, nil, nil, dir, DefaultPartitionParams())
			runtime.KeepAlive(layouts)
		}
	})
}
