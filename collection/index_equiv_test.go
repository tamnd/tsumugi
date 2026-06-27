package collection_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// ccrawlParquet is the real crawl export the scale-oriented tests build from when it is
// present. It is a Common Crawl markdown shard, the language and content distribution
// the engine is meant to serve. The tests that need it skip when it is absent so the
// suite still runs on a machine without the data.
const ccrawlParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// equivCascade is the cascade both brokers share in the equivalence tests. The model is
// query-independent, so a document scores identically wherever it is reranked; the test
// only needs the two brokers to run the same cascade, since the claim under test is that
// the routing artifact does not change which candidates reach the rerank.
func equivCascade() *rank.Cascade {
	model := rank.Compile(nil, len(feature.DefaultSchema()))
	return rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
}

// openInIndexOrder opens a collection's shards in the artifact's manifest order, the
// order the index-backed broker's routing ids are keyed to.
func openInIndexOrder(t testing.TB, dir string, ix *collection.Index) []*search.Shard {
	t.Helper()
	shards := make([]*search.Shard, 0, ix.NumShards())
	for _, info := range ix.Shards {
		p := filepath.Join(dir, filepath.Base(info.Path))
		s, err := search.OpenShard(p, equivCascade())
		if err != nil {
			t.Fatalf("open shard %s: %v", p, err)
		}
		shards = append(shards, s)
	}
	return shards
}

// brokers builds an index-backed broker and a scan broker over the same shards opened in
// the same order, so a difference between them is the artifact path's alone.
func brokers(t testing.TB, dir string) (indexBacked, scan *search.Broker, close func()) {
	t.Helper()
	ix, err := collection.LoadIndex(dir)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	shA := openInIndexOrder(t, dir, ix)
	shB := openInIndexOrder(t, dir, ix)

	routing := search.NewRoutingIndex(ix.RoutingMap(), ix.AlwaysRouted(), len(shA))
	stats := search.GlobalStats{
		DocCount:   ix.Stats.DocCount,
		TokenCount: ix.Stats.TokenCount,
		AvgDocLen:  ix.Stats.AvgDocLen,
	}
	indexBacked = search.NewBrokerWith(shA, equivCascade(), routing, stats)
	scan = search.NewBroker(shB, equivCascade())
	return indexBacked, scan, func() {
		_ = indexBacked.Close()
		_ = scan.Close()
	}
}

// sameHits reports whether two ranked result lists are identical, doc id and score, in
// order. The broker's merge is deterministic, so an exact match is the right assertion,
// not a recall threshold.
func sameHits(a, b []search.Hit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].DocID != b[i].DocID || a[i].Score != b[i].Score {
			return false
		}
	}
	return true
}

// writeSyntheticJSONL writes a small crawl with a controllable vocabulary so the
// equivalence test exercises both common and rare terms without needing the real data.
func writeSyntheticJSONL(t *testing.T, path string, n int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		host := fmt.Sprintf("host%02d.example", i%8)
		line := fmt.Sprintf(
			`{"url":"https://%s/p%d","host":"%s","markdown":"# Title %d\nshared corpus text token%d body of document %d"}`+"\n",
			host, i, host, i, i, i)
		if _, err := f.WriteString(line); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestIndexBrokerEqualsScanSynthetic proves the index-backed broker returns bit-identical
// top-k to the scan broker over a synthetic collection, for common terms, rare terms,
// multi-term queries, and absent terms. This is the exactness claim of the artifact: the
// persisted routing must select exactly the shards a full scan would, so the merged
// top-k is unchanged.
func TestIndexBrokerEqualsScanSynthetic(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "crawl.jsonl")
	writeSyntheticJSONL(t, src, 400)
	out := filepath.Join(tmp, "coll")
	if _, err := collection.Build(collection.Options{Source: src, Out: out, ShardSize: 50}); err != nil {
		t.Fatalf("Build: %v", err)
	}

	idx, scan, done := brokers(t, out)
	defer done()

	queries := []string{
		"shared", "corpus text", "token7", "token123 document", "body shared corpus",
		"absenttermxyz", "token7 absenttermxyz",
	}
	ctx := context.Background()
	for _, qt := range queries {
		q := search.Query{Text: qt, K: 10}
		if !sameHits(idx.Search(ctx, q), scan.Search(ctx, q)) {
			t.Errorf("index and scan brokers disagree for query %q", qt)
		}
	}
}

// TestIndexBrokerEqualsScanCCrawl is the same exactness claim on real Common Crawl
// markdown, the language and content distribution the engine targets. It builds a
// modest collection from the parquet and asserts the artifact path matches the scan
// path for a spread of natural-language queries.
func TestIndexBrokerEqualsScanCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	t.Logf("built %d docs across %d shards from %d hosts", res.Docs, res.Shards, res.Hosts)
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards to test cross-shard routing, got %d", res.Shards)
	}

	idx, scan, done := brokers(t, out)
	defer done()

	queries := []string{
		"the", "time", "data", "page", "information", "new york", "world news",
		"how to", "best price", "contact us", "privacy policy", "search results",
	}
	ctx := context.Background()
	matched := 0
	for _, qt := range queries {
		q := search.Query{Text: qt, K: 20}
		a := idx.Search(ctx, q)
		b := scan.Search(ctx, q)
		if !sameHits(a, b) {
			t.Errorf("index and scan brokers disagree for query %q (idx %d hits, scan %d hits)", qt, len(a), len(b))
		}
		if len(a) > 0 {
			matched++
		}
	}
	if matched == 0 {
		t.Errorf("no query returned hits, the corpus or queries are not exercising retrieval")
	}
	t.Logf("%d/%d queries returned hits, all identical across index and scan brokers", matched, len(queries))
}

// BenchmarkLoadIndex measures the artifact load, the serve startup cost the manifest
// replaces the full-shard scan with. Pair it with BenchmarkBuildRoutingScan to see the
// scale win: load is reading one file, the scan reopens and walks every shard.
func BenchmarkLoadIndex(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := b.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000}); err != nil {
		b.Fatalf("Build: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ix, err := collection.LoadIndex(out)
		if err != nil {
			b.Fatal(err)
		}
		if ix.NumShards() == 0 {
			b.Fatal("empty index")
		}
	}
}

// BenchmarkBuildRoutingScan measures the legacy startup path: open every shard and walk
// its vocabulary to rebuild the routing index. This is the cost the artifact removes,
// and it grows with the corpus, which is why it does not hold at fleet scale.
func BenchmarkBuildRoutingScan(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := b.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000}); err != nil {
		b.Fatalf("Build: %v", err)
	}
	ix, err := collection.LoadIndex(out)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		shards := openInIndexOrder(b, out, ix)
		_ = search.BuildRoutingIndex(shards)
		for _, s := range shards {
			_ = s.Close()
		}
	}
}
