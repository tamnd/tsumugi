package collection_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/search"
)

// These tests prove the M13 phase-one gather on real Common Crawl data: the broker sums
// each query term's document frequency across the shards routing selects and divides the
// fleet-wide document count by it to get one collection-wide idf per term. The proof is
// that this gather reconstructs, exactly, the document frequency and idf a single index
// over every document would compute. It is a real-data check of the inputs to the idf,
// not of the final ranked scores: the BM25F length normalization keeps a shard-local
// average field length by design, since unlike the idf it does not factor out of the
// block-max bound, so two shards do not produce bit-identical contributions on documents
// of different lengths. The idf is the cross-shard term the milestone makes exact, and it
// is exactly the term these tests pin.

// buildMultiAndCombined builds the same ccrawl documents two ways: a multi-shard
// collection and a single combined shard over all of them, so the gather across the
// shards can be checked against the combined index's own statistics.
func buildMultiAndCombined(t testing.TB, limit, shardSize int) (multiDir string, combined *search.Shard, total int) {
	t.Helper()
	tmp := t.TempDir()
	multiDir = filepath.Join(tmp, "multi")
	resM, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: multiDir, ShardSize: shardSize, Limit: limit})
	if err != nil {
		t.Fatalf("build multi: %v", err)
	}
	if resM.Shards < 2 {
		t.Fatalf("need at least 2 shards, got %d", resM.Shards)
	}

	combinedDir := filepath.Join(tmp, "combined")
	resC, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: combinedDir, ShardSize: limit + 1, Limit: limit})
	if err != nil {
		t.Fatalf("build combined: %v", err)
	}
	if resC.Shards != 1 {
		t.Fatalf("combined should be one shard, got %d", resC.Shards)
	}
	if resC.Docs != resM.Docs {
		t.Fatalf("combined has %d docs, multi has %d", resC.Docs, resM.Docs)
	}

	paths, err := filepath.Glob(filepath.Join(combinedDir, "*.tsumugi"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("combined shard glob: %v %v", paths, err)
	}
	combined, err = search.OpenShard(paths[0], equivCascade())
	if err != nil {
		t.Fatalf("open combined: %v", err)
	}
	return multiDir, combined, int(resM.Docs)
}

// TestGlobalIDFGatherMatchesCombinedCCrawl checks that gathering a term's df across the
// routed shards equals the combined index's df, and that the resulting collection-wide
// idf equals the combined index's idf, for a spread of real queries. The gather walks
// only the shards routing selects, so a match also proves routing never drops a shard
// that holds the term, which is the precondition for the gathered df to be exact.
func TestGlobalIDFGatherMatchesCombinedCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}

	multiDir, combined, total := buildMultiAndCombined(t, 8000, 1000)
	defer func() { _ = combined.Close() }()

	ix, err := collection.LoadIndex(multiDir)
	if err != nil {
		t.Fatalf("load index: %v", err)
	}
	shards := openInIndexOrder(t, multiDir, ix)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()
	routing := search.NewRoutingIndex(ix.RoutingMap(), ix.AlwaysRouted(), len(shards))
	n := uint64(total)

	queries := []string{
		"the", "time", "data", "page", "information", "new york", "world news",
		"how to", "best price", "contact us", "privacy policy", "search results",
	}
	checked := 0
	for _, q := range queries {
		combinedDF := combined.LexDocFreqs(q)
		if len(combinedDF) == 0 {
			continue
		}
		// Gather df over exactly the shards routing selects, the broker's phase one.
		gathered := map[string]uint32{}
		for _, si := range routing.Route(search.Query{Text: q}) {
			for term, df := range shards[si].LexDocFreqs(q) {
				gathered[term] += df
			}
		}
		for term, want := range combinedDF {
			if gathered[term] != want {
				t.Errorf("query %q term %q: gathered df %d, combined df %d", q, term, gathered[term], want)
			}
			if lexical.IDF(n, uint64(gathered[term])) != lexical.IDF(n, uint64(want)) {
				t.Errorf("query %q term %q: gathered idf differs from combined idf", q, term)
			}
			checked++
		}
		// The gather must not invent a term the combined index does not hold.
		for term := range gathered {
			if _, ok := combinedDF[term]; !ok {
				t.Errorf("query %q: gathered term %q absent from combined index", q, term)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no query terms were checked, the corpus or queries are not exercising the gather")
	}
	t.Logf("verified %d term-idf reconstructions across %d shards", checked, len(shards))
}

// BenchmarkGlobalIDFGather measures the phase-one gather, the extra round the broker runs
// before retrieval to learn the collection-wide idf. It reads bloom filters and
// dictionaries only, no posting lists, so it is meant to be cheap next to the retrieval
// it precedes; this is the number that says how cheap on real data.
func BenchmarkGlobalIDFGather(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	multiDir, combined, _ := buildMultiAndCombined(b, 8000, 1000)
	_ = combined.Close()

	ix, err := collection.LoadIndex(multiDir)
	if err != nil {
		b.Fatalf("load index: %v", err)
	}
	shards := openInIndexOrder(b, multiDir, ix)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()
	routing := search.NewRoutingIndex(ix.RoutingMap(), ix.AlwaysRouted(), len(shards))
	const q = "best price information"
	targets := routing.Route(search.Query{Text: q})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gathered := map[string]uint32{}
		for _, si := range targets {
			for term, df := range shards[si].LexDocFreqs(q) {
				gathered[term] += df
			}
		}
		if len(gathered) == 0 {
			b.Fatal("empty gather")
		}
	}
}
