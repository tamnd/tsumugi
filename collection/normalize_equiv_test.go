package collection_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/search"
)

// closeEnough reports whether two fleet averages agree to floating-point tolerance.
// The averages are sums of integer token counts divided by the same document count
// regardless of partitioning, so the only difference a partition can introduce is the
// order floating-point addition runs in, which this tolerance absorbs.
func closeEnough(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

// statsOf builds a scan broker over a freshly built collection and returns its
// fleet-wide statistics, the values the online BM25F normalizes each field by.
func statsOf(t testing.TB, src, out string, shardSize, limit int) search.GlobalStats {
	t.Helper()
	if _, err := collection.Build(collection.Options{Source: src, Out: out, ShardSize: shardSize, Limit: limit}); err != nil {
		t.Fatalf("Build (shardSize=%d): %v", shardSize, err)
	}
	shards := openCollectionShards(t, out)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()
	return search.NewBroker(shards, equivCascade()).Stats()
}

// openCollectionShards opens every shard a built collection's manifest names, the set
// a scan broker computes its fleet statistics over.
func openCollectionShards(t testing.TB, dir string) []*search.Shard {
	t.Helper()
	ix, err := collection.LoadIndex(dir)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	return openInIndexOrder(t, dir, ix)
}

// sameFieldStats fails the test when two fleet statistics differ on any field the
// online BM25F reads, the assertion that a document's field-weighted score does not
// depend on how the collection was partitioned into shards.
func sameFieldStats(t testing.TB, a, b search.GlobalStats, ctx string) {
	t.Helper()
	if a.DocCount != b.DocCount {
		t.Errorf("%s: doc count differs: %d vs %d", ctx, a.DocCount, b.DocCount)
	}
	if !closeEnough(a.AvgDocLen, b.AvgDocLen) {
		t.Errorf("%s: avg doc len differs: %.6f vs %.6f", ctx, a.AvgDocLen, b.AvgDocLen)
	}
	for f := 0; f < 3; f++ {
		if !closeEnough(a.AvgFieldLen[f], b.AvgFieldLen[f]) {
			t.Errorf("%s: avg field len[%d] differs: %.6f vs %.6f", ctx, f, a.AvgFieldLen[f], b.AvgFieldLen[f])
		}
	}
}

// TestPerFieldStatsPartitionIndependentSynthetic proves the fleet per-field average
// lengths are identical whether the same documents land in one shard or many. This is
// the property that makes per-field BM25F shard-independent: the online extractor
// normalizes each field by these fleet averages, so if they do not move with the
// partition, neither does a document's field-weighted score, which is gap #2's claim
// for the field-length normalizer the way M13's fleet df is for idf.
func TestPerFieldStatsPartitionIndependentSynthetic(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "crawl.jsonl")
	writeSyntheticJSONL(t, src, 400)

	whole := statsOf(t, src, filepath.Join(tmp, "one"), 10000, 0)
	split := statsOf(t, src, filepath.Join(tmp, "many"), 25, 0)

	sameFieldStats(t, whole, split, "synthetic one-shard vs many-shard")
	for f := 0; f < 3; f++ {
		if whole.AvgFieldLen[f] <= 0 {
			t.Errorf("field %d should carry a positive fleet average, got %.6f", f, whole.AvgFieldLen[f])
		}
	}
}

// TestPerFieldStatsManifestRoundTrip proves the manifest persists the fleet per-field
// averages a scan would compute, so a broker started from the artifact normalizes
// each field exactly as one that rescanned the shards would. Without this the
// index-backed serve path would carry zero field averages and silently drop per-field
// length normalization, the regression the round-trip guards.
func TestPerFieldStatsManifestRoundTrip(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "crawl.jsonl")
	writeSyntheticJSONL(t, src, 400)
	out := filepath.Join(tmp, "coll")

	scan := statsOf(t, src, out, 25, 0)

	ix, err := collection.LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	if !closeEnough(ix.Stats.AvgDocLen, scan.AvgDocLen) {
		t.Errorf("manifest avg doc len %.6f, scan %.6f", ix.Stats.AvgDocLen, scan.AvgDocLen)
	}
	for f := 0; f < 3; f++ {
		if !closeEnough(ix.Stats.AvgFieldLen[f], scan.AvgFieldLen[f]) {
			t.Errorf("manifest avg field len[%d] %.6f, scan %.6f", f, ix.Stats.AvgFieldLen[f], scan.AvgFieldLen[f])
		}
		if ix.Stats.AvgFieldLen[f] <= 0 {
			t.Errorf("manifest field %d average should be positive, got %.6f", f, ix.Stats.AvgFieldLen[f])
		}
	}

	// The index-backed and scan brokers must agree on the field averages too, since the
	// serve path reads them from the manifest the round-trip just checked.
	idx, scanB, done := brokers(t, out)
	defer done()
	sameFieldStats(t, idx.Stats(), scanB.Stats(), "index-backed vs scan broker")
}

// TestPerFieldStatsPartitionIndependentCCrawl is the same partition-independence proof
// on real Common Crawl markdown, the language and content distribution the engine
// targets. Real titles, urls, and bodies have field lengths that vary widely, so the
// averages are non-trivial, and the test confirms they still do not move with the
// shard partitioning.
func TestPerFieldStatsPartitionIndependentCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	whole := statsOf(t, ccrawlParquet, filepath.Join(tmp, "one"), 100000, 8000)
	split := statsOf(t, ccrawlParquet, filepath.Join(tmp, "many"), 1000, 8000)

	sameFieldStats(t, whole, split, "ccrawl one-shard vs many-shard")
	for f := 0; f < 3; f++ {
		if whole.AvgFieldLen[f] <= 0 {
			t.Errorf("ccrawl field %d should carry a positive fleet average, got %.6f", f, whole.AvgFieldLen[f])
		}
	}
	t.Logf("ccrawl fleet field averages: title %.2f body %.2f url %.2f",
		whole.AvgFieldLen[0], whole.AvgFieldLen[1], whole.AvgFieldLen[2])
}
