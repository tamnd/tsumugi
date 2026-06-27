package collection_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/lexical"
)

// TestBuildRecordsAnalyzerHashCCrawl builds a collection from the real crawl export and
// checks every shard records the build-side analyzer hash, that they all agree, and that
// the manifest carries the same value. This is the chain the broker's startup refusal
// rests on: the build stamps the hash, the manifest summarizes it, and the broker
// compares its query-side analyzer against it. The hash must equal the query-side
// DefaultAnalyzer hash, since the broker analyzes queries with that same analyzer.
func TestBuildRecordsAnalyzerHashCCrawl(t *testing.T) {
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
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards to test cross-shard agreement, got %d", res.Shards)
	}

	want := lexical.DefaultAnalyzer.Hash()
	if want == 0 {
		t.Fatalf("DefaultAnalyzer hash is zero, the recorded hash would be indistinguishable from unset")
	}

	infos, err := collection.List(out)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(infos) != res.Shards {
		t.Fatalf("listed %d shards, build reported %d", len(infos), res.Shards)
	}
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			t.Fatalf("open %s: %v", filepath.Base(info.Path), err)
		}
		h, ok := r.AnalyzerHash()
		_ = r.Close()
		if !ok {
			t.Fatalf("%s recorded no analyzer hash", filepath.Base(info.Path))
		}
		if h != want {
			t.Fatalf("%s analyzer hash = %#016x, want %#016x", filepath.Base(info.Path), h, want)
		}
	}

	ix, err := collection.LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	if ix.AnalyzerHash != want {
		t.Fatalf("manifest analyzer hash = %#016x, want %#016x", ix.AnalyzerHash, want)
	}
	t.Logf("all %d shards and the manifest agree on analyzer hash %#016x", len(infos), want)
}
