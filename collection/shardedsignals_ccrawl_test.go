package collection

import (
	"os"
	"path/filepath"
	"testing"
)

// TestShardedSignalsMatchMergedCCrawl runs the assembly equivalence on the real
// CC-MAIN-2026-25 sample: it builds a real multi-shard collection from the parquet,
// loads the per-shard graph regions, assembles the signals off them, and asserts they
// equal the merged in-core pass field for field over the same documents.
//
// This is the proof over real data. The real bodies drive the content signals, the real
// URLs drive the host and domain grouping and the partition global ids, and the graph is
// the sparse one a broad crawl yields. A broad crawl resolves almost no in-corpus links,
// so the dense cross-shard rank and count paths are exercised by the synthetic gate and
// by each form's own slice; here the meaningful spread is in the content and grouping
// signals, which is what this gate exercises (it asserts a real language spread and a
// nonzero static rank so the equivalence is not over all-zero vectors).
//
// It runs without the race detector: the real build times out under -race, the same skip
// the other ccrawl build tests take.
func TestShardedSignalsMatchMergedCCrawl(t *testing.T) {
	if testing.Short() {
		t.Skip("ccrawl build is slow; skipped under -short")
	}
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}

	tmp := t.TempDir()
	out := filepath.Join(tmp, "col")
	const shardSize = 3000
	res, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: shardSize})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("want a multi-shard collection, got %d shards", res.Shards)
	}

	docs, gids, _ := buildLayout(t, ccrawlGraphParquet)
	dir := buildDir(docs)
	regions, closeAll := loadShardGraphs(t, out)
	defer closeAll()

	got := shardedSignals(regions, docs, gids, nil, nil, dir, DefaultPartitionParams())
	want, _, _ := globalSignals(docs, nil, nil)

	distinctLang := map[uint32]struct{}{}
	var nonzeroStatic int
	for i := range want.staticRank {
		distinctLang[want.langID[i]] = struct{}{}
		if want.staticRank[i] > 0 {
			nonzeroStatic++
		}
	}
	if len(distinctLang) < 2 {
		t.Fatalf("want a real language spread, got %d distinct languages", len(distinctLang))
	}
	if nonzeroStatic == 0 {
		t.Fatal("static rank is all zero; gate is vacuous")
	}

	signalsClose(t, got, want)
	t.Logf("docs=%d shards=%d distinctLangs=%d nonzeroStatic=%d", len(docs), len(regions), len(distinctLang), nonzeroStatic)
}
