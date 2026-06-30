package collection_test

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
)

// collectTree walks a collection directory and returns every regular file's bytes keyed
// by its path relative to the directory, the whole output of a build in one comparable
// map.
func collectTree(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		out[rel] = b
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	return out
}

// TestReproducibleBuildCCrawl builds the real crawl export twice with a pinned epoch and
// asserts the two collections are byte-for-byte identical. The synthetic test proves the
// determinism on a hand-made corpus; this proves it survives the real distribution of
// hosts, languages, and link structure, where the graph reordering, the streamed link
// signals, the per-field dictionaries, and the routing index all do real work over
// millions of edges. If any of those introduced a map-iteration or timing dependency,
// the trees would diverge here even though the synthetic corpus stayed identical. It
// also confirms every shard records a non-zero build_config_hash and that the two builds
// agree on it.
func TestReproducibleBuildCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}

	const epoch uint64 = 1_700_000_000
	build := func(out string) collection.Result {
		res, err := collection.Build(collection.Options{
			Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000, BuildEpoch: epoch,
		})
		if err != nil {
			t.Fatalf("build into %s: %v", out, err)
		}
		return res
	}

	tmp := t.TempDir()
	out1 := filepath.Join(tmp, "c1")
	out2 := filepath.Join(tmp, "c2")
	res := build(out1)
	build(out2)
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards to exercise cross-shard work, got %d", res.Shards)
	}

	a := collectTree(t, out1)
	b := collectTree(t, out2)
	if len(a) != len(b) {
		t.Fatalf("file counts differ: %d vs %d", len(a), len(b))
	}
	for name, ab := range a {
		bb, ok := b[name]
		if !ok {
			t.Errorf("%s present in first build, absent in second", name)
			continue
		}
		if !bytes.Equal(ab, bb) {
			t.Errorf("%s differs between builds: %d vs %d bytes", name, len(ab), len(bb))
		}
	}
	t.Logf("two real-data builds wrote %d byte-identical files across %d shards", len(a), res.Shards)

	// Every shard must carry a non-zero configuration digest, and the two builds must
	// agree on it shard for shard.
	infos1, err := collection.List(out1)
	if err != nil {
		t.Fatalf("list c1: %v", err)
	}
	infos2, err := collection.List(out2)
	if err != nil {
		t.Fatalf("list c2: %v", err)
	}
	for i := range infos1 {
		h1 := configHashOf(t, infos1[i].Path)
		h2 := configHashOf(t, infos2[i].Path)
		if h1 == 0 {
			t.Fatalf("shard %d recorded a zero config hash", i)
		}
		if h1 != h2 {
			t.Fatalf("shard %d config hash differs: %#016x vs %#016x", i, h1, h2)
		}
	}
}

func configHashOf(t *testing.T, path string) uint64 {
	t.Helper()
	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer func() { _ = r.Close() }()
	h, ok := r.BuildConfigHash()
	if !ok {
		t.Fatalf("%s recorded no build_config_hash", path)
	}
	return h
}
