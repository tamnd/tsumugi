package collection

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
)

// readTree walks a collection directory and returns every regular file's bytes keyed
// by its path relative to the directory, the whole on-disk output of a build in one
// comparable map. A reproducibility check compares two such maps: identical key sets
// and identical bytes mean the build wrote a byte-for-byte identical collection.
func readTree(t *testing.T, dir string) map[string][]byte {
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

// TestBuildByteIdenticalShards builds the same synthetic crawl twice with a pinned
// build epoch and asserts the two collections are byte-for-byte identical: the same set
// of files, each with the same contents. This is the reproducibility contract doc 12
// rests on, that a build is a pure function of its source, its seeds, and its epoch, so
// a rebuilt collection can be checksum-verified against the one it should reproduce.
// Pinning the epoch removes the one clock the build reads; everything else, the node
// ordering, the signal sums, the dictionaries, the routing index, is already
// deterministic, so the only way the trees differ is a regression in that determinism.
func TestBuildByteIdenticalShards(t *testing.T) {
	tmp := t.TempDir()
	src := writeJSONL(t, tmp, "crawl.jsonl", 60, 0)

	const epoch uint64 = 1_700_000_000
	out1 := filepath.Join(tmp, "c1")
	out2 := filepath.Join(tmp, "c2")
	opts := func(out string) Options {
		return Options{Source: src, Out: out, ShardSize: 20, BuildEpoch: epoch,
			TrustSeeds: []string{"https://host00.example/p0"}}
	}
	if _, err := Build(opts(out1)); err != nil {
		t.Fatalf("first build: %v", err)
	}
	if _, err := Build(opts(out2)); err != nil {
		t.Fatalf("second build: %v", err)
	}

	a := readTree(t, out1)
	b := readTree(t, out2)
	if len(a) != len(b) {
		t.Fatalf("file counts differ: %d vs %d", len(a), len(b))
	}
	if len(a) < 3 {
		t.Fatalf("expected at least shards + index + graph, got %d files", len(a))
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
	t.Logf("two builds wrote %d byte-identical files", len(a))
}

// TestBuildEpochStampedAndDeterministic checks the epoch is the only clock input: a
// shard records the epoch it was built with, two builds at different epochs differ only
// because of it, and the configuration digest is unchanged by the epoch since the epoch
// names when a build ran, not how it was configured.
func TestBuildEpochStampedAndDeterministic(t *testing.T) {
	tmp := t.TempDir()
	src := writeJSONL(t, tmp, "crawl.jsonl", 30, 0)

	build := func(out string, epoch uint64) {
		if _, err := Build(Options{Source: src, Out: out, ShardSize: 30, BuildEpoch: epoch}); err != nil {
			t.Fatalf("build epoch %d: %v", epoch, err)
		}
	}
	outA := filepath.Join(tmp, "a")
	outB := filepath.Join(tmp, "b")
	build(outA, 111)
	build(outB, 222)

	shardA := shardPath(outA, 0)
	shardB := shardPath(outB, 0)
	ra, err := tsumugi.Open(shardA)
	if err != nil {
		t.Fatalf("open A: %v", err)
	}
	defer func() { _ = ra.Close() }()
	rb, err := tsumugi.Open(shardB)
	if err != nil {
		t.Fatalf("open B: %v", err)
	}
	defer func() { _ = rb.Close() }()

	if ra.Header.BuildEpoch != 111 || rb.Header.BuildEpoch != 222 {
		t.Fatalf("epochs not stamped: A=%d B=%d", ra.Header.BuildEpoch, rb.Header.BuildEpoch)
	}
	// The epoch changes the header, so the shard bytes differ between the two builds.
	if bytes.Equal(readFile(t, shardA), readFile(t, shardB)) {
		t.Fatal("shards at different epochs should differ in their header bytes")
	}
	// The configuration digest names how the build was configured, not when it ran, so
	// the two shards must agree on it despite the different epochs.
	ha, okA := ra.BuildConfigHash()
	hb, okB := rb.BuildConfigHash()
	if !okA || !okB {
		t.Fatalf("config hash missing: A=%v B=%v", okA, okB)
	}
	if ha != hb {
		t.Fatalf("config hash changed with epoch: %#016x vs %#016x", ha, hb)
	}
}

// TestBuildConfigHashStable pins the configuration digest contract: a shard records the
// digest of the configuration it was built under, the digest matches the one the build
// computes, and it changes when any configuration input changes while staying the same
// when only the corpus or the epoch changes. This is what lets a reader tell two shards
// are configuration-compatible from one stat instead of reopening every region.
func TestBuildConfigHashStable(t *testing.T) {
	tmp := t.TempDir()
	src := writeJSONL(t, tmp, "crawl.jsonl", 20, 0)

	out := filepath.Join(tmp, "c")
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 20, BuildEpoch: 7}); err != nil {
		t.Fatalf("build: %v", err)
	}
	r, err := tsumugi.Open(shardPath(out, 0))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	got, ok := r.BuildConfigHash()
	_ = r.Close()
	if !ok {
		t.Fatal("shard recorded no build_config_hash")
	}
	want := buildConfigHash(20, nil, nil, false, 0)
	if got != want {
		t.Fatalf("recorded config hash %#016x, computed %#016x", got, want)
	}

	base := buildConfigHash(20, nil, nil, false, 0)
	cases := []struct {
		name string
		hash uint64
		same bool
	}{
		{"same inputs", buildConfigHash(20, nil, nil, false, 0), true},
		{"different shard size", buildConfigHash(50, nil, nil, false, 0), false},
		{"added trust seed", buildConfigHash(20, []string{"https://a/"}, nil, false, 0), false},
		{"added spam seed", buildConfigHash(20, nil, []string{"https://b/"}, false, 0), false},
		{"impact ordering", buildConfigHash(20, nil, nil, true, 0), false},
		{"dense dimension", buildConfigHash(20, nil, nil, false, 256), false},
		{"seed order irrelevant",
			buildConfigHash(20, []string{"https://b/", "https://a/"}, nil, false, 0),
			buildConfigHash(20, []string{"https://a/", "https://b/"}, nil, false, 0) == base},
	}
	for _, c := range cases {
		if c.name == "seed order irrelevant" {
			// This case asserts two seed orderings agree with each other, not with base.
			h1 := buildConfigHash(20, []string{"https://b/", "https://a/"}, nil, false, 0)
			h2 := buildConfigHash(20, []string{"https://a/", "https://b/"}, nil, false, 0)
			if h1 != h2 {
				t.Errorf("seed order changed the digest: %#016x vs %#016x", h1, h2)
			}
			continue
		}
		if (c.hash == base) != c.same {
			t.Errorf("%s: hash==base is %v, want %v (base=%#016x got=%#016x)",
				c.name, c.hash == base, c.same, base, c.hash)
		}
	}
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}
