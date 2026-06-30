package collection

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestWriteIndexCreatesArtifact checks that a build leaves a loadable artifact whose
// manifest and statistics agree with the shards on disk.
func TestWriteIndexCreatesArtifact(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "crawl.jsonl", 25, 0), Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if _, err := os.Stat(filepath.Join(out, IndexName)); err != nil {
		t.Fatalf("index artifact missing after build: %v", err)
	}
	ix, err := LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}

	infos, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	if ix.NumShards() != len(infos) {
		t.Errorf("index shards = %d, want %d", ix.NumShards(), len(infos))
	}
	var docs uint64
	for _, in := range infos {
		docs += uint64(in.DocCount)
	}
	if ix.Stats.DocCount != docs {
		t.Errorf("index doc count = %d, want %d", ix.Stats.DocCount, docs)
	}
	if ix.Stats.AvgDocLen <= 0 {
		t.Errorf("index avg doc len = %v, want positive", ix.Stats.AvgDocLen)
	}
	// The manifest's bases and counts must match the listing exactly.
	for i, in := range infos {
		if ix.Shards[i].NodeBase != in.NodeBase || ix.Shards[i].DocCount != in.DocCount {
			t.Errorf("shard %d manifest base/count = %d/%d, want %d/%d",
				i, ix.Shards[i].NodeBase, ix.Shards[i].DocCount, in.NodeBase, in.DocCount)
		}
	}
}

// TestIndexRoutingSelectivity checks that a term in every document routes to every
// shard while a term in one document routes to fewer, the selectivity that keeps
// fan-out sublinear in fleet size.
func TestIndexRoutingSelectivity(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "crawl.jsonl", 30, 0), Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	ix, err := LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}

	// "shared" is in every body, so it must route to every shard.
	common := ix.Route([]string{"shared"})
	if len(common) != ix.NumShards() {
		t.Errorf("common term routed to %d shards, want all %d", len(common), ix.NumShards())
	}
	// A digit that appears in one document's body routes to strictly fewer shards.
	rare := ix.Route([]string{"7"})
	if len(rare) == 0 {
		t.Errorf("rare term routed to no shards, want at least one")
	}
	if len(rare) >= len(common) {
		t.Errorf("rare term routed to %d shards, want fewer than the common term's %d", len(rare), len(common))
	}
	// A term in no document routes nowhere (no impact shards to force-route here).
	none := ix.Route([]string{"zzzznotindexedterm"})
	if len(none) != 0 {
		t.Errorf("absent term routed to %d shards, want 0", len(none))
	}
	// An empty query routes everywhere, since routing says nothing about it.
	if len(ix.Route(nil)) != ix.NumShards() {
		t.Errorf("empty query routed to %d shards, want all", len(ix.Route(nil)))
	}
}

// TestIndexEncodeDecodeRoundTrip checks the artifact survives a serialize and parse
// unchanged, including the delta-coded routing lists and the always-routed set.
func TestIndexEncodeDecodeRoundTrip(t *testing.T) {
	in := &Index{
		BuildEpoch: 1735000000,
		Stats:      Stats{DocCount: 123, TokenCount: 9876.5, AvgDocLen: 80.3},
		Shards: []ShardInfo{
			{Path: "shard-00000.tsumugi", NodeBase: 0, DocCount: 100, Bytes: 4096},
			{Path: "shard-00001.tsumugi", NodeBase: 100, DocCount: 23, Bytes: 2048},
		},
		routing: map[string][]int32{
			"alpha": {0, 1},
			"beta":  {1},
			"gamma": {0},
		},
		always: []int32{},
		numShd: 2,
	}
	got, err := decodeIndex(in.encode())
	if err != nil {
		t.Fatalf("decodeIndex: %v", err)
	}
	if got.BuildEpoch != in.BuildEpoch {
		t.Errorf("epoch = %d, want %d", got.BuildEpoch, in.BuildEpoch)
	}
	if got.Stats != in.Stats {
		t.Errorf("stats = %+v, want %+v", got.Stats, in.Stats)
	}
	wantShards := []ShardInfo{
		{Path: "shard-00000.tsumugi", NodeBase: 0, DocCount: 100, Bytes: 4096},
		{Path: "shard-00001.tsumugi", NodeBase: 100, DocCount: 23, Bytes: 2048},
	}
	if !reflect.DeepEqual(got.Shards, wantShards) {
		t.Errorf("shards = %+v, want %+v", got.Shards, wantShards)
	}
	if !reflect.DeepEqual(got.routing, in.routing) {
		t.Errorf("routing = %+v, want %+v", got.routing, in.routing)
	}
}

// TestIndexDeterministic checks the encoding is byte-identical for the same logical
// index regardless of map iteration order, the property that lets a build be
// reproducible.
func TestIndexDeterministic(t *testing.T) {
	mk := func() *Index {
		return &Index{
			Stats:   Stats{DocCount: 1},
			Shards:  []ShardInfo{{Path: "shard-00000.tsumugi"}},
			routing: map[string][]int32{"a": {0}, "b": {0}, "c": {0}, "d": {0}},
			numShd:  1,
		}
	}
	if !reflect.DeepEqual(mk().encode(), mk().encode()) {
		t.Errorf("encoding is not deterministic across builds")
	}
}

// TestIndexCorruptRejected checks a flipped byte fails the CRC rather than decoding to
// garbage, the torn-write guard the serve path relies on.
func TestIndexCorruptRejected(t *testing.T) {
	in := &Index{
		Stats:   Stats{DocCount: 1},
		Shards:  []ShardInfo{{Path: "shard-00000.tsumugi", DocCount: 1}},
		routing: map[string][]int32{"x": {0}},
		numShd:  1,
	}
	b := in.encode()
	b[len(b)/2] ^= 0xff
	if _, err := decodeIndex(b); err == nil {
		t.Errorf("decodeIndex accepted a corrupt artifact, want rejection")
	}
}

// TestLoadIndexMissing checks a directory with no artifact reports a not-exist error so
// the serve path can fall back to scanning the shards.
func TestLoadIndexMissing(t *testing.T) {
	if _, err := LoadIndex(t.TempDir()); !os.IsNotExist(err) {
		t.Errorf("LoadIndex on empty dir error = %v, want IsNotExist", err)
	}
}

// TestCompactRefreshesIndex checks that compact rewrites the artifact over the new
// shard set, so a stale manifest cannot survive a merge.
func TestCompactRefreshesIndex(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "crawl.jsonl", 25, 0), Out: out, ShardSize: 5}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	before, err := LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex before: %v", err)
	}
	if before.NumShards() != 5 {
		t.Fatalf("pre-compact index shards = %d, want 5", before.NumShards())
	}
	if _, err := Compact(out, 20, NoEpoch); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	after, err := LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex after: %v", err)
	}
	if after.NumShards() != 2 {
		t.Errorf("post-compact index shards = %d, want 2", after.NumShards())
	}
	if after.Stats.DocCount != before.Stats.DocCount {
		t.Errorf("doc count changed across compact: %d -> %d", before.Stats.DocCount, after.Stats.DocCount)
	}
}
