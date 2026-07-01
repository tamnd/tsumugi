package collection_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/vector"
)

// TestCompactPreservesDenseRegion checks that a compact of a dense-enabled collection
// re-embeds at the same dimension rather than silently dropping the dense plane. A compact
// rebuilds shards from their forward stores, so it has to detect the source collection's
// vector dimension and rebuild with it; otherwise a compacted collection would serve
// lexical-only where the original served dense. It builds a small dense collection, compacts
// it to a single shard, and asserts the compacted shard still carries a vector region at the
// original dimension with the vector_dim stat recorded.
func TestCompactPreservesDenseRegion(t *testing.T) {
	tmp := t.TempDir()
	lines := []string{
		`{"url":"https://a.example/one","host":"a.example","markdown":"# One\nalpha bravo charlie delta echo foxtrot golf content here for the first page"}`,
		`{"url":"https://a.example/two","host":"a.example","markdown":"# Two\nhotel india juliet kilo lima mike november content for the second page here"}`,
		`{"url":"https://b.example/three","host":"b.example","markdown":"# Three\noscar papa quebec romeo sierra tango uniform content for the third page here"}`,
		`{"url":"https://b.example/four","host":"b.example","markdown":"# Four\nvictor whiskey xray yankee zulu content spread over the fourth page body here"}`,
	}
	src := filepath.Join(tmp, "src.jsonl")
	if err := os.WriteFile(src, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	const dim = 64
	out := filepath.Join(tmp, "coll")
	// Two shards so the compact has something to merge.
	if _, err := collection.Build(collection.Options{Source: src, Out: out, ShardSize: 2, DenseDim: dim}); err != nil {
		t.Fatalf("dense build: %v", err)
	}
	before, err := collection.List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) < 2 {
		t.Fatalf("want at least 2 shards before compact, got %d", len(before))
	}

	if _, err := collection.Compact(out, 1000, 0); err != nil {
		t.Fatalf("compact: %v", err)
	}
	after, err := collection.List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 1 {
		t.Fatalf("want 1 shard after compact into a large shard size, got %d", len(after))
	}

	r, err := tsumugi.Open(after[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	if !r.HasRegion(tsumugi.RegionVector) {
		t.Fatal("compact dropped the vector region")
	}
	if v, ok := r.Stat(tsumugi.StatVectorDim); !ok || int(v) != dim {
		t.Fatalf("compacted vector_dim stat = %v (ok=%v), want %d", v, ok, dim)
	}
	vb, err := r.Region(tsumugi.RegionVector)
	if err != nil {
		t.Fatal(err)
	}
	reg, err := vector.Open(vb)
	if err != nil {
		t.Fatalf("open compacted vector region: %v", err)
	}
	if reg.Dim() != dim {
		t.Fatalf("compacted region Dim() = %d, want %d", reg.Dim(), dim)
	}
}
