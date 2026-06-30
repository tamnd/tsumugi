package collection

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/forward"
)

// writeJSONL writes n synthetic crawl records across a few hosts to a .jsonl file and
// returns its path. The bodies carry a shared word so every shard indexes real text.
func writeJSONL(t *testing.T, dir, name string, n, hostOffset int) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		host := fmt.Sprintf("host%02d.example", (hostOffset+i)%4)
		line := fmt.Sprintf(
			`{"url":"https://%s/p%d","host":"%s","markdown":"# Page %d\nshared body text for document %d"}`+"\n",
			host, i, host, i, i)
		if _, err := f.WriteString(line); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestBuildListShards(t *testing.T) {
	tmp := t.TempDir()
	src := writeJSONL(t, tmp, "crawl.jsonl", 25, 0)
	out := filepath.Join(tmp, "coll")

	res, err := Build(Options{Source: src, Out: out, ShardSize: 10})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if res.Docs != 25 || res.Shards != 3 {
		t.Fatalf("build result docs/shards = %d/%d, want 25/3", res.Docs, res.Shards)
	}

	infos, err := List(out)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(infos) != 3 {
		t.Fatalf("listed %d shards, want 3", len(infos))
	}
	// Node bases must be dense and contiguous in global id order.
	wantBase := []uint32{0, 10, 20}
	wantCount := []uint32{10, 10, 5}
	for i, in := range infos {
		if in.NodeBase != wantBase[i] || in.DocCount != wantCount[i] {
			t.Errorf("shard %d base/count = %d/%d, want %d/%d",
				i, in.NodeBase, in.DocCount, wantBase[i], wantCount[i])
		}
	}
}

func TestAddExtendsCollection(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "a.jsonl", 12, 0), Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	res, err := Add(Options{Source: writeJSONL(t, tmp, "b.jsonl", 8, 2), Out: out, ShardSize: 10})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if res.Docs != 8 {
		t.Fatalf("add docs = %d, want 8", res.Docs)
	}

	infos, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	// Build laid down two shards (10 + 2); Add continues the id space past id 12.
	var total uint32
	for _, in := range infos {
		total += in.DocCount
	}
	if total != 20 {
		t.Fatalf("total docs after add = %d, want 20", total)
	}
	last := infos[len(infos)-1]
	if last.NodeBase != 12 {
		t.Errorf("added shard base = %d, want it to continue at 12", last.NodeBase)
	}
}

func TestCompactMergesAndPreservesDocs(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "crawl.jsonl", 25, 0), Out: out, ShardSize: 5}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	before, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != 5 {
		t.Fatalf("pre-compact shards = %d, want 5", len(before))
	}

	res, err := Compact(out, 20, NoEpoch)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if res.Docs != 25 || res.Shards != 2 {
		t.Fatalf("compact result docs/shards = %d/%d, want 25/2", res.Docs, res.Shards)
	}

	after, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 2 {
		t.Fatalf("post-compact shards = %d, want 2", len(after))
	}
	// The staging directory must be gone and the id space dense again.
	if _, err := os.Stat(filepath.Join(out, ".compact")); !os.IsNotExist(err) {
		t.Errorf("staging dir should be removed after compact")
	}
	var total uint32
	for _, in := range after {
		total += in.DocCount
	}
	if total != 25 {
		t.Fatalf("total docs after compact = %d, want 25", total)
	}

	// Every document's forward store must still carry the url and body it was built
	// from, the invariant a later compact rebuilds from.
	r, err := tsumugi.Open(after[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	b, err := r.Region(tsumugi.RegionForward)
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := forward.Open(b)
	if err != nil {
		t.Fatal(err)
	}
	u, _ := fwd.Column("url", 0)
	body, _ := fwd.Column("body", 0)
	if len(u) == 0 || len(body) == 0 {
		t.Errorf("compacted shard doc 0 url/body = %q/%q, want both preserved", u, body)
	}
}
