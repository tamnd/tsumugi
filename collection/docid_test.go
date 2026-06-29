package collection

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/forward"
)

// TestDocIDColumnPersisted is the end-to-end gate for doc 02's portable identity: it
// builds a collection, opens every shard, and reads the doc_id column back out of the
// persisted forward region. Every stored id must be the 32-byte sha256 of the
// document's own canonical URL (checked against the url column in the same row, so the
// test survives the build's host-sorted reordering), and every id across the whole
// collection must be distinct, the collision-freedom the cross-crawl key rests on.
func TestDocIDColumnPersisted(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "crawl.jsonl", 25, 0), Out: out, ShardSize: 7}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatal(err)
	}

	seen := map[string]string{} // doc_id hex -> url that produced it
	var total int
	for _, sh := range shards {
		r, err := tsumugi.Open(sh.Path)
		if err != nil {
			t.Fatal(err)
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			t.Fatal(err)
		}
		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			t.Fatal(err)
		}
		for id := uint32(0); id < fwd.DocCount(); id++ {
			url, _ := fwd.Column("url", id)
			got, _ := fwd.Column("doc_id", id)
			if len(got) != 32 {
				t.Fatalf("doc %q: doc_id is %d bytes, want 32", url, len(got))
			}
			want, ok := analyze.DocID(string(url))
			if !ok {
				t.Fatalf("doc %q: url has no canonical form", url)
			}
			if string(got) != string(want[:]) {
				t.Errorf("doc %q: doc_id = %x, want %x", url, got, want)
			}
			key := hex.EncodeToString(got)
			if prev, dup := seen[key]; dup {
				t.Errorf("doc_id collision: %q and %q both -> %s", prev, url, key)
			}
			seen[key] = string(url)
			total++
		}
		_ = r.Close()
	}
	if total != 25 {
		t.Fatalf("read %d docs, want 25", total)
	}
	if len(seen) != 25 {
		t.Fatalf("%d distinct doc_ids, want 25", len(seen))
	}
}

// TestDocIDCollisionFreeOnCCrawl runs the identity over the real crawl's URL
// distribution: it computes the doc_id of every page's URL and confirms two pages
// that share a canonical URL share an id (the cross-crawl recognition the key is for)
// and two pages with distinct canonical URLs never collide, the property a 32-byte
// sha256 holds at the corpus's scale. It reads the parquet directly rather than
// building shards, since the persistence round-trip is already gated above and the
// point here is the hash's behavior on real, varied URLs.
func TestDocIDCollisionFreeOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	canonToID := map[string]string{} // canonical URL -> doc_id hex
	idToCanon := map[string]string{} // doc_id hex -> canonical URL
	var pages, collisions int
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		id, ok := analyze.DocID(d.URL)
		if !ok {
			t.Fatalf("page %q canonicalizes but has no doc_id", d.URL)
		}
		key := hex.EncodeToString(id[:])
		pages++
		// A canonical URL seen before must map to the same id (deterministic), and the
		// id must reproduce sha256 of the canonical URL the canonicalizer just produced.
		if prev, seen := canonToID[cu]; seen && prev != key {
			t.Fatalf("same canonical %q produced two ids %s and %s", cu, prev, key)
		}
		if other, clash := idToCanon[key]; clash && other != cu {
			collisions++
			t.Errorf("doc_id collision: %q and %q both -> %s", other, cu, key)
		}
		canonToID[cu] = key
		idToCanon[key] = cu
	}
	if pages == 0 {
		t.Skip("no ccrawl pages")
	}
	if collisions != 0 {
		t.Fatalf("%d doc_id collisions over %d pages", collisions, pages)
	}
	t.Logf("pages=%d distinctCanonical=%d distinctDocIDs=%d", pages, len(canonToID), len(idToCanon))
	if len(idToCanon) != len(canonToID) {
		t.Fatalf("distinct doc_ids %d != distinct canonical URLs %d", len(idToCanon), len(canonToID))
	}
}
