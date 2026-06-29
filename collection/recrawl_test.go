package collection

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/forward"
)

// writeCrawl writes the given URLs as a crawl export, one record a URL with a body so
// every page is indexed. The host is parsed from the URL so a record's host matches the
// page it names, and crawlDate stamps the fetch so a later test can reason about which
// copy wins.
func writeCrawl(t *testing.T, dir, name string, urls []string, crawlDate string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	var b strings.Builder
	for i, u := range urls {
		host := analyze.HostOf(u)
		fmt.Fprintf(&b,
			`{"url":%q,"host":%q,"markdown":"# Page %d\nshared body text for document %d","crawl_date":%q}`+"\n",
			u, host, i, i, crawlDate)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRecrawlDirRoundTrip is the artifact gate: a build writes the membership directory
// next to the shards, and reloading it answers a member lookup with the document's
// global id and rejects a URL the collection does not hold. The global id read back
// must match the manifest's node base plus the document's row, the handle a recrawl
// uses to find the page it already has.
func TestRecrawlDirRoundTrip(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	urls := make([]string, 12)
	for i := range urls {
		urls[i] = fmt.Sprintf("https://host%02d.example/p%d", i%3, i)
	}
	if _, err := Build(Options{Source: writeCrawl(t, tmp, "a.jsonl", urls, "2026-06-01"), Out: out, ShardSize: 5}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if _, err := os.Stat(recrawlPath(out)); err != nil {
		t.Fatalf("recrawl directory not written: %v", err)
	}

	rd, err := LoadRecrawlDir(out)
	if err != nil {
		t.Fatalf("LoadRecrawlDir: %v", err)
	}
	if rd.Len() != 12 {
		t.Fatalf("directory holds %d pages, want 12", rd.Len())
	}

	// Build a canonical-URL to global-id map straight from the shards' forward stores,
	// the ground truth the directory must reproduce.
	want := forwardURLToGlobalID(t, out)
	if len(want) != 12 {
		t.Fatalf("forward stores hold %d distinct urls, want 12", len(want))
	}
	for raw, gid := range want {
		got, ok := rd.Lookup(raw)
		if !ok {
			t.Errorf("member %q not found in directory", raw)
			continue
		}
		if got != gid {
			t.Errorf("member %q: directory id %d, forward id %d", raw, got, gid)
		}
	}
	// A page the collection does not hold is rejected, the membership half of the
	// directory the cross-crawl dedup keys off.
	for i := 0; i < 200; i++ {
		u := fmt.Sprintf("https://absent%02d.example/q%d", i%5, i)
		if _, ok := rd.Lookup(u); ok {
			t.Fatalf("directory claims absent page %q", u)
		}
	}
	// A URL spelled with a tracking parameter and a fragment resolves to the same page
	// as its canonical form, the alias fold the canonical identity guarantees.
	if got, ok := rd.Lookup("https://host00.example/p0/?utm_source=x#top"); !ok || got != want["https://host00.example/p0"] {
		t.Errorf("alias spelling did not fold to the canonical page: got %d ok=%v", got, ok)
	}
}

// TestAddDropsRecrawledPages is the cross-crawl dedup gate: a collection is built from
// one crawl, then a later crawl that re-fetches some of those pages and carries some
// genuinely new ones is added. Only the new pages must build, the re-fetches dropped,
// so the collection grows by the new count rather than by the whole second crawl.
func TestAddDropsRecrawledPages(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")

	first := []string{
		"https://a.example/1", "https://a.example/2",
		"https://b.example/1", "https://b.example/2", "https://c.example/1",
	}
	if _, err := Build(Options{Source: writeCrawl(t, tmp, "a.jsonl", first, "2026-06-01"), Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// The second crawl re-fetches three pages (spelled with aliases to prove the fold)
	// and brings two new ones.
	second := []string{
		"https://a.example/1/?utm_source=news", // re-fetch of a.example/1
		"https://b.example/2#section",          // re-fetch of b.example/2
		"https://c.example/1",                  // re-fetch of c.example/1
		"https://d.example/1",                  // new
		"https://a.example/3",                  // new
	}
	res, err := Add(Options{Source: writeCrawl(t, tmp, "b.jsonl", second, "2026-06-09"), Out: out, ShardSize: 10})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if res.Docs != 2 {
		t.Fatalf("add built %d documents, want 2 (three re-fetches dropped)", res.Docs)
	}

	// The collection holds the original five plus the two new pages, no duplicates.
	total := totalDocs(t, out)
	if total != 7 {
		t.Fatalf("collection holds %d documents, want 7", total)
	}
	// Every original page is present exactly once and so is each new page; no canonical
	// URL appears twice.
	urls := forwardURLToGlobalID(t, out)
	if len(urls) != 7 {
		t.Fatalf("collection holds %d distinct canonical urls, want 7", len(urls))
	}
	for _, u := range []string{"https://a.example/1", "https://b.example/2", "https://c.example/1", "https://d.example/1", "https://a.example/3"} {
		cu, _ := analyze.CanonicalURL(u)
		if _, ok := urls[cu]; !ok {
			t.Errorf("expected page %q missing from collection", cu)
		}
	}
}

// TestAddAllRecrawledIsNoOp checks the degenerate case: a crawl whose every page the
// collection already holds adds nothing, leaving the collection unchanged rather than
// writing empty shards.
func TestAddAllRecrawledIsNoOp(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	urls := []string{"https://a.example/1", "https://a.example/2", "https://b.example/1"}
	src := writeCrawl(t, tmp, "a.jsonl", urls, "2026-06-01")
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	before := totalDocs(t, out)

	res, err := Add(Options{Source: src, Out: out, ShardSize: 10})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if res.Docs != 0 {
		t.Fatalf("re-adding the same crawl built %d documents, want 0", res.Docs)
	}
	if after := totalDocs(t, out); after != before {
		t.Fatalf("collection grew from %d to %d on a no-op add", before, after)
	}
}

// TestAddWithoutRecrawlDirFallsBack checks an add against a collection built before the
// membership directory existed still works: with no directory to dedup against, every
// document builds, the pre-directory behavior, rather than the add failing.
func TestAddWithoutRecrawlDirFallsBack(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeCrawl(t, tmp, "a.jsonl", []string{"https://a.example/1", "https://a.example/2"}, "2026-06-01"), Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	// Simulate an old collection by removing the directory the build just wrote.
	if err := os.Remove(recrawlPath(out)); err != nil {
		t.Fatal(err)
	}
	// This crawl re-fetches one held page and brings one new one; with no directory both
	// build, so the collection grows by two.
	res, err := Add(Options{Source: writeCrawl(t, tmp, "b.jsonl", []string{"https://a.example/1", "https://a.example/9"}, "2026-06-09"), Out: out, ShardSize: 10})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if res.Docs != 2 {
		t.Fatalf("add without a directory built %d documents, want 2 (no dedup)", res.Docs)
	}
	// The add wrote a fresh directory over the union, so a second add now dedups.
	if _, err := os.Stat(recrawlPath(out)); err != nil {
		t.Fatalf("add did not write a directory: %v", err)
	}
}

// TestDecodeRecrawlRejectsCorrupt checks a damaged artifact is refused, so a torn file
// triggers a rebuild rather than a misread membership oracle.
func TestDecodeRecrawlRejectsCorrupt(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeCrawl(t, tmp, "a.jsonl", []string{"https://a.example/1", "https://a.example/2"}, "2026-06-01"), Out: out, ShardSize: 10}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	good, err := os.ReadFile(recrawlPath(out))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeRecrawl(good); err != nil {
		t.Fatalf("decode of a good artifact failed: %v", err)
	}
	// Flip a byte in the body: the trailing CRC must catch it.
	bad := append([]byte(nil), good...)
	bad[len(bad)/2] ^= 0xff
	if _, err := decodeRecrawl(bad); err == nil {
		t.Errorf("decode accepted a corrupted artifact")
	}
	// Truncation is refused too.
	if _, err := decodeRecrawl(good[:len(good)/2]); err == nil {
		t.Errorf("decode accepted a truncated artifact")
	}
}

// forwardURLToGlobalID reads every shard's forward store and returns each document's
// canonical URL mapped to its global id (node base plus row), the ground truth the
// recrawl directory is checked against.
func forwardURLToGlobalID(t *testing.T, dir string) map[string]uint32 {
	t.Helper()
	infos, err := List(dir)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]uint32{}
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			t.Fatal(err)
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			t.Fatal(err)
		}
		fwd, err := forward.Open(b)
		if err != nil {
			t.Fatal(err)
		}
		for id := uint32(0); id < fwd.DocCount(); id++ {
			raw, _ := fwd.Column("url", id)
			cu, ok := analyze.CanonicalURL(string(raw))
			if !ok {
				continue
			}
			out[cu] = info.NodeBase + id
		}
		_ = r.Close()
	}
	return out
}

func totalDocs(t *testing.T, dir string) int {
	t.Helper()
	infos, err := List(dir)
	if err != nil {
		t.Fatal(err)
	}
	var total int
	for _, info := range infos {
		total += int(info.DocCount)
	}
	return total
}

// BenchmarkRecrawlLookup times a membership probe against a built directory, the work
// an add does once a held page to decide whether it is a re-fetch. It is the cost the
// cross-crawl dedup pays a document; at 100,000 shards an add probes the directory once
// a candidate document rather than rescanning shards, so this probe is the hot path.
func BenchmarkRecrawlLookup(b *testing.B) {
	tmp := b.TempDir()
	out := filepath.Join(tmp, "coll")
	const n = 50000
	var sb strings.Builder
	for i := 0; i < n; i++ {
		u := fmt.Sprintf("https://host%05d.example/p%d", i%1000, i)
		fmt.Fprintf(&sb, `{"url":%q,"host":%q,"markdown":"# P\nbody %d"}`+"\n", u, analyze.HostOf(u), i)
	}
	src := filepath.Join(tmp, "a.jsonl")
	if err := os.WriteFile(src, []byte(sb.String()), 0o644); err != nil {
		b.Fatal(err)
	}
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 8192}); err != nil {
		b.Fatalf("Build: %v", err)
	}
	rd, err := LoadRecrawlDir(out)
	if err != nil {
		b.Fatalf("LoadRecrawlDir: %v", err)
	}
	// Alternate a held page and an absent one so the benchmark times both the member and
	// the non-member path the dedup actually walks.
	queries := make([]string, 2*n)
	for i := 0; i < n; i++ {
		queries[2*i] = fmt.Sprintf("https://host%05d.example/p%d", i%1000, i)
		queries[2*i+1] = fmt.Sprintf("https://absent%05d.example/q%d", i%1000, i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var hits int
	for i := 0; i < b.N; i++ {
		if _, ok := rd.Lookup(queries[i%len(queries)]); ok {
			hits++
		}
	}
	_ = hits
}

// distinctCanonicalURLs reads the ccrawl source and returns the set of canonical URLs
// over every bodied page, the membership oracle the directory built from the same
// corpus must reproduce exactly.
func distinctCanonicalURLs(t *testing.T, path string) map[string]struct{} {
	t.Helper()
	src, err := convert.OpenSource(path)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()
	set := map[string]struct{}{}
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		set[cu] = struct{}{}
	}
	return set
}

// TestRecrawlOnCCrawl is the real-data gate. It builds a collection from a ccrawl
// parquet, then checks the membership directory holds exactly the corpus's distinct
// canonical URLs and answers a lookup for each one. It then adds the same source again:
// every page is a re-fetch, so the add must build nothing and the collection must not
// grow, the cross-crawl dedup proven over real, varied URLs rather than synthetic ones.
func TestRecrawlOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 4096}); err != nil {
		t.Fatalf("Build: %v", err)
	}

	want := distinctCanonicalURLs(t, ccrawlGraphParquet)
	if len(want) == 0 {
		t.Skip("no ccrawl pages")
	}
	rd, err := LoadRecrawlDir(out)
	if err != nil {
		t.Fatalf("LoadRecrawlDir: %v", err)
	}
	if int(rd.Len()) != len(want) {
		t.Fatalf("directory holds %d pages, want %d distinct canonical urls", rd.Len(), len(want))
	}
	for cu := range want {
		if _, ok := rd.Lookup(cu); !ok {
			t.Fatalf("real page %q missing from directory", cu)
		}
	}

	before := totalDocs(t, out)
	if before != len(want) {
		t.Fatalf("collection holds %d documents, want %d after intra-crawl dedup", before, len(want))
	}
	res, err := Add(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 4096})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if res.Docs != 0 {
		t.Fatalf("re-adding the whole crawl built %d documents, want 0 (all re-fetches)", res.Docs)
	}
	if after := totalDocs(t, out); after != before {
		t.Fatalf("collection grew from %d to %d re-adding the same crawl", before, after)
	}
}
