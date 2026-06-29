package collection

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
)

// TestDedupByIdentity pins the collapse rules on a crafted set: three spellings of one
// page (a bare URL, a trailing slash with a tracking parameter, a fragment) fold to one
// document, the most recent fetch of that page wins, two genuinely different pages both
// survive, and a URL with no canonical form is kept untouched. The survivors stay in
// first-seen order so the build is reproducible.
func TestDedupByIdentity(t *testing.T) {
	in := []convert.Document{
		{URL: "https://h.test/p", Body: "old", CrawlDate: "2026-06-01"},
		{URL: "https://other.test/q", Body: "q", CrawlDate: "2026-06-01"},
		{URL: "https://h.test/p/?utm_source=x", Body: "new", CrawlDate: "2026-06-09"},
		{URL: "https://h.test/p#frag", Body: "mid", CrawlDate: "2026-06-05"},
		{URL: "mailto:x@y.test", Body: "nocanon", CrawlDate: "2026-06-01"},
		{URL: "https://h.test/other", Body: "other", CrawlDate: "2026-06-02"},
	}
	out, dropped := dedupByIdentity(in)
	if dropped != 2 {
		t.Fatalf("dropped = %d, want 2", dropped)
	}
	wantURLs := []string{
		"https://h.test/p/?utm_source=x", // the most recent spelling of the folded page
		"https://other.test/q",
		"mailto:x@y.test",
		"https://h.test/other",
	}
	if len(out) != len(wantURLs) {
		t.Fatalf("survivors = %d, want %d: %+v", len(out), len(wantURLs), out)
	}
	for i, w := range wantURLs {
		if out[i].URL != w {
			t.Errorf("survivor %d url = %q, want %q", i, out[i].URL, w)
		}
	}
	// The folded page must have kept the most recent body, not the first-seen one.
	if out[0].Body != "new" {
		t.Errorf("folded page body = %q, want %q (most recent fetch)", out[0].Body, "new")
	}
	// Every survivor with a canonical form is distinct by it.
	seen := map[string]struct{}{}
	for _, d := range out {
		if cu, ok := analyze.CanonicalURL(d.URL); ok {
			if _, dup := seen[cu]; dup {
				t.Errorf("survivor %q duplicates a canonical URL", d.URL)
			}
			seen[cu] = struct{}{}
		}
	}
}

// TestDedupByIdentityTieKeepsFirst checks the deterministic tie-break: when two
// spellings of one page share a crawl date, the one already held stays, so the build
// does not depend on map iteration order.
func TestDedupByIdentityTieKeepsFirst(t *testing.T) {
	in := []convert.Document{
		{URL: "https://h.test/a", Body: "first", CrawlDate: "2026-06-01"},
		{URL: "https://h.test/a/", Body: "second", CrawlDate: "2026-06-01"},
	}
	out, dropped := dedupByIdentity(in)
	if dropped != 1 || len(out) != 1 {
		t.Fatalf("out=%d dropped=%d, want 1/1", len(out), dropped)
	}
	if out[0].Body != "first" {
		t.Errorf("tie kept body %q, want %q", out[0].Body, "first")
	}
}

// TestBuildDedupsDuplicateURLs is the end-to-end gate: a crawl that carries one page
// under three spellings plus two distinct pages builds a collection of three
// documents, not five, so the duplicates never reach the index.
func TestBuildDedupsDuplicateURLs(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "crawl.jsonl")
	lines := []string{
		`{"url":"https://h.example/p","host":"h.example","markdown":"# P\nbody one","crawl_date":"2026-06-01"}`,
		`{"url":"https://h.example/p/?utm_source=x","host":"h.example","markdown":"# P\nbody one","crawl_date":"2026-06-09"}`,
		`{"url":"https://h.example/p#top","host":"h.example","markdown":"# P\nbody one","crawl_date":"2026-06-05"}`,
		`{"url":"https://h.example/q","host":"h.example","markdown":"# Q\nbody two","crawl_date":"2026-06-01"}`,
		`{"url":"https://g.example/r","host":"g.example","markdown":"# R\nbody three","crawl_date":"2026-06-01"}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(tmp, "coll")
	res, err := Build(Options{Source: path, Out: out, ShardSize: 10})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if res.Docs != 3 {
		t.Fatalf("built %d docs, want 3 (two duplicate spellings dropped)", res.Docs)
	}
}

// TestDedupOnCCrawl runs the collapse over the real crawl: it reads every page,
// applies the identity dedup, and checks the survivors are exactly the distinct
// canonical URLs, with the drop count equal to the alias spellings the corpus carries.
// It logs the numbers so the dedup the build performs on real data is on the record.
func TestDedupOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	var raw []convert.Document
	canon := map[string]struct{}{}
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
		raw = append(raw, d)
		if cu, ok := analyze.CanonicalURL(d.URL); ok {
			canon[cu] = struct{}{}
		}
	}
	_ = src.Close()
	if len(raw) == 0 {
		t.Skip("no ccrawl documents with a body")
	}

	out, dropped := dedupByIdentity(raw)
	t.Logf("raw=%d survivors=%d dropped=%d distinctCanonical=%d", len(raw), len(out), dropped, len(canon))
	if len(raw)-dropped != len(out) {
		t.Fatalf("survivors %d != raw %d - dropped %d", len(out), len(raw), dropped)
	}
	// Every survivor with a canonical form is unique by it, and the survivor count of
	// canonicalizable pages matches the distinct canonical URL count.
	seen := map[string]struct{}{}
	for _, d := range out {
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		if _, dup := seen[cu]; dup {
			t.Fatalf("survivor %q duplicates a canonical URL", d.URL)
		}
		seen[cu] = struct{}{}
	}
	if len(seen) != len(canon) {
		t.Fatalf("distinct canonical survivors %d != distinct canonical URLs %d", len(seen), len(canon))
	}
}

