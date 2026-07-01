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
	"github.com/tamnd/tsumugi/lexical"
)

// anchorCCrawlDocs reads up to limit documents from the real ccrawl parquet the way
// the build does (bodyless records skipped), for the anchor tests and benchmark that
// need the real markdown link distribution rather than a synthetic one. It takes a
// testing.TB so the benchmark can share it with the tests.
func anchorCCrawlDocs(tb testing.TB, limit int) []convert.Document {
	tb.Helper()
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		tb.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		tb.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()
	var docs []convert.Document
	for len(docs) < limit {
		d, ok, err := src.Next()
		if err != nil {
			tb.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
	}
	return docs
}

// TestAnchorExtractionOnCCrawl exercises the anchor extractor over the real crawl's
// markdown, the distribution that matters: real inline links with real describing
// text and real url spellings, not a hand-built body. It asserts the extractor is
// well-formed on that input rather than that the inversion resolves in-corpus. This
// slice of the crawl is near one page per host, so almost no link points at another
// captured page (the same near-flat property that has the graph tests inject synthetic
// far edges); natural inbound anchor resolution is a corpus property the extractor does
// not control. What the extractor must guarantee on real input: every phrase it emits
// is non-empty and whitespace-normalized, and every target it emits is also a Links()
// target, so an anchor edge can never resolve to a document a graph edge would not.
func TestAnchorExtractionOnCCrawl(t *testing.T) {
	docs := anchorCCrawlDocs(t, 100000)

	totalPhrases := 0
	for _, d := range docs {
		anchors := analyze.AnchorLinks(d)
		totalPhrases += len(anchors)

		linkSet := map[string]bool{}
		for _, u := range analyze.Links(d) {
			linkSet[u] = true
		}
		for _, a := range anchors {
			if a.Text == "" {
				t.Fatalf("empty anchor text emitted for %q -> %q", d.URL, a.URL)
			}
			if a.Text != normalizeAnchorTextForTest(a.Text) {
				t.Fatalf("anchor text %q not whitespace-normalized", a.Text)
			}
			if !linkSet[a.URL] {
				t.Fatalf("anchor target %q from %q is not a Links() target", a.URL, d.URL)
			}
		}
	}
	if totalPhrases == 0 {
		t.Fatal("real corpus yielded no inline-link anchor text at all; extractor found nothing")
	}
	t.Logf("extracted %d inline anchor phrases over %d real docs", totalPhrases, len(docs))
}

// normalizeAnchorTextForTest mirrors analyze.normalizeAnchorText's whitespace fold so
// the real-data test can assert an emitted phrase is already in normal form without
// reaching into the analyze package's unexported helper. It does not apply the rune
// cap, which the phrases seen here never reach.
func normalizeAnchorTextForTest(s string) string { return strings.Join(strings.Fields(s), " ") }

// benchAnchorCorpus builds a dense synthetic corpus of n documents where each links a
// handful of others across a spread of domains, so the inversion has real work to do,
// unlike the near-flat real slice where almost nothing resolves in-corpus. It is the
// input BenchmarkAnchorFields measures the inversion throughput on.
func benchAnchorCorpus(n int) []convert.Document {
	docs := make([]convert.Document, n)
	for i := 0; i < n; i++ {
		host := fmt.Sprintf("h%03d.example", i%200)
		var b strings.Builder
		fmt.Fprintf(&b, "page %d body ", i)
		for k := 1; k <= 8; k++ {
			tgt := (i + k*97) % n
			fmt.Fprintf(&b, "[describe target %d](https://h%03d.example/p%d) ", tgt, tgt%200, tgt)
		}
		docs[i] = convert.Document{
			URL:  fmt.Sprintf("https://%s/p%d", host, i),
			Host: host,
			Body: b.String(),
		}
	}
	return docs
}

func BenchmarkAnchorFields(b *testing.B) {
	const n = 50000
	docs := benchAnchorCorpus(n)
	dir := buildDir(docs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = anchorFields(docs, dir)
	}
}

// TestAssembleAnchorFieldWeighting checks the frequency each phrase reaches in a
// target's anchor field: a phrase enters once per distinct source domain, an
// off-domain phrase enters at double that, and no phrase exceeds the cap. Phrases
// are single tokens here so a whitespace split counts each phrase's frequency
// directly. The emitted order is sorted, so the assembly is deterministic.
func TestAssembleAnchorFieldWeighting(t *testing.T) {
	manyDomains := map[int]bool{}
	for d := 100; d < 130; d++ { // 30 distinct domains, well over the cap
		manyDomains[d] = true
	}
	phrases := map[string]*phraseInfo{
		"alpha":   {domains: map[int]bool{1: true, 2: true}, off: true}, // 2 domains * 2 = 4
		"bravo":   {domains: map[int]bool{9: true}, off: false},         // 1 domain, within = 1
		"charlie": {domains: manyDomains, off: true},                    // 60 -> capped at 16
	}
	got := assembleAnchorField(phrases)

	counts := map[string]int{}
	for _, tok := range strings.Fields(got) {
		counts[tok]++
	}
	if counts["alpha"] != 4 {
		t.Errorf("alpha frequency = %d, want 4", counts["alpha"])
	}
	if counts["bravo"] != 1 {
		t.Errorf("bravo frequency = %d, want 1", counts["bravo"])
	}
	if counts["charlie"] != maxAnchorPhraseWeight {
		t.Errorf("charlie frequency = %d, want %d (capped)", counts["charlie"], maxAnchorPhraseWeight)
	}
	// Sorted emission: alpha before bravo before charlie.
	if want := "alpha alpha alpha alpha bravo"; !strings.HasPrefix(got, want) {
		t.Errorf("assembled field = %q, want prefix %q", got, want)
	}
}

// TestAnchorFieldsOffDomainOutweighs runs the inversion over a tiny corpus where the
// same target is linked once from off-domain and once from within its own domain
// under two different phrases. The off-domain phrase must enter the target's anchor
// field at higher frequency than the within-domain one, the endorsement asymmetry
// the spec calls for.
func TestAnchorFieldsOffDomainOutweighs(t *testing.T) {
	docs := []convert.Document{
		{URL: "https://alpha.example/a", Host: "alpha.example", Body: "out [offword](https://charlie.example/target)"},
		{URL: "https://charlie.example/other", Host: "charlie.example", Body: "out [sameword](https://charlie.example/target)"},
		{URL: "https://charlie.example/target", Host: "charlie.example", Body: "plain body"},
	}
	dir := buildDir(docs)
	fields := anchorFields(docs, dir)

	// Find the target's field by its URL, since the order is the input order here (no
	// build permutation runs in this direct call).
	var target string
	for i, d := range docs {
		if d.URL == "https://charlie.example/target" {
			target = fields[i]
		}
	}
	counts := map[string]int{}
	for _, tok := range strings.Fields(target) {
		counts[tok]++
	}
	if counts["offword"] <= counts["sameword"] {
		t.Fatalf("off-domain anchor %d should outweigh within-domain %d (field %q)", counts["offword"], counts["sameword"], target)
	}
	if counts["offword"] != offDomainAnchorWeight || counts["sameword"] != 1 {
		t.Fatalf("offword=%d sameword=%d, want %d and 1", counts["offword"], counts["sameword"], offDomainAnchorWeight)
	}
}

// TestAnchorFieldsDeterministic builds the anchor fields twice over the same corpus
// and requires them byte-identical, the property the reproducibility gate rests on.
func TestAnchorFieldsDeterministic(t *testing.T) {
	docs := []convert.Document{
		{URL: "https://a.example/1", Host: "a.example", Body: "[to b](https://b.example/x) [to c](https://c.example/y)"},
		{URL: "https://b.example/x", Host: "b.example", Body: "[to c](https://c.example/y) [to a](https://a.example/1)"},
		{URL: "https://c.example/y", Host: "c.example", Body: "[to a](https://a.example/1)"},
	}
	dir := buildDir(docs)
	a := anchorFields(docs, dir)
	b := anchorFields(docs, dir)
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("anchor field %d differs: %q vs %q", i, a[i], b[i])
		}
	}
}

// writeAnchorJSONL writes a small cross-linked crawl where an off-domain source and a
// same-domain source both link one target under words the target's own text never
// uses. It returns the source path and the target's canonical URL.
func writeAnchorJSONL(t *testing.T, dir string) (src, targetURL string) {
	t.Helper()
	targetURL = "https://target.example/doc"
	lines := []string{
		// Off-domain source: the word "zebranchor" is inbound anchor text for the target
		// and appears in the target's field only, never in the target's own body.
		`{"url":"https://source.example/a","host":"source.example","markdown":"intro [zebranchor beacon](https://target.example/doc) outro filler words here"}`,
		// Same-domain source, a different describing word.
		`{"url":"https://target.example/hub","host":"target.example","markdown":"see [quokkanchor overview](https://target.example/doc) for details and more"}`,
		// The target: its own text carries a unique body word but neither anchor word.
		`{"url":"https://target.example/doc","host":"target.example","markdown":"# Target\ntanukibody main content of the page"}`,
		// Filler so the graph reorder has more than a trivial set to order.
		`{"url":"https://source.example/b","host":"source.example","markdown":"more [zebranchor beacon](https://target.example/doc) linking text"}`,
		`{"url":"https://third.example/c","host":"third.example","markdown":"unrelated [zebranchor beacon](https://target.example/doc) page body"}`,
	}
	path := filepath.Join(dir, "anchor.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return path, targetURL
}

// findByURL scans a shard's forward url column for the given URL and returns its
// dense docID, so a test can name a document by its stable URL rather than guess the
// id the graph reorder assigned.
func findByURL(t *testing.T, fwd *forward.Region, n uint32, url string) uint32 {
	t.Helper()
	for id := uint32(0); id < n; id++ {
		if v, ok := fwd.Column("url", id); ok && string(v) == url {
			return id
		}
	}
	t.Fatalf("url %q not found in forward store", url)
	return 0
}

// TestAnchorFieldMakesInboundTermSearchable is the end-to-end gate: a term that
// appears only in a document's inbound anchor text, never in its own body, title, or
// url, retrieves that document after a build, because the anchor field was populated
// into the lexical index. It also confirms the document is still retrievable by its
// own unique body word, so the anchor field did not displace the existing fields.
func TestAnchorFieldMakesInboundTermSearchable(t *testing.T) {
	tmp := t.TempDir()
	src, targetURL := writeAnchorJSONL(t, tmp)
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 1000}); err != nil {
		t.Fatal(err)
	}

	infos, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 1 {
		t.Fatalf("got %d shards, want 1", len(infos))
	}
	r, err := tsumugi.Open(infos[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()

	lb, err := r.Region(tsumugi.RegionLexical)
	if err != nil {
		t.Fatal(err)
	}
	lr, err := lexical.Open(lb)
	if err != nil {
		t.Fatal(err)
	}
	fb, err := r.Region(tsumugi.RegionForward)
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := forward.Open(fb)
	if err != nil {
		t.Fatal(err)
	}

	targetID := findByURL(t, fwd, r.DocCount(), targetURL)

	// The target carries "zebranchor" only through inbound anchor text. If it comes
	// back for that term, the anchor field was indexed on it.
	assertRetrieves := func(term string) {
		cands, err := lr.SearchTerms([]string{term}, 50)
		if err != nil {
			t.Fatalf("search %q: %v", term, err)
		}
		for _, c := range cands {
			if c.DocID == targetID {
				return
			}
		}
		t.Fatalf("term %q did not retrieve the target doc %d (got %v)", term, targetID, cands)
	}

	assertRetrieves("zebranchor")  // off-domain inbound anchor only
	assertRetrieves("quokkanchor") // same-domain inbound anchor only
	assertRetrieves("tanukibody")  // the target's own body word, still works

	// The target's own body has none of the anchor words, so the only way it scores for
	// them is the anchor field. Guard that assumption so the test cannot pass vacuously.
	if body, ok := fwd.Column("body", targetID); ok {
		for _, w := range []string{"zebranchor", "quokkanchor"} {
			if strings.Contains(strings.ToLower(string(body)), w) {
				t.Fatalf("target body unexpectedly contains %q, test is not isolating the anchor field", w)
			}
		}
	}
}
