package analyze

import (
	"net/url"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
)

// TestCanonicalURLWorkedExample is doc 02's worked example: four spellings of one
// page fold to one identity, modulo the scheme. The three https spellings, which
// differ only in a default port, a trailing slash, a fragment, and a tracking query
// parameter, must produce the identical canonical URL. The http spelling, which also
// carries mixed case and a dot-segment detour in the path, folds to the same form but
// for the scheme, because the http-to-https upgrade is the observation-driven step 3
// this stage deliberately does not apply to a bare string.
func TestCanonicalURLWorkedExample(t *testing.T) {
	const wantHTTPS = "https://example.com/about?id=7"
	httpsVariants := []string{
		"https://example.com/about/?id=7&utm_source=x",
		"https://example.com:443/about?id=7#team",
		"https://Example.com/about?id=7",
	}
	for _, raw := range httpsVariants {
		got, ok := CanonicalURL(raw)
		if !ok || got != wantHTTPS {
			t.Errorf("CanonicalURL(%q) = %q, %v; want %q, true", raw, got, ok, wantHTTPS)
		}
	}
	const rawHTTP = "HTTP://Example.COM:80/About/../about/?utm_source=x&id=7#team"
	got, ok := CanonicalURL(rawHTTP)
	if want := "http://example.com/about?id=7"; !ok || got != want {
		t.Errorf("CanonicalURL(%q) = %q, %v; want %q, true", rawHTTP, got, ok, want)
	}
}

// TestCanonicalURLNoOverCollapse is the counter-example side of the worked example:
// the fold must not merge pages that differ in content. A query value that selects a
// different item and a path segment that selects a different page both stay distinct,
// because step 9 only drops tracking keys and step 7 only resolves dot segments.
func TestCanonicalURLNoOverCollapse(t *testing.T) {
	pairs := [][2]string{
		{"https://shop.test/item?id=7", "https://shop.test/item?id=8"},
		{"https://site.test/en/about", "https://site.test/fr/about"},
		{"https://site.test/About", "https://site.test/about"}, // path keeps its case
		{"https://site.test/a?x=1&y=2", "https://site.test/a?x=1"},
	}
	for _, p := range pairs {
		a, okA := CanonicalURL(p[0])
		b, okB := CanonicalURL(p[1])
		if !okA || !okB {
			t.Fatalf("CanonicalURL failed: %q->%v %q->%v", p[0], okA, p[1], okB)
		}
		if a == b {
			t.Errorf("over-collapsed distinct pages: %q and %q both -> %q", p[0], p[1], a)
		}
	}
}

// TestCanonicalURLSteps pins each deterministic step on a minimal input, so a
// regression names the exact step that broke rather than only failing the worked
// example.
func TestCanonicalURLSteps(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"lowercase scheme and host", "HTTPS://Example.COM/Path", "https://example.com/Path"},
		{"keep path case", "https://h.test/AbC/DeF", "https://h.test/AbC/DeF"},
		{"drop default http port", "http://h.test:80/a", "http://h.test/a"},
		{"drop default https port", "https://h.test:443/a", "https://h.test/a"},
		{"keep non-default port", "https://h.test:8443/a", "https://h.test:8443/a"},
		{"drop fragment", "https://h.test/a#section", "https://h.test/a"},
		{"resolve dot segments", "https://h.test/a/b/../c", "https://h.test/a/c"},
		{"resolve leading parent", "https://h.test/../a", "https://h.test/a"},
		{"resolve current dir", "https://h.test/a/./b", "https://h.test/a/b"},
		{"collapse duplicate slashes", "https://h.test/a//b///c", "https://h.test/a/b/c"},
		{"drop trailing slash", "https://h.test/a/b/", "https://h.test/a/b"},
		{"keep root slash", "https://h.test/", "https://h.test/"},
		{"empty path is root", "https://h.test", "https://h.test/"},
		{"decode unreserved escape", "https://h.test/a%2Db", "https://h.test/a-b"},
		{"uppercase reserved escape", "https://h.test/a%2fb", "https://h.test/a%2Fb"},
		{"keep escaped space", "https://h.test/a%20b", "https://h.test/a%20b"},
		{"sort query keys", "https://h.test/a?z=1&a=2", "https://h.test/a?a=2&z=1"},
		{"drop utm params", "https://h.test/a?utm_source=x&id=3", "https://h.test/a?id=3"},
		{"drop gclid", "https://h.test/a?gclid=abc&q=1", "https://h.test/a?q=1"},
		{"empty after filter", "https://h.test/a?utm_source=x", "https://h.test/a"},
		{"keep valueless key", "https://h.test/a?flag", "https://h.test/a?flag"},
		{"order repeated keys by value", "https://h.test/a?k=2&k=1", "https://h.test/a?k=1&k=2"},
	}
	for _, c := range cases {
		got, ok := CanonicalURL(c.in)
		if !ok || got != c.want {
			t.Errorf("%s: CanonicalURL(%q) = %q, %v; want %q, true", c.name, c.in, got, ok, c.want)
		}
	}
}

// TestCanonicalURLRejects checks the bool is false for a string that is not a usable
// absolute web URL, the contract the graph build relies on to skip a non-edge.
func TestCanonicalURLRejects(t *testing.T) {
	for _, raw := range []string{
		"mailto:x@y.test",
		"ftp://h.test/a",
		"/relative/only",
		"not a url",
		"http://",
		"",
	} {
		if got, ok := CanonicalURL(raw); ok {
			t.Errorf("CanonicalURL(%q) = %q, true; want false", raw, got)
		}
	}
}

// TestCanonicalURLIdempotent is the frozen-identity property the build depends on:
// canonicalizing an already-canonical URL is a no-op, so the MPH key a document is
// stored under is the same key a later lookup of the same page computes. A fold that
// were not idempotent would split a page's identity across build and query.
func TestCanonicalURLIdempotent(t *testing.T) {
	for _, raw := range []string{
		"HTTPS://Example.COM:443/A/../b//c/?z=1&utm_source=x&a=2#top",
		"http://h.test:80/p/./q/",
		"https://h.test/a%2Db%2fc",
		"https://h.test/",
	} {
		once, ok := CanonicalURL(raw)
		if !ok {
			t.Fatalf("CanonicalURL(%q) failed", raw)
		}
		twice, ok := CanonicalURL(once)
		if !ok || twice != once {
			t.Errorf("not idempotent: %q -> %q -> %q", raw, once, twice)
		}
	}
}

// TestCanonicalURLFoldsAliasesOnCCrawl measures the fold on the real crawl: it
// canonicalizes every page URL and confirms the result is well-formed and frozen, and
// logs how many raw spellings collapse to how many canonical identities, the dedup
// the document and graph stages get for free. Every canonical URL must be idempotent,
// carry no fragment, no default port, no trailing slash off the root, and no tracking
// parameter, the invariants the rest of the build assumes hold.
func TestCanonicalURLFoldsAliasesOnCCrawl(t *testing.T) {
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Skipf("ccrawl export not available: %v", err)
	}
	defer func() { _ = src.Close() }()

	raw := map[string]struct{}{}
	canon := map[string]struct{}{}
	var pages int
	for pages < 8000 {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read ccrawl: %v", err)
		}
		if !ok {
			break
		}
		cu, ok := CanonicalURL(d.URL)
		if !ok {
			continue
		}
		pages++
		raw[d.URL] = struct{}{}
		canon[cu] = struct{}{}

		again, ok := CanonicalURL(cu)
		if !ok || again != cu {
			t.Fatalf("page %q: canonical %q not idempotent (-> %q)", d.URL, cu, again)
		}
		u := mustParse(t, cu)
		if u.Fragment != "" {
			t.Fatalf("canonical %q keeps a fragment", cu)
		}
		if p := u.Port(); (u.Scheme == "http" && p == "80") || (u.Scheme == "https" && p == "443") {
			t.Fatalf("canonical %q keeps a default port", cu)
		}
		if u.RawQuery != "" {
			for _, part := range strings.Split(u.RawQuery, "&") {
				key := part
				if i := strings.IndexByte(part, '='); i >= 0 {
					key = part[:i]
				}
				if trackingParam(key) {
					t.Fatalf("canonical %q keeps tracking param %q", cu, key)
				}
			}
		}
		if path := u.EscapedPath(); path != "/" && len(path) > 0 && path[len(path)-1] == '/' {
			t.Fatalf("canonical %q keeps a non-root trailing slash", cu)
		}
	}
	if pages == 0 {
		t.Skip("no ccrawl documents")
	}
	t.Logf("pages=%d rawURLs=%d canonicalURLs=%d collapsed=%d",
		pages, len(raw), len(canon), len(raw)-len(canon))
}

// TestCanonicalURLFoldWorkOnCCrawl measures what the new deterministic steps buy on
// real link targets, where the dirty spellings live that the page-URL set does not
// have. It extracts every link the corpus carries, resolves it, and compares the
// distinct-target count under the old conservative fold (scheme and host case,
// default port, fragment only) against the full fold. The gap is the alias collapse
// steps 6 through 9 add, and it is logged so the slice's payoff on real data is on
// the record rather than asserted from crafted strings.
func TestCanonicalURLFoldWorkOnCCrawl(t *testing.T) {
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Skipf("ccrawl export not available: %v", err)
	}
	defer func() { _ = src.Close() }()

	conservative := map[string]struct{}{}
	full := map[string]struct{}{}
	var pages, targets int
	for pages < 8000 {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read ccrawl: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		pages++
		base, err := parseBase(d.URL)
		if err != nil {
			continue
		}
		for _, raw := range rawLinkStrings(d.Body) {
			ref, err := parseBase(raw)
			if err != nil {
				continue
			}
			abs := base.ResolveReference(ref)
			c, okC := conservativeNormalize(abs)
			f, okF := CanonicalURL(abs.String())
			if !okC || !okF {
				continue
			}
			targets++
			conservative[c] = struct{}{}
			full[f] = struct{}{}
		}
	}
	if targets == 0 {
		t.Skip("no ccrawl link targets")
	}
	collapsed := len(conservative) - len(full)
	t.Logf("pages=%d targets=%d conservativeDistinct=%d fullDistinct=%d aliasesFolded=%d (%.1f%%)",
		pages, targets, len(conservative), len(full), collapsed,
		100*float64(collapsed)/float64(len(conservative)))
	if collapsed <= 0 {
		t.Fatalf("the full fold collapsed no aliases the conservative fold kept; steps 6-9 are doing nothing on real data")
	}
}

// parseBase parses a URL string the way the extractor does, returning an error for a
// string with no host so the measurement skips it rather than counting a non-URL.
func parseBase(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, err
	}
	return u, nil
}

// rawLinkStrings pulls the raw link URLs out of a markdown body with the same two
// regexes the extractor uses, so the measurement sees exactly the spellings the build
// would.
func rawLinkStrings(body string) []string {
	var out []string
	for _, m := range mdAutoLink.FindAllStringSubmatch(body, -1) {
		out = append(out, m[1])
	}
	for _, m := range mdInlineLink.FindAllStringSubmatch(body, -1) {
		out = append(out, m[1])
	}
	return out
}

// conservativeNormalize is the fold this slice replaced: scheme and host lowercased,
// default port and fragment dropped, nothing else. The ccrawl A/B test compares its
// distinct-target count against the full fold to measure the alias collapse the new
// steps add. It is kept here, in the test, because it is no longer the production path.
func conservativeNormalize(u *url.URL) (string, bool) {
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", false
	}
	host := strings.ToLower(u.Host)
	if host == "" {
		return "", false
	}
	if i := strings.LastIndexByte(host, ':'); i >= 0 {
		port := host[i+1:]
		if (scheme == "http" && port == "80") || (scheme == "https" && port == "443") {
			host = host[:i]
		}
	}
	v := *u
	v.Scheme = scheme
	v.Host = host
	v.Fragment = ""
	return v.String(), true
}
