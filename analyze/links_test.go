package analyze

import (
	"net/url"
	"reflect"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
)

// TestLinksExtractsAndNormalizes covers the shapes the markdown body carries: an
// autolink, an inline link with a title, a relative link resolved against the page,
// a root-relative link, a protocol-relative link, a scheme that is not a web link, a
// fragment-only link, the page's own URL, and a duplicate. The result is the
// normalized, deduplicated, self-dropped edge list in first-seen order.
func TestLinksExtractsAndNormalizes(t *testing.T) {
	d := convert.Document{
		URL: "https://site.test/dir/page",
		Body: strings.Join([]string{
			"see <https://Other.test:443/a#frag> for more",      // autolink, default port + fragment stripped, host lowercased
			"an [inline](https://b.test/x \"with title\") link", // inline with title
			"a [relative](sub/leaf) link",                       // resolves against /dir/page -> /dir/sub/leaf
			"a [rooted](/top) link",                             // root-relative -> /top
			"a [scheme-relative](//c.test/y) link",              // protocol-relative inherits https
			"a [mail](mailto:x@y.test) link",                    // not a web link
			"a [frag](#section) link",                           // fragment only, resolves to self
			"a [self](https://site.test/dir/page) link",         // the page's own URL
			"a [dup](https://b.test/x) link",                    // duplicate of the inline target
		}, "\n"),
	}
	got := Links(d)
	want := []string{
		"https://other.test/a",
		"https://b.test/x",
		"https://site.test/dir/sub/leaf",
		"https://site.test/top",
		"https://c.test/y",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Links\n got=%v\nwant=%v", got, want)
	}
}

// TestLinksDropsSelfLoops checks that every spelling of the page's own URL is
// dropped, since a self-loop is not an edge the graph wants. The fragment and the
// default port make the raw strings differ from the page URL but normalize to it.
func TestLinksDropsSelfLoops(t *testing.T) {
	d := convert.Document{
		URL:  "http://h.test:80/p",
		Body: "x <http://h.test/p> y [a](http://h.test/p#top) z [b](http://other.test/q)",
	}
	got := Links(d)
	want := []string{"http://other.test/q"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Links\n got=%v\nwant=%v", got, want)
	}
}

// TestLinksBadBaseURL returns nil rather than panicking when the page URL does not
// parse to an absolute web URL, since without a base the relative links cannot be
// resolved.
func TestLinksBadBaseURL(t *testing.T) {
	if got := Links(convert.Document{URL: "::::", Body: "[a](b)"}); got != nil {
		t.Fatalf("bad base url: got %v, want nil", got)
	}
	if got := Links(convert.Document{URL: "", Body: "[a](http://h.test/x)"}); got != nil {
		t.Fatalf("empty base url: got %v, want nil", got)
	}
}

// TestLinksEmptyBody returns nil for a page with no links.
func TestLinksEmptyBody(t *testing.T) {
	if got := Links(convert.Document{URL: "https://h.test/p", Body: "no links here"}); got != nil {
		t.Fatalf("no links: got %v, want nil", got)
	}
}

// ccrawlParquet is the real Common Crawl markdown shard the link extractor is
// exercised against, the content and language distribution the engine serves. The
// test skips when the file is absent so the suite runs without the data.
const ccrawlParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// TestLinksOnCCrawl runs the extractor over the real crawl and asserts the
// properties the graph stage relies on, on real-world markdown rather than crafted
// strings: a meaningful fraction of pages carry links, every extracted link is an
// absolute http or https URL with a host, no page links to itself, and each page's
// list is deduplicated. It logs the yield so the edge density the graph build will
// see is on the record.
func TestLinksOnCCrawl(t *testing.T) {
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Skipf("ccrawl export not available: %v", err)
	}
	defer func() { _ = src.Close() }()

	var pages, withLinks, edges int
	for pages < 5000 {
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
		links := Links(d)
		if len(links) > 0 {
			withLinks++
		}
		edges += len(links)

		self, _ := normalizeURL(mustParse(t, d.URL))
		seen := map[string]struct{}{}
		for _, l := range links {
			u := mustParse(t, l)
			if s := strings.ToLower(u.Scheme); s != "http" && s != "https" {
				t.Fatalf("page %s: link %q has non-web scheme", d.URL, l)
			}
			if u.Host == "" {
				t.Fatalf("page %s: link %q has no host", d.URL, l)
			}
			if u.Fragment != "" {
				t.Fatalf("page %s: link %q keeps a fragment", d.URL, l)
			}
			if l == self {
				t.Fatalf("page %s: link list contains a self-loop", d.URL)
			}
			if _, dup := seen[l]; dup {
				t.Fatalf("page %s: link %q is duplicated", d.URL, l)
			}
			seen[l] = struct{}{}
		}
	}
	if pages == 0 {
		t.Skip("no ccrawl documents")
	}
	frac := float64(withLinks) / float64(pages)
	t.Logf("pages=%d withLinks=%d (%.1f%%) edges=%d avgPerPage=%.1f",
		pages, withLinks, frac*100, edges, float64(edges)/float64(pages))
	if frac < 0.1 {
		t.Fatalf("only %.1f%% of real pages carry links, want at least 10%%; extraction is missing the markdown link forms", frac*100)
	}
}

func mustParse(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse %q: %v", raw, err)
	}
	return u
}
