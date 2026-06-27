package analyze

import (
	"net/url"
	"regexp"
	"strings"

	"github.com/tamnd/tsumugi/convert"
)

// Outbound link extraction. The crawl export carries a page as markdown, so the
// page's links survive as markdown link syntax rather than as a separate column,
// and the link graph the offline signals need has to be recovered from the body.
// This file does that recovery: it pulls the outbound URLs out of the markdown,
// resolves the relative ones against the page URL, and normalizes them to a stable
// absolute form so the same target written two ways becomes one edge.
//
// Extraction lives here, downstream of the thin convert.Document, for the reason
// the package comment gives: everything derivable from one document in isolation is
// derived in one place. A document's outbound links are exactly that, derivable
// from its body and its own URL, so they belong next to the title and the content
// features rather than in the source adapter.

// mdInlineLink matches a markdown inline link, [text](url) with an optional title,
// capturing the URL. The URL run stops at whitespace or the closing paren, which is
// the common case; a URL with a literal paren in it is rare enough to skip rather
// than mis-split a title onto the end of it.
var mdInlineLink = regexp.MustCompile(`\[[^\]]*\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)`)

// mdAutoLink matches a markdown autolink, <url>, capturing the URL. Only http and
// https autolinks are taken; a mailto or other scheme in angle brackets is not an
// outbound web link and is left out.
var mdAutoLink = regexp.MustCompile(`<(https?://[^>\s]+)>`)

// Links returns the outbound links of one crawl document as normalized absolute
// URLs, deduplicated and with the page's own URL dropped, in first-seen order. The
// order is stable so a build is reproducible: the same body yields the same edge
// list every run. A document whose body carries no links, or whose own URL does not
// parse, returns nil, which the graph stage reads as a page with no out-links.
func Links(d convert.Document) []string {
	base, err := url.Parse(strings.TrimSpace(d.URL))
	if err != nil || base.Host == "" {
		return nil
	}
	self, _ := normalizeURL(base)

	var out []string
	seen := map[string]struct{}{}
	add := func(raw string) {
		ref, err := url.Parse(strings.TrimSpace(raw))
		if err != nil {
			return
		}
		abs, ok := normalizeURL(base.ResolveReference(ref))
		if !ok || abs == self {
			return
		}
		if _, dup := seen[abs]; dup {
			return
		}
		seen[abs] = struct{}{}
		out = append(out, abs)
	}

	for _, m := range mdAutoLink.FindAllStringSubmatch(d.Body, -1) {
		add(m[1])
	}
	for _, m := range mdInlineLink.FindAllStringSubmatch(d.Body, -1) {
		add(m[1])
	}
	return out
}

// CanonicalURL reduces a raw URL string to the same stable absolute form Links
// produces for its targets, so a document's own URL can be matched against the link
// targets of other documents to resolve an edge. The bool is false when the string
// is not a usable absolute web URL. It is the conservative normalization the link
// graph keys off; the canonical identity stage folds URLs harder.
func CanonicalURL(raw string) (string, bool) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "", false
	}
	return normalizeURL(u)
}

// normalizeURL reduces a URL to a stable absolute form for edge identity: it keeps
// only http and https, lowercases the scheme and host, drops the fragment, and
// drops the default port, so http://Example.com:80/a#x and http://example.com/a are
// one edge. It is deliberately conservative; full canonical-URL folding (path case,
// trailing slash, query ordering, tracking-param stripping) is the canonical
// identity stage's job, not the link extractor's. The bool is false when the URL is
// not a usable absolute web link.
func normalizeURL(u *url.URL) (string, bool) {
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
	u.Scheme = scheme
	u.Host = host
	u.Fragment = ""
	return u.String(), true
}
