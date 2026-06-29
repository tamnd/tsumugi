package analyze

import (
	"crypto/sha256"
	"net/url"
	"sort"
	"strings"
)

// DocID is doc 02's portable document identity: the SHA-256 of a URL's canonical
// form, the 32-byte key that names the same page across crawls and across shards. The
// bool is false when the URL is not a usable absolute web link, the same contract
// CanonicalURL keeps, so the build can skip a row that has no identity rather than
// store the hash of an empty string. It is the durable, human-auditable key the dense
// docID (a per-shard build-order position) and the global node id (an MPH slot) are
// not: two crawls of one page compute the same doc_id because they compute the same
// canonical URL, which is what lets a later crawl recognize a page it has seen.
func DocID(raw string) ([32]byte, bool) {
	cu, ok := CanonicalURL(raw)
	if !ok {
		return [32]byte{}, false
	}
	return sha256.Sum256([]byte(cu)), true
}

// Canonical URL identity (doc 02). A page is reachable under many URLs that name the
// same document, and an edge or a document counted once per spelling inflates the
// graph and splits a page's in-links across its aliases. This file folds a parsed
// URL to one frozen canonical form so the same page written two ways becomes one
// identity, the key both the link graph and the document MPH build on.
//
// The folding is the deterministic part of doc 02's algorithm, the steps that hold
// on essentially every server without per-host observation:
//
//	2  lowercase the scheme and host (the path and query keep their case)
//	4  drop the default port (80 for http, 443 for https)
//	5  drop the fragment (it never selects a different document from the server)
//	6  normalize percent-escapes (decode the unreserved ones, uppercase the rest)
//	7  resolve "." and ".." path segments and collapse duplicate slashes
//	8  drop a trailing slash except on the root path
//	9  drop tracking query parameters, then sort the survivors by key
//
// The observation-driven steps doc 02 also names are deliberately not here, because
// they need evidence a single URL string does not carry: step 3 (upgrade http to
// https) needs a per-host observation that the host serves https, step 10 (follow an
// in-page rel=canonical hint) needs the page's own declaration, and the redirect,
// www-vs-nonwww, and index.html folds need an observed redirect or a host rule. Those
// belong to the evidence stage that records crawl behavior, not the string fold, and
// folding them blindly here would merge pages that are not the same document.

// trackingParam reports whether a query key is a tracking parameter the canonical
// form drops (step 9). These keys never select a different document; they only carry
// campaign or session attribution, so two URLs that differ only in them are the same
// page and must fold to one identity. The utm_ family is matched by prefix and the
// rest by exact name, against the lowercased key.
func trackingParam(key string) bool {
	k := strings.ToLower(key)
	if strings.HasPrefix(k, "utm_") {
		return true
	}
	switch k {
	case "gclid", "fbclid", "dclid", "gclsrc", "msclkid", "yclid",
		"ref", "ref_src", "ref_url", "referrer",
		"sessionid", "session_id", "sid", "phpsessid", "jsessionid",
		"mc_cid", "mc_eid", "igshid", "_ga", "_gl", "spm", "scm":
		return true
	}
	return false
}

// normalizeURL folds a parsed URL to its canonical identity form, the string both an
// edge target and a document's own URL reduce to so the two can be matched. The bool
// is false when the URL is not a usable absolute web link (a scheme other than http
// or https, or no host). The steps it applies are the deterministic ones listed in
// the file comment; userinfo is dropped, since a canonical document identity does not
// carry credentials.
func normalizeURL(u *url.URL) (string, bool) {
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", false
	}
	host := strings.ToLower(u.Host)
	if host == "" {
		return "", false
	}
	// Step 4: drop the default port for the scheme.
	if i := strings.LastIndexByte(host, ':'); i >= 0 {
		port := host[i+1:]
		if (scheme == "http" && port == "80") || (scheme == "https" && port == "443") {
			host = host[:i]
		}
	}

	// Steps 6, 7, 8 on the path, then step 9 on the query. The path is taken in its
	// escaped form so an escaped slash (%2F) is not mistaken for a segment separator
	// when dot segments are resolved.
	path := canonicalPath(u.EscapedPath())
	query := canonicalQuery(u.RawQuery)

	var b strings.Builder
	b.Grow(len(scheme) + len(host) + len(path) + len(query) + 4)
	b.WriteString(scheme)
	b.WriteString("://")
	b.WriteString(host)
	b.WriteString(path)
	if query != "" {
		b.WriteByte('?')
		b.WriteString(query)
	}
	return b.String(), true
}

// canonicalPath applies steps 6, 7, and 8 to an escaped path. It normalizes the
// percent-escapes, then walks the segments resolving "." (drop) and ".." (pop) and
// dropping empty segments, which collapses duplicate slashes and removes the trailing
// slash in one pass. The result always starts with "/", and the root is "/" alone.
func canonicalPath(escaped string) string {
	p := normalizeEscapes(escaped)
	if p == "" || p == "/" {
		return "/"
	}
	out := make([]string, 0, strings.Count(p, "/")+1)
	for _, seg := range strings.Split(p, "/") {
		switch seg {
		case "", ".":
			// An empty segment is a leading, trailing, or duplicate slash; "." is the
			// current directory. Both drop, since the slashes are rebuilt on join.
		case "..":
			if len(out) > 0 {
				out = out[:len(out)-1]
			}
		default:
			out = append(out, seg)
		}
	}
	return "/" + strings.Join(out, "/")
}

// canonicalQuery applies step 9 to a raw query string. It splits the query into
// parameters, drops the tracking deny-list, normalizes the percent-escapes of every
// surviving key and value, and sorts by key (then value, so repeated keys are
// ordered deterministically). The result has no leading '?', and is empty when no
// parameter survives, so the caller drops the '?' entirely.
func canonicalQuery(raw string) string {
	if raw == "" {
		return ""
	}
	type param struct {
		key, val string
		hasVal   bool
	}
	var ps []param
	for _, part := range strings.Split(raw, "&") {
		if part == "" {
			continue
		}
		key, val := part, ""
		hasVal := false
		if i := strings.IndexByte(part, '='); i >= 0 {
			key, val, hasVal = part[:i], part[i+1:], true
		}
		key = normalizeEscapes(key)
		if trackingParam(key) {
			continue
		}
		ps = append(ps, param{key, normalizeEscapes(val), hasVal})
	}
	if len(ps) == 0 {
		return ""
	}
	sort.SliceStable(ps, func(i, j int) bool {
		if ps[i].key != ps[j].key {
			return ps[i].key < ps[j].key
		}
		return ps[i].val < ps[j].val
	})
	var b strings.Builder
	b.Grow(len(raw))
	for i, p := range ps {
		if i > 0 {
			b.WriteByte('&')
		}
		b.WriteString(p.key)
		if p.hasVal {
			b.WriteByte('=')
			b.WriteString(p.val)
		}
	}
	return b.String()
}

const upperHex = "0123456789ABCDEF"

// normalizeEscapes rewrites the percent-escapes in s (step 6): an escape of an
// unreserved character (ALPHA / DIGIT / '-' '.' '_' '~') is replaced by the literal
// character, and any other escape is kept but its two hex digits are uppercased to
// the canonical %XX form. A malformed escape (no two hex digits) is left as-is, and a
// byte that is not part of an escape passes through unchanged.
func normalizeEscapes(s string) string {
	if !strings.Contains(s, "%") {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		if s[i] == '%' && i+2 < len(s) && isHex(s[i+1]) && isHex(s[i+2]) {
			v := unhex(s[i+1])<<4 | unhex(s[i+2])
			if isUnreserved(v) {
				b.WriteByte(v)
			} else {
				b.WriteByte('%')
				b.WriteByte(upperHex[v>>4])
				b.WriteByte(upperHex[v&0x0f])
			}
			i += 3
			continue
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

// isUnreserved reports whether b is an RFC 3986 unreserved character, the set whose
// percent-escapes decode to the literal character with no change of meaning.
func isUnreserved(b byte) bool {
	switch {
	case b >= 'A' && b <= 'Z', b >= 'a' && b <= 'z', b >= '0' && b <= '9':
		return true
	case b == '-' || b == '.' || b == '_' || b == '~':
		return true
	}
	return false
}

func isHex(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F')
}

func unhex(b byte) byte {
	switch {
	case b >= '0' && b <= '9':
		return b - '0'
	case b >= 'a' && b <= 'f':
		return b - 'a' + 10
	default:
		return b - 'A' + 10
	}
}
