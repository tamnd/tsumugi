package analyze

import (
	"encoding/hex"
	"testing"
)

// TestDocIDPinned locks the identity to an exact value, so a change to either the
// canonicalization or the hash that would silently renumber every document across the
// whole corpus is caught here. The id is sha256 of the canonical URL, and a tracking
// parameter and a fragment that fold away in canonicalization must not change it.
func TestDocIDPinned(t *testing.T) {
	const want = "746af0ced5836908db7e0ad21c3775e1da5db2552cdebe8054f718f600b24eee"
	for _, raw := range []string{
		"https://example.com/about",
		"https://example.com/about/?utm_source=x#team",
		"https://Example.com:443/about",
	} {
		id, ok := DocID(raw)
		if !ok {
			t.Fatalf("DocID(%q) not ok", raw)
		}
		if got := hex.EncodeToString(id[:]); got != want {
			t.Errorf("DocID(%q) = %s, want %s", raw, got, want)
		}
	}
}

// TestDocIDAliasesAgree is the cross-crawl property the key exists for: every spelling
// of one page produces the same id, so a later crawl recognizes a page it has seen,
// and distinct pages produce distinct ids, so two pages are never merged.
func TestDocIDAliasesAgree(t *testing.T) {
	aliases := []string{
		"https://shop.test/p/?id=7&utm_campaign=spring",
		"https://shop.test:443/p?id=7#reviews",
		"https://SHOP.test/p?id=7",
	}
	first, ok := DocID(aliases[0])
	if !ok {
		t.Fatalf("DocID(%q) not ok", aliases[0])
	}
	for _, a := range aliases[1:] {
		id, ok := DocID(a)
		if !ok || id != first {
			t.Errorf("alias %q -> %x, want %x", a, id, first)
		}
	}
	for _, other := range []string{
		"https://shop.test/p?id=8", // different item
		"https://shop.test/q?id=7", // different path
		"https://other.test/p?id=7",
	} {
		id, ok := DocID(other)
		if !ok {
			t.Fatalf("DocID(%q) not ok", other)
		}
		if id == first {
			t.Errorf("distinct page %q collided with %q", other, aliases[0])
		}
	}
}

// TestDocIDRejects checks the bool is false for a URL with no canonical form, the
// contract the build relies on to store the zero id rather than the hash of an empty
// string.
func TestDocIDRejects(t *testing.T) {
	for _, raw := range []string{"mailto:x@y.test", "/relative", "not a url", ""} {
		if id, ok := DocID(raw); ok {
			t.Errorf("DocID(%q) = %x, true; want false", raw, id)
		}
	}
}
