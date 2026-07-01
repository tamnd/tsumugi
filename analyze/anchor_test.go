package analyze

import (
	"reflect"
	"testing"

	"github.com/tamnd/tsumugi/convert"
)

// TestAnchorLinksKeepsTextAndTarget is the core of the extractor: an inline link
// yields both its visible text and the same normalized absolute target Links would
// resolve, so the inversion can index the phrase on the document the target names.
func TestAnchorLinksKeepsTextAndTarget(t *testing.T) {
	d := convert.Document{
		URL:  "https://a.example/page",
		Host: "a.example",
		Body: "see [the Go docs](https://b.example/docs) and [a relative one](/local).",
	}
	got := AnchorLinks(d)
	want := []AnchorLink{
		{Text: "the Go docs", URL: "https://b.example/docs"},
		{Text: "a relative one", URL: "https://a.example/local"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("AnchorLinks = %#v, want %#v", got, want)
	}
}

// TestAnchorLinksTargetsMatchLinks pins the invariant the inversion relies on: every
// target AnchorLinks emits is a target Links also emits, so an anchor edge resolves
// to the same document a graph edge does. AnchorLinks is a subset (it drops autolinks
// and empty-text links), never a superset.
func TestAnchorLinksTargetsMatchLinks(t *testing.T) {
	d := convert.Document{
		URL:  "https://a.example/",
		Host: "a.example",
		Body: "[one](https://b.example/x) <https://c.example/auto> [](https://d.example/empty) [two](https://b.example/y)",
	}
	linkSet := map[string]bool{}
	for _, u := range Links(d) {
		linkSet[u] = true
	}
	for _, a := range AnchorLinks(d) {
		if !linkSet[a.URL] {
			t.Errorf("anchor target %q is not among Links() targets %v", a.URL, Links(d))
		}
	}
}

// TestAnchorLinksSkipsAutolinksAndEmptyText makes the two subset rules explicit: an
// autolink has no describing phrase, and a link with empty or whitespace-only text
// names nothing, so neither contributes an anchor phrase.
func TestAnchorLinksSkipsAutolinksAndEmptyText(t *testing.T) {
	d := convert.Document{
		URL:  "https://a.example/",
		Host: "a.example",
		Body: "<https://b.example/auto> [   ](https://c.example/blank) [real](https://d.example/ok)",
	}
	got := AnchorLinks(d)
	if len(got) != 1 || got[0].Text != "real" || got[0].URL != "https://d.example/ok" {
		t.Fatalf("AnchorLinks = %#v, want a single real link", got)
	}
}

// TestAnchorLinksDropsSelf keeps a page's link to itself out of its own anchor set,
// the same self drop Links does, so a page cannot anchor-describe itself through a
// self link.
func TestAnchorLinksDropsSelf(t *testing.T) {
	d := convert.Document{
		URL:  "https://a.example/page",
		Host: "a.example",
		Body: "back to [myself](https://a.example/page) and out to [other](https://b.example/)",
	}
	got := AnchorLinks(d)
	if len(got) != 1 || got[0].URL != "https://b.example/" {
		t.Fatalf("AnchorLinks = %#v, want only the off-page link", got)
	}
}

// TestNormalizeAnchorText collapses internal whitespace and trims, and caps a
// pathologically long phrase at the rune limit without splitting a rune.
func TestNormalizeAnchorText(t *testing.T) {
	if got := normalizeAnchorText("  see   the\ndocs  "); got != "see the docs" {
		t.Errorf("whitespace collapse = %q, want %q", got, "see the docs")
	}
	long := make([]rune, maxAnchorRunes+50)
	for i := range long {
		long[i] = 'x'
	}
	if got := normalizeAnchorText(string(long)); len([]rune(got)) != maxAnchorRunes {
		t.Errorf("cap = %d runes, want %d", len([]rune(got)), maxAnchorRunes)
	}
}
