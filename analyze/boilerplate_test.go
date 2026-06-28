package analyze

import (
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
)

// TestBoilerplateRatioProseIsLow checks a page of prose with one inline link scores
// near zero: almost all its visible text is content, not link chrome.
func TestBoilerplateRatioProseIsLow(t *testing.T) {
	body := strings.Join([]string{
		"# A real article",
		"",
		"This is a long paragraph of genuine prose about a topic, with many words of",
		"actual content and only a single [reference](https://example.com/source) buried",
		"in the middle of the sentence among all the other words that carry the meaning.",
	}, "\n")
	r := boilerplateRatio(body)
	if r < 0 || r > 1 {
		t.Fatalf("ratio %g out of range", r)
	}
	if r > 0.2 {
		t.Fatalf("prose boilerplate ratio %g, want low (mostly content)", r)
	}
}

// TestBoilerplateRatioNavIsHigh checks a navigation block of link rows scores near
// one: its visible text is almost all link labels, the markdown shape of chrome.
func TestBoilerplateRatioNavIsHigh(t *testing.T) {
	body := strings.Join([]string{
		"[Home](https://x.test/) [About](https://x.test/about) [Blog](https://x.test/blog)",
		"[Contact](https://x.test/contact) [Login](https://x.test/login)",
	}, "\n")
	r := boilerplateRatio(body)
	if r < 0.9 {
		t.Fatalf("nav-only boilerplate ratio %g, want near 1", r)
	}
}

// TestBoilerplateRatioMixed checks a page that is half nav chrome and half prose
// lands between the two extremes, the discrimination the signal exists to make.
func TestBoilerplateRatioMixed(t *testing.T) {
	nav := "[Home](https://x.test/) [About](https://x.test/about) [Blog](https://x.test/blog) [Shop](https://x.test/shop)"
	prose := "This paragraph is plain prose carrying the actual content of the page with no links at all in it whatsoever here."
	mixed := nav + "\n\n" + prose
	r := boilerplateRatio(mixed)
	full := boilerplateRatio(nav)
	none := boilerplateRatio(prose)
	if !(r > none && r < full) {
		t.Fatalf("mixed ratio %g should sit between prose %g and nav %g", r, none, full)
	}
}

// TestBoilerplateRatioEmpty checks a body with no visible text scores zero rather
// than dividing by zero.
func TestBoilerplateRatioEmpty(t *testing.T) {
	if r := boilerplateRatio(""); r != 0 {
		t.Fatalf("empty body ratio %g, want 0", r)
	}
	if r := boilerplateRatio("\n\n   \n"); r != 0 {
		t.Fatalf("whitespace body ratio %g, want 0", r)
	}
}

func BenchmarkBoilerplateRatio(b *testing.B) {
	var sb strings.Builder
	// A page-shaped body: a nav row, several prose paragraphs, and a footer link row.
	sb.WriteString("[Home](https://x.test/) [About](https://x.test/about) [Blog](https://x.test/blog) [Shop](https://x.test/shop)\n\n")
	for i := 0; i < 40; i++ {
		sb.WriteString("This is a paragraph of genuine prose content carrying the meaning of the page with the occasional [link](https://x.test/ref) in it.\n\n")
	}
	sb.WriteString("[Privacy](https://x.test/privacy) [Terms](https://x.test/terms) [Contact](https://x.test/contact)\n")
	body := sb.String()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = boilerplateRatio(body)
	}
}

// TestBoilerplateFeatureWired checks Document populates the boilerplate column and a
// link-heavy page reads higher than a prose page through the public surface.
func TestBoilerplateFeatureWired(t *testing.T) {
	prose := Document(convert.Document{
		URL:  "https://h.test/article",
		Body: "# Title\n\nA full paragraph of real prose content with plenty of words and no links to speak of anywhere in the text.",
	})
	nav := Document(convert.Document{
		URL:  "https://h.test/index",
		Body: "[One](https://h.test/1) [Two](https://h.test/2) [Three](https://h.test/3) [Four](https://h.test/4)",
	})
	p := prose.Features[feature.FeatBoilerplate]
	n := nav.Features[feature.FeatBoilerplate]
	if n <= p {
		t.Fatalf("nav boilerplate %g should exceed prose %g", n, p)
	}
	if p < 0 || p > 1 || n < 0 || n > 1 {
		t.Fatalf("boilerplate out of [0,1]: prose %g nav %g", p, n)
	}
}
