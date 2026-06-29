package analyze

import (
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
)

func TestDocumentTitleAndFeatures(t *testing.T) {
	d := convert.Document{
		URL:  "https://example.com/docs/guide/intro",
		Host: "example.com",
		Body: "# Getting Started\nThis is a prose page about getting started with the tool.",
	}
	a := Document(d)

	if a.Title != "Getting Started" {
		t.Errorf("title = %q, want the markdown heading", a.Title)
	}
	if got := a.Features[feature.FeatHTTPS]; got != 1 {
		t.Errorf("https feature = %v, want 1 for an https url", got)
	}
	if got := a.Features[feature.FeatURLDepth]; got != 3 {
		t.Errorf("url depth = %v, want 3 path segments", got)
	}
	// FeatLanguage is no longer derived per document here; it holds the detected-language
	// id the build fills over the whole collection, so the analyze stage leaves it unset.
	if _, ok := a.Features[feature.FeatLanguage]; ok {
		t.Errorf("language feature set in analyze, want it filled by the build's detection pass")
	}
	if got := a.Features[feature.FeatDocLen]; got <= 0 {
		t.Errorf("doc len = %v, want a positive token count", got)
	}
	if got := a.Features[feature.FeatContentQuality]; got <= 0 || got > 100 {
		t.Errorf("content quality = %v, want a 0..100 score", got)
	}
}

func TestDeriveTitleFallsBackToFirstLine(t *testing.T) {
	d := convert.Document{URL: "http://x.test/p", Body: "plain opening line\nmore text"}
	a := Document(d)
	if a.Title != "plain opening line" {
		t.Errorf("title = %q, want the first non-empty line when there is no heading", a.Title)
	}
	if got := a.Features[feature.FeatHTTPS]; got != 0 {
		t.Errorf("https feature = %v, want 0 for an http url", got)
	}
}

func TestEmptyBodyTitle(t *testing.T) {
	a := Document(convert.Document{URL: "http://x.test/", Body: ""})
	if a.Title != "" {
		t.Errorf("empty body title = %q, want empty", a.Title)
	}
	if got := a.Features[feature.FeatContentQuality]; got != 0 {
		t.Errorf("empty body quality = %v, want 0", got)
	}
}

func TestStaticRankRewardsContentPenalizesDepth(t *testing.T) {
	shallow := Document(convert.Document{
		URL:  "https://h.test/page",
		Body: longBody(),
	})
	deep := Document(convert.Document{
		URL:  "https://h.test/a/b/c/d/e/page",
		Body: longBody(),
	})
	if shallow.Features[feature.FeatStaticRank] <= deep.Features[feature.FeatStaticRank] {
		t.Errorf("shallow static rank %v should beat deep %v for the same body",
			shallow.Features[feature.FeatStaticRank], deep.Features[feature.FeatStaticRank])
	}
}

func longBody() string {
	const word = "content "
	out := make([]byte, 0, len(word)*200)
	for i := 0; i < 200; i++ {
		out = append(out, word...)
	}
	return string(out)
}
