package query

import (
	"testing"

	"github.com/tamnd/tsumugi/lexical"
)

// langSelector is the canonical AnalyzerFor a broker builds: it routes each language to
// the lexical registry's analyzer. The tests use it so ParseDetected exercises the real
// per-language analyzers, not a stand-in.
func langSelector(lang string) Analyzer {
	return lexical.ForLanguage(lang)
}

// fakeDetector is a table-driven Detector so the parser tests pin routing behavior
// without depending on the real n-gram model's exact confidence on short strings.
type fakeDetector struct {
	lang      string
	confident bool
}

func (f fakeDetector) DetectLang(text string) (string, bool) { return f.lang, f.confident }

// A confident detection routes to that language's analyzer and records the language on
// the parsed query. English routing stems, so "running" must reach "run", proving the
// per-language analyzer actually ran rather than the default.
func TestParseDetectedRoutesConfidentLanguage(t *testing.T) {
	pq := ParseDetected("running", langSelector, fakeDetector{"en", true}, SoftOR)
	if pq.Lang != "en" || !pq.LangConfident {
		t.Fatalf("Lang=%q confident=%v, want en/true", pq.Lang, pq.LangConfident)
	}
	if len(pq.Terms) != 1 || pq.Terms[0].Term != "run" {
		t.Fatalf("terms = %+v, want the english stem run", pq.Terms)
	}
}

// A low-confidence detection records the guessed language but routes on the default
// analyzer, the spec's rule that a weak guess must not drive a per-language stem. With
// English routing suppressed, "running" stays unstemmed.
func TestParseDetectedLowConfidenceUsesDefault(t *testing.T) {
	pq := ParseDetected("running", langSelector, fakeDetector{"en", false}, SoftOR)
	if pq.Lang != "en" || pq.LangConfident {
		t.Fatalf("Lang=%q confident=%v, want en/false", pq.Lang, pq.LangConfident)
	}
	if len(pq.Terms) != 1 || pq.Terms[0].Term != "running" {
		t.Fatalf("terms = %+v, want the unstemmed running under the default analyzer", pq.Terms)
	}
}

// A nil detector is safe: no language is detected, the query routes on the default
// analyzer, and the result matches plain Parse. This is the pre-detector-configured
// path a broker takes before it loads the model.
func TestParseDetectedNilDetector(t *testing.T) {
	pq := ParseDetected("running fast", langSelector, nil, SoftOR)
	if pq.Lang != "" || pq.LangConfident {
		t.Fatalf("Lang=%q confident=%v, want empty/false", pq.Lang, pq.LangConfident)
	}
	plain := Parse("running fast", lexical.DefaultAnalyzer, SoftOR)
	if len(pq.Terms) != len(plain.Terms) {
		t.Fatalf("detected %d terms, plain %d, want equal", len(pq.Terms), len(plain.Terms))
	}
	for i := range pq.Terms {
		if pq.Terms[i].Term != plain.Terms[i].Term {
			t.Fatalf("term %d = %q, want %q", i, pq.Terms[i].Term, plain.Terms[i].Term)
		}
	}
}

// The real detector wired end to end must route an English sentence to the English
// analyzer and a CJK query to its language, the integration the broker relies on.
func TestParseDetectedWithRealDetector(t *testing.T) {
	det := lexical.LangDetector()
	en := ParseDetected("the best search engine for the query", langSelector, det, SoftOR)
	if en.Lang != "en" || !en.LangConfident {
		t.Errorf("english sentence Lang=%q confident=%v, want en/true", en.Lang, en.LangConfident)
	}
	zh := ParseDetected("搜索引擎返回最佳结果", langSelector, det, SoftOR)
	if zh.Lang != "zh" || !zh.LangConfident {
		t.Errorf("chinese query Lang=%q confident=%v, want zh/true", zh.Lang, zh.LangConfident)
	}
}

func BenchmarkParseDetectedEnglish(b *testing.B) {
	det := lexical.LangDetector()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ParseDetected("the best search engine for the query", langSelector, det, SoftOR)
	}
}
