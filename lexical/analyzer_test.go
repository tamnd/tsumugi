package lexical

import (
	"reflect"
	"testing"
)

// TestDefaultAnalyzerMatchesLegacy is the safety gate: the default analyzer must
// produce exactly the lowercase-and-tokenize output the index has always indexed,
// so adding the configurable chain leaves every existing shard byte-identical.
func TestDefaultAnalyzerMatchesLegacy(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello, World!", []string{"hello", "world"}},
		{"state-of-the-art", []string{"state", "of", "the", "art"}},
		{"iPhone 15 Pro", []string{"iphone", "15", "pro"}},
		{"  spaced   out  ", []string{"spaced", "out"}},
		{"café", []string{"café"}},
		{"", nil},
		{"!!!", nil},
	}
	for _, c := range cases {
		if got := Analyze(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Analyze(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestAnalyzerNFKC checks NFKC folds the compatibility variants web text is full of,
// full-width letters and ligatures, to their plain form so the query and the index
// see one byte sequence.
func TestAnalyzerNFKC(t *testing.T) {
	a := &Analyzer{NFKC: true}
	// Full-width "ＡＢＣ" must fold to "abc" after lowercasing.
	if got := a.Analyze("ＡＢＣ"); !reflect.DeepEqual(got, []string{"abc"}) {
		t.Errorf("full-width = %v, want [abc]", got)
	}
	// The "ﬁ" ligature folds to "fi", so "ﬁle" becomes "file".
	if got := a.Analyze("ﬁle"); !reflect.DeepEqual(got, []string{"file"}) {
		t.Errorf("ligature = %v, want [file]", got)
	}
	// Without NFKC the default leaves the ligature alone, the byte difference NFKC fixes.
	if got := Analyze("ﬁle"); reflect.DeepEqual(got, []string{"file"}) {
		t.Errorf("default analyzer should not fold the ligature, got %v", got)
	}
}

// TestAnalyzerFoldAccents checks the accent fold maps an accented token to its plain
// form so an accent-careless query matches an accented document.
func TestAnalyzerFoldAccents(t *testing.T) {
	a := &Analyzer{NFKC: true, FoldAccents: true}
	got := a.Analyze("Café Müller naïve")
	want := []string{"cafe", "muller", "naive"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("fold = %v, want %v", got, want)
	}
}

// TestAnalyzerStem checks the English stemmer bridges inflected forms on both sides,
// which is how "running" finds "run" in a BM25F shard with no learned plane.
func TestAnalyzerStem(t *testing.T) {
	a := &Analyzer{Stemmer: "english"}
	got := a.Analyze("running runs ran the universities")
	want := []string{"run", "run", "ran", "the", "univers"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("stem = %v, want %v", got, want)
	}
}

// TestAnalyzerStopwords checks the optional drop list removes the configured tokens
// and keeps the rest, the spec's keep-by-default with an optional drop set.
func TestAnalyzerStopwords(t *testing.T) {
	a := &Analyzer{Stopwords: map[string]struct{}{"the": {}, "of": {}}}
	got := a.Analyze("the king of the hill")
	want := []string{"king", "hill"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("stopwords = %v, want %v", got, want)
	}
}

// TestAnalyzerHash is the consistency-guard gate: two analyzers hash equal exactly
// when they analyze identically, so the broker's one-number check catches a query
// analyzer that does not match the index analyzer.
func TestAnalyzerHash(t *testing.T) {
	base := &Analyzer{}
	same := &Analyzer{}
	if base.Hash() != same.Hash() {
		t.Fatal("identical configs must hash equal")
	}
	variants := []*Analyzer{
		{NFKC: true},
		{FoldAccents: true},
		{Stemmer: "english"},
		{Stopwords: map[string]struct{}{"the": {}}},
	}
	for i, v := range variants {
		if v.Hash() == base.Hash() {
			t.Errorf("variant %d must differ from the default config", i)
		}
	}
	// The stopword set order must not change the hash, so two drop lists with the
	// same words in a different insertion order hash equal.
	a := &Analyzer{Stopwords: map[string]struct{}{"a": {}, "b": {}}}
	b := &Analyzer{Stopwords: map[string]struct{}{"b": {}, "a": {}}}
	if a.Hash() != b.Hash() {
		t.Error("stopword set order must not change the hash")
	}
}

// TestAnalyzerBothSidesAgree is the contract in miniature: the same analyzer over a
// document field and over a query produces matching tokens, so the dictionary lookup
// compares like with like even with the full chain on.
func TestAnalyzerBothSidesAgree(t *testing.T) {
	a := &Analyzer{NFKC: true, FoldAccents: true, Stemmer: "english"}
	doc := a.Analyze("The Cafés are Running fast")
	query := a.Analyze("café running")
	docSet := map[string]bool{}
	for _, t := range doc {
		docSet[t] = true
	}
	for _, q := range query {
		if !docSet[q] {
			t.Errorf("query term %q absent from doc tokens %v, analyzer chain disagreed across sides", q, doc)
		}
	}
}
