package query_test

import (
	"reflect"
	"testing"

	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
)

// def is the analyzer the parser tests run with: the default lexical analyzer, the one
// byte-for-byte identical to the index's own tokenizer, so a parsed term is exactly the
// dictionary term it will look up. A few tests build their own configured analyzer to
// exercise stemming and stopwords through the parser.
var def = lexical.DefaultAnalyzer

// termStrs projects the term field out of a query-term slice, the shape most of the
// assertions compare against.
func termStrs(ts []query.QueryTerm) []string {
	out := make([]string, len(ts))
	for i, t := range ts {
		out[i] = t.Term
	}
	return out
}

func TestParseFreeText(t *testing.T) {
	pq := query.Parse("best coffee shops", def, query.SoftOR)
	if got := pq.LexicalTerms(); !reflect.DeepEqual(got, []string{"best", "coffee", "shops"}) {
		t.Fatalf("lexical terms %v", got)
	}
	if len(pq.Required) != 0 || len(pq.Excluded) != 0 || len(pq.Phrases) != 0 || len(pq.Filters) != 0 {
		t.Fatalf("free query carried operators: %+v", pq)
	}
	if pq.Mode != query.SoftOR {
		t.Fatalf("mode %v", pq.Mode)
	}
}

func TestParseRequiredAndExcluded(t *testing.T) {
	pq := query.Parse("coffee +espresso -decaf", def, query.SoftOR)
	if got := termStrs(pq.Terms); !reflect.DeepEqual(got, []string{"coffee"}) {
		t.Fatalf("free terms %v", got)
	}
	if got := termStrs(pq.Required); !reflect.DeepEqual(got, []string{"espresso"}) {
		t.Fatalf("required %v", got)
	}
	if got := termStrs(pq.Excluded); !reflect.DeepEqual(got, []string{"decaf"}) {
		t.Fatalf("excluded %v", got)
	}
	// LexicalTerms is free + required, never the excluded set.
	if got := pq.LexicalTerms(); !reflect.DeepEqual(got, []string{"coffee", "espresso"}) {
		t.Fatalf("lexical terms %v", got)
	}
}

func TestParsePhrase(t *testing.T) {
	pq := query.Parse(`"new york times" subscription`, def, query.SoftOR)
	if len(pq.Phrases) != 1 {
		t.Fatalf("phrases %+v", pq.Phrases)
	}
	if got := pq.Phrases[0].Terms; !reflect.DeepEqual(got, []string{"new", "york", "times"}) {
		t.Fatalf("phrase terms %v", got)
	}
	if pq.Phrases[0].Slop != 0 {
		t.Fatalf("strict phrase should have slop 0, got %d", pq.Phrases[0].Slop)
	}
	if got := termStrs(pq.Terms); !reflect.DeepEqual(got, []string{"subscription"}) {
		t.Fatalf("free terms %v", got)
	}
}

func TestParseUnclosedQuote(t *testing.T) {
	// The forgiving parser reads an unclosed quote to the end of the string as a phrase
	// rather than erroring.
	pq := query.Parse(`apple "fresh fruit`, def, query.SoftOR)
	if len(pq.Phrases) != 1 || !reflect.DeepEqual(pq.Phrases[0].Terms, []string{"fresh", "fruit"}) {
		t.Fatalf("unclosed quote phrases %+v", pq.Phrases)
	}
	if got := termStrs(pq.Terms); !reflect.DeepEqual(got, []string{"apple"}) {
		t.Fatalf("free terms %v", got)
	}
}

func TestParseHostFilter(t *testing.T) {
	pq := query.Parse("rust generics site:Doc.Rust-Lang.org", def, query.SoftOR)
	if len(pq.Filters) != 1 || pq.Filters[0].Kind != query.FilterHost {
		t.Fatalf("filters %+v", pq.Filters)
	}
	if pq.Filters[0].Value != "doc.rust-lang.org" {
		t.Fatalf("host value %q", pq.Filters[0].Value)
	}
	if got := pq.LexicalTerms(); !reflect.DeepEqual(got, []string{"rust", "generics"}) {
		t.Fatalf("lexical terms %v", got)
	}
}

func TestParseHostFilterStripsSchemeAndWww(t *testing.T) {
	for _, raw := range []string{"site:https://www.example.com/", "site:www.Example.com", "site:http://example.com"} {
		pq := query.Parse(raw, def, query.SoftOR)
		if len(pq.Filters) != 1 || pq.Filters[0].Value != "example.com" {
			t.Fatalf("%q gave filters %+v", raw, pq.Filters)
		}
	}
}

func TestParseFieldScope(t *testing.T) {
	pq := query.Parse("title:golang body:concurrency", def, query.SoftOR)
	if len(pq.Terms) != 2 {
		t.Fatalf("terms %+v", pq.Terms)
	}
	if pq.Terms[0].Term != "golang" || pq.Terms[0].Field != 0 {
		t.Fatalf("title term %+v", pq.Terms[0])
	}
	if pq.Terms[1].Term != "concurrency" || pq.Terms[1].Field != 1 {
		t.Fatalf("body term %+v", pq.Terms[1])
	}
}

func TestParseFieldScopedRequired(t *testing.T) {
	// A +title:x keeps the required intent and the field scope is not lost for retrieval.
	pq := query.Parse("+title:rust", def, query.SoftOR)
	if len(pq.Required) != 1 || pq.Required[0].Term != "rust" {
		t.Fatalf("required %+v", pq.Required)
	}
}

func TestParseUnknownFieldIsFreeText(t *testing.T) {
	// foo: is not a known field, so the whole token falls back to free text including
	// the colon, the forgiving-parser rule.
	pq := query.Parse("foo:bar", def, query.SoftOR)
	if got := termStrs(pq.Terms); !reflect.DeepEqual(got, []string{"foo", "bar"}) {
		t.Fatalf("unknown field terms %v", got)
	}
	if len(pq.Filters) != 0 {
		t.Fatalf("unknown field made a filter %+v", pq.Filters)
	}
}

func TestParseLoneOperatorsDropped(t *testing.T) {
	// A bare + or - with no term attached is dropped rather than erroring.
	pq := query.Parse("coffee + - shop", def, query.SoftOR)
	if got := pq.LexicalTerms(); !reflect.DeepEqual(got, []string{"coffee", "shop"}) {
		t.Fatalf("lone operators leaked %v", got)
	}
	if len(pq.Required) != 0 || len(pq.Excluded) != 0 {
		t.Fatalf("lone operators became terms: %+v", pq)
	}
}

func TestParseEmpty(t *testing.T) {
	for _, raw := range []string{"", "   ", "\t\n"} {
		pq := query.Parse(raw, def, query.SoftOR)
		if !pq.Empty() {
			t.Fatalf("%q not empty: %+v", raw, pq)
		}
	}
	// A query that carries only a filter has nothing to retrieve on and is empty too.
	pq := query.Parse("site:example.com", def, query.SoftOR)
	if !pq.Empty() {
		t.Fatalf("filter-only query not empty: %+v", pq)
	}
}

func TestParseHardAND(t *testing.T) {
	pq := query.Parse("a b c", def, query.HardAND)
	if pq.Mode != query.HardAND {
		t.Fatalf("mode %v", pq.Mode)
	}
}

func TestLexicalTermsDedup(t *testing.T) {
	// A term that appears free and required collapses to one lexical lookup, first-seen
	// order preserved.
	pq := query.Parse("rust +rust generics", def, query.SoftOR)
	if got := pq.LexicalTerms(); !reflect.DeepEqual(got, []string{"rust", "generics"}) {
		t.Fatalf("dedup lexical terms %v", got)
	}
}

func TestNormKeyStableAcrossCasingAndSpacing(t *testing.T) {
	a := query.Parse("Best   Coffee", def, query.SoftOR)
	b := query.Parse("best coffee", def, query.SoftOR)
	if a.NormKey != b.NormKey {
		t.Fatalf("casing/spacing changed key: %q vs %q", a.NormKey, b.NormKey)
	}
}

func TestNormKeyOrderIndependentForFreeTerms(t *testing.T) {
	// Soft-OR free terms are order-independent, so reordering them collides on one key.
	a := query.Parse("coffee shop best", def, query.SoftOR)
	b := query.Parse("best shop coffee", def, query.SoftOR)
	if a.NormKey != b.NormKey {
		t.Fatalf("free-term order changed key: %q vs %q", a.NormKey, b.NormKey)
	}
}

func TestNormKeyPhraseOrderMatters(t *testing.T) {
	// A phrase is ordered, so two different phrases must not collide.
	a := query.Parse(`"new york"`, def, query.SoftOR)
	b := query.Parse(`"york new"`, def, query.SoftOR)
	if a.NormKey == b.NormKey {
		t.Fatalf("distinct phrases collided on key %q", a.NormKey)
	}
}

func TestNormKeyModeMatters(t *testing.T) {
	a := query.Parse("a b", def, query.SoftOR)
	b := query.Parse("a b", def, query.HardAND)
	if a.NormKey == b.NormKey {
		t.Fatalf("mode did not change key %q", a.NormKey)
	}
}

func TestNormKeyDistinctQueriesDoNotCollide(t *testing.T) {
	a := query.Parse("coffee shop", def, query.SoftOR)
	b := query.Parse("tea shop", def, query.SoftOR)
	if a.NormKey == b.NormKey {
		t.Fatalf("distinct queries collided on key %q", a.NormKey)
	}
}

func TestNormKeyFieldScopeDoesNotCollide(t *testing.T) {
	// A field-scoped term carries its field into the key so title:rust and a plain rust
	// are different cache entries.
	a := query.Parse("title:rust", def, query.SoftOR)
	b := query.Parse("rust", def, query.SoftOR)
	if a.NormKey == b.NormKey {
		t.Fatalf("field scope collided with free term, key %q", a.NormKey)
	}
}

func TestNormKeyFilterOrderIndependent(t *testing.T) {
	a := query.Parse("x site:a.com site:b.com", def, query.SoftOR)
	b := query.Parse("x site:b.com site:a.com", def, query.SoftOR)
	if a.NormKey != b.NormKey {
		t.Fatalf("filter order changed key: %q vs %q", a.NormKey, b.NormKey)
	}
}

// TestParseRunsThroughConfiguredAnalyzer proves the parser does not bypass the analysis
// chain: a stemming, stopword-dropping analyzer reshapes the terms inside operators just
// as it would the free text, so the query side stays consistent with an index built with
// the same analyzer.
func TestParseRunsThroughConfiguredAnalyzer(t *testing.T) {
	a := &lexical.Analyzer{
		Stemmer:   "english",
		Stopwords: map[string]struct{}{"the": {}},
	}
	pq := query.Parse(`the running +jumps "the dogs"`, a, query.SoftOR)
	// "the" is a stopword and drops; "running" stems to "run"; "jumps" to "jump".
	if got := termStrs(pq.Terms); !reflect.DeepEqual(got, []string{"run"}) {
		t.Fatalf("free terms %v", got)
	}
	if got := termStrs(pq.Required); !reflect.DeepEqual(got, []string{"jump"}) {
		t.Fatalf("required %v", got)
	}
	// The phrase keeps order but the stopword drops and "dogs" stems to "dog".
	if got := pq.Phrases[0].Terms; !reflect.DeepEqual(got, []string{"dog"}) {
		t.Fatalf("phrase terms %v", got)
	}
}

// TestParseRealCCrawlQueries runs a spread of realistic web queries through the parser
// and checks the operator structure each one is supposed to produce, the kind of mix a
// real query log carries.
func TestParseRealCCrawlQueries(t *testing.T) {
	cases := []struct {
		raw      string
		terms    []string
		required []string
		excluded []string
		phrases  int
		filters  int
	}{
		{raw: "how to install python on macos", terms: []string{"how", "to", "install", "python", "on", "macos"}},
		{raw: `"machine learning" tutorial -beginner`, terms: []string{"tutorial"}, excluded: []string{"beginner"}, phrases: 1},
		{raw: "climate change site:nature.com", terms: []string{"climate", "change"}, filters: 1},
		{raw: "+react hooks useState", terms: []string{"hooks", "usestate"}, required: []string{"react"}},
		{raw: `title:golang "error handling"`, terms: []string{"golang"}, phrases: 1},
	}
	for _, c := range cases {
		pq := query.Parse(c.raw, def, query.SoftOR)
		if c.terms != nil && !reflect.DeepEqual(termStrs(pq.Terms), c.terms) {
			t.Errorf("%q free terms %v, want %v", c.raw, termStrs(pq.Terms), c.terms)
		}
		if got := termStrs(pq.Required); len(c.required) != 0 && !reflect.DeepEqual(got, c.required) {
			t.Errorf("%q required %v, want %v", c.raw, got, c.required)
		}
		if got := termStrs(pq.Excluded); len(c.excluded) != 0 && !reflect.DeepEqual(got, c.excluded) {
			t.Errorf("%q excluded %v, want %v", c.raw, got, c.excluded)
		}
		if len(pq.Phrases) != c.phrases {
			t.Errorf("%q phrases %d, want %d", c.raw, len(pq.Phrases), c.phrases)
		}
		if len(pq.Filters) != c.filters {
			t.Errorf("%q filters %d, want %d", c.raw, len(pq.Filters), c.filters)
		}
		// NormKey must be reproducible: parsing the same string twice gives the same key.
		if pq.NormKey != query.Parse(c.raw, def, query.SoftOR).NormKey {
			t.Errorf("%q normkey not reproducible", c.raw)
		}
	}
}
