package query_test

import (
	"reflect"
	"testing"

	"github.com/tamnd/tsumugi/expand"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
)

// fakeExpander maps a term to fixed alternatives, so the ApplyExpansion wiring is tested
// without the curated table.
type fakeExpander map[string][]string

func (f fakeExpander) Expand(term string) []string { return f[term] }

// TestApplyExpansionFillsAlts checks that expansion fills a free term's Alts with the
// table's alternatives, keeps the original term, and rewrites the cache key so the
// expanded query caches distinctly from the bare one.
func TestApplyExpansionFillsAlts(t *testing.T) {
	pq := query.Parse("nyc weather", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyExpansion(fakeExpander{"nyc": {"new york city"}})

	var nyc *query.QueryTerm
	for i := range pq.Terms {
		if pq.Terms[i].Term == "nyc" {
			nyc = &pq.Terms[i]
		}
	}
	if nyc == nil {
		t.Fatal("nyc term missing after expansion")
	}
	if !reflect.DeepEqual(nyc.Alts, []string{"new york city"}) {
		t.Errorf("nyc.Alts = %v, want [new york city]", nyc.Alts)
	}
	if !contains(pq.LexicalTerms(), "nyc") {
		t.Error("expansion dropped the original term")
	}
	if pq.NormKey == before {
		t.Error("cache key was not recomputed after expansion added alternatives")
	}
}

// TestApplyExpansionRequiredTerms checks that a required (+term) is expanded too, while
// an excluded (-term) is left alone, because broadening a must-not would drop documents
// the user did not ask to exclude.
func TestApplyExpansionRequiredTerms(t *testing.T) {
	pq := query.Parse("+js -color guide", def, query.SoftOR)
	pq.ApplyExpansion(fakeExpander{
		"js":    {"javascript"},
		"color": {"colour"},
	})
	if len(pq.Required) != 1 || !reflect.DeepEqual(pq.Required[0].Alts, []string{"javascript"}) {
		t.Errorf("required js not expanded: %+v", pq.Required)
	}
	for _, e := range pq.Excluded {
		if len(e.Alts) != 0 {
			t.Errorf("excluded term was expanded: %+v", e)
		}
	}
}

// TestApplyExpansionIdempotent checks that running expansion twice does not duplicate the
// alternatives and that an alternative equal to the term is never listed.
func TestApplyExpansionIdempotent(t *testing.T) {
	pq := query.Parse("color", def, query.SoftOR)
	e := fakeExpander{"color": {"colour", "color"}} // includes the term itself
	pq.ApplyExpansion(e)
	pq.ApplyExpansion(e)
	if !reflect.DeepEqual(pq.Terms[0].Alts, []string{"colour"}) {
		t.Errorf("Alts = %v, want [colour] (idempotent, no self-alt)", pq.Terms[0].Alts)
	}
}

// TestApplyExpansionNilIsNoop checks the broker can call ApplyExpansion with no table
// configured and nothing changes.
func TestApplyExpansionNilIsNoop(t *testing.T) {
	pq := query.Parse("rust borrow checker", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyExpansion(nil)
	if pq.NormKey != before {
		t.Error("nil expander changed the cache key")
	}
	for _, term := range pq.Terms {
		if len(term.Alts) != 0 {
			t.Error("nil expander filled an Alts slice")
		}
	}
}

// TestApplyExpansionWithRealTable wires the real curated table into the query pipeline
// and proves an acronym and a spelling variant expand end to end.
func TestApplyExpansionWithRealTable(t *testing.T) {
	tab := expand.Default(lexical.Analyze)
	pq := query.Parse("nyc gray pizza", def, query.SoftOR)
	pq.ApplyExpansion(tab)

	want := map[string][]string{
		"nyc":   {"new york city"},
		"gray":  {"grey"},
		"pizza": nil,
	}
	for _, term := range pq.Terms {
		if exp, ok := want[term.Term]; ok {
			if !reflect.DeepEqual(term.Alts, exp) {
				t.Errorf("%q.Alts = %v, want %v", term.Term, term.Alts, exp)
			}
		}
	}
}
