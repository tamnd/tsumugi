package query_test

import (
	"testing"

	"github.com/tamnd/tsumugi/query"
	"github.com/tamnd/tsumugi/spell"
)

// fakeCorrector maps a misspelling to a fixed correction with a fixed auto flag, so the
// ApplyCorrection wiring is tested without a real index.
type fakeCorrector struct {
	repl map[string]string
	auto map[string]bool
}

func (f fakeCorrector) Correct(term string) (string, bool, bool) {
	r, ok := f.repl[term]
	if !ok {
		return "", false, false
	}
	return r, f.auto[term], true
}

// TestApplyCorrectionAutoSubstitutes checks that an auto-correction replaces the term in
// place, marks the query corrected, and rewrites the cache key to the corrected form so
// the corrected query caches under what it actually retrieves on.
func TestApplyCorrectionAutoSubstitutes(t *testing.T) {
	pq := query.Parse("gloang tutorial", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyCorrection(fakeCorrector{
		repl: map[string]string{"gloang": "golang"},
		auto: map[string]bool{"gloang": true},
	})
	if !pq.Corrected {
		t.Fatal("auto-correction did not mark the query corrected")
	}
	terms := pq.LexicalTerms()
	if !contains(terms, "golang") || contains(terms, "gloang") {
		t.Errorf("terms after auto-correct = %v, want gloang replaced by golang", terms)
	}
	if pq.NormKey == before {
		t.Error("cache key was not recomputed after an auto-correction changed the terms")
	}
}

// TestApplyCorrectionDidYouMeanKeepsTerms checks that a non-auto correction leaves the
// terms as typed and only records the suggestion, the conservative did-you-mean mode.
func TestApplyCorrectionDidYouMeanKeepsTerms(t *testing.T) {
	pq := query.Parse("kubernets guide", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyCorrection(fakeCorrector{
		repl: map[string]string{"kubernets": "kubernetes"},
		auto: map[string]bool{"kubernets": false},
	})
	if pq.Corrected {
		t.Error("did-you-mean must not mark the query corrected")
	}
	if !contains(pq.LexicalTerms(), "kubernets") {
		t.Errorf("did-you-mean changed the terms: %v", pq.LexicalTerms())
	}
	if pq.NormKey != before {
		t.Error("did-you-mean must not change the cache key")
	}
	if pq.Suggestion == "" {
		t.Error("did-you-mean produced no suggestion string")
	}
}

// TestApplyCorrectionNilIsNoop checks the broker can call ApplyCorrection with no
// corrector configured and nothing changes.
func TestApplyCorrectionNilIsNoop(t *testing.T) {
	pq := query.Parse("rust borrow checker", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyCorrection(nil)
	if pq.Corrected || pq.Suggestion != "" || pq.NormKey != before {
		t.Error("nil corrector changed the query")
	}
}

// TestApplyCorrectionWithRealIndex wires the real spell.QueryCorrector into the query
// pipeline and proves a misspelled term auto-corrects to the far more frequent corpus
// term end to end, the integration of the corrector and the parser.
func TestApplyCorrectionWithRealIndex(t *testing.T) {
	ix := spell.Build([]spell.TermFreq{
		{Term: "golang", Freq: 800},
		{Term: "tutorial", Freq: 300},
		{Term: "rust", Freq: 900},
	}, 2)
	corr := spell.NewQueryCorrector(ix)

	pq := query.Parse("gloang tutorial", def, query.SoftOR)
	pq.ApplyCorrection(corr)
	if !pq.Corrected || !contains(pq.LexicalTerms(), "golang") {
		t.Errorf("real corrector did not fix gloang -> golang: corrected=%v terms=%v", pq.Corrected, pq.LexicalTerms())
	}

	// A correctly spelled term is left untouched.
	pq2 := query.Parse("rust", def, query.SoftOR)
	pq2.ApplyCorrection(corr)
	if pq2.Corrected || !contains(pq2.LexicalTerms(), "rust") {
		t.Errorf("real corrector altered a correct term: %v", pq2.LexicalTerms())
	}
}

func contains(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}
