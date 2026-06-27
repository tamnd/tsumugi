package spell

// QueryCorrector adapts an Index and a correction Policy to the query package's
// correction hook, so the broker can run correction over a parsed query without the
// spell package depending on the query package. Its Correct method returns the
// primitives the query.Corrector interface declares, so a QueryCorrector value
// satisfies that interface structurally and is passed straight to
// ParsedQuery.ApplyCorrection.
type QueryCorrector struct {
	Index  *Index
	Policy Policy
}

// NewQueryCorrector builds a QueryCorrector over an index with the default policy.
func NewQueryCorrector(ix *Index) QueryCorrector {
	return QueryCorrector{Index: ix, Policy: DefaultPolicy()}
}

// Correct runs the policy for one query term: ok is false to leave the term as typed,
// auto is true to substitute the replacement, and false to only offer it as a
// did-you-mean.
func (q QueryCorrector) Correct(term string) (replacement string, auto bool, ok bool) {
	if q.Index == nil {
		return "", false, false
	}
	d := q.Index.Correct(term, q.Policy)
	switch d.Action {
	case AutoCorrect:
		return d.Suggestion.Term, true, true
	case DidYouMean:
		return d.Suggestion.Term, false, true
	default:
		return "", false, false
	}
}
