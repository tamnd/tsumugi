package spell

// Action is what the corrector decided to do with a query term.
type Action uint8

const (
	// None leaves the term as typed: it is a real word the corpus has, or no
	// correction was within range.
	None Action = iota
	// DidYouMean offers the correction as a suggestion without changing the results,
	// the conservative default for an interactive search box.
	DidYouMean
	// AutoCorrect substitutes the correction and returns its results directly, the
	// aggressive mode reserved for a high-confidence correction of a hard miss.
	AutoCorrect
)

// Policy holds the thresholds that decide when to correct and how aggressively, the
// "when to correct" and "did-you-mean versus auto-correct" rules doc 10 pins. The
// deployment tunes these; DefaultPolicy is a sensible web-search starting point.
type Policy struct {
	// MaxDist caps the distance of an accepted correction; it is at most the index's
	// build-time maximum edit distance.
	MaxDist int
	// LowFreq is the document-frequency floor below which a term present in the
	// dictionary is still treated as a possible rare misspelling and the corrector
	// runs for it; a term at or above the floor is used as typed, no correction.
	LowFreq uint64
	// AutoMaxDist is the largest distance an auto-correct may have; beyond it the
	// correction is offered as a suggestion rather than applied. A distance-one
	// correction is the high-confidence case the canon describes.
	AutoMaxDist int
	// AutoFreqRatio is how many times more frequent the correction must be than the
	// original for an auto-correct; a much more common word is the near-certain intent.
	// When the original is absent (a hard miss) the ratio is satisfied by definition.
	AutoFreqRatio uint64
}

// DefaultPolicy is the web-search default: correct only true misses and very rare
// terms, auto-correct only a distance-one correction to a far more frequent term, and
// offer everything else as did-you-mean.
func DefaultPolicy() Policy {
	return Policy{
		MaxDist:       2,
		LowFreq:       2,
		AutoMaxDist:   1,
		AutoFreqRatio: 10,
	}
}

// Decision is the corrector's verdict for one query term: the action and, when the
// action is not None, the suggestion to offer or apply.
type Decision struct {
	Action     Action
	Suggestion Suggestion
}

// Correct runs the correction policy for one query term. It first checks the term as
// typed: a term present with a frequency at or above the policy's floor is used as-is,
// because the user spelled a real word the corpus has and second-guessing it is wrong,
// so correction stays off the path for the common correctly-spelled query. Only a true
// miss or a suspiciously rare term runs the SymSpell lookup, and the verdict is
// auto-correct when the correction is close and far more frequent than a hard-miss
// original, did-you-mean otherwise.
func (ix *Index) Correct(term string, p Policy) Decision {
	maxd := p.MaxDist
	if maxd > ix.maxEdit {
		maxd = ix.maxEdit
	}
	origFreq, present := ix.Freq(term)
	if present && origFreq > p.LowFreq {
		return Decision{Action: None}
	}
	var best Suggestion
	found := false
	for _, s := range ix.Lookup(term) {
		if s.Dist > maxd {
			continue
		}
		if s.Term == term {
			// The term itself surfaced as its own zero-distance candidate; it is the
			// as-typed case already handled, not a correction to a different term.
			continue
		}
		best = s
		found = true
		break
	}
	if !found {
		return Decision{Action: None}
	}
	if autoCorrects(origFreq, present, best, p) {
		return Decision{Action: AutoCorrect, Suggestion: best}
	}
	return Decision{Action: DidYouMean, Suggestion: best}
}

// autoCorrects decides whether a correction is confident enough to apply silently. The
// correction must be within the auto distance, and either the original is a hard miss
// (absent from the dictionary entirely) or the correction is far more frequent than the
// original by the policy's ratio.
func autoCorrects(origFreq uint64, present bool, best Suggestion, p Policy) bool {
	if best.Dist > p.AutoMaxDist {
		return false
	}
	if !present {
		return true
	}
	return best.Freq >= origFreq*p.AutoFreqRatio
}
