package rank

import "sort"

// Linear is the L1 cut: a linear score over a fixed subset of feature columns plus
// the carried retrieval rank, with no trees and no branching. It is a cheap recall
// filter, fit offline to agree with L2 on the coarse question of which documents
// are worth the model's time, not on their exact order, so it cuts a document only
// when L2 would have ranked it low anyway. Cols selects the feature columns it
// reads and Weights are their coefficients; RetrievalWeight scales the candidate's
// carried fusion score, which is the retrieval signal L1 keeps as an input.
type Linear struct {
	Cols            []int
	Weights         []float64
	RetrievalWeight float64
}

// score is the linear combination for one candidate: the retrieval term plus the
// weighted sum of the selected feature columns.
func (l *Linear) score(doc []float64, retrieval float64) float64 {
	s := l.RetrievalWeight * retrieval
	for i, c := range l.Cols {
		s += l.Weights[i] * doc[c]
	}
	return s
}

// Cut scores every candidate with the linear combination and returns the top keep
// by that score, ties to the smaller docID. Each candidate's incoming Score is the
// retrieval signal; the returned candidates carry the L1 score. keep at or below
// zero, or larger than the input, returns the whole set ordered.
func (l *Linear) Cut(cands []Candidate, feat FeatureFunc, keep int) []Candidate {
	out := make([]Candidate, len(cands))
	for i, cd := range cands {
		out[i] = Candidate{DocID: cd.DocID, Score: l.score(feat(cd.DocID), cd.Score)}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DocID < out[j].DocID
	})
	if keep > 0 && len(out) > keep {
		out = out[:keep]
	}
	return out
}
