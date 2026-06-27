package vector

import "sort"

// DefaultRRFk is the canon Reciprocal Rank Fusion constant.
const DefaultRRFk = 60

// Fuse merges two ranked docID lists by Reciprocal Rank Fusion. It needs no score
// calibration because it throws away the magnitudes and keeps only the ranks: a
// document's fused score is the sum over the lists it appears in of 1/(k+rank+1),
// so a document both planes rank highly sums two strong contributions and rises.
// This is how dense recall enters the cascade, as a rank-based vote, never as a
// raw score the later stages must trust on its own scale.
func Fuse(lexical, dense []uint32, k int) []uint32 {
	if k <= 0 {
		k = DefaultRRFk
	}
	score := map[uint32]float64{}
	for rank, d := range lexical {
		score[d] += 1.0 / float64(k+rank+1)
	}
	for rank, d := range dense {
		score[d] += 1.0 / float64(k+rank+1)
	}
	out := make([]uint32, 0, len(score))
	for d := range score {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool {
		if score[out[i]] != score[out[j]] {
			return score[out[i]] > score[out[j]]
		}
		return out[i] < out[j]
	})
	return out
}
