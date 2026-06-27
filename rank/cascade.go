package rank

import "sort"

// Candidate is a document moving through the cascade: its dense docID and the
// score of the stage that produced it. The score is restated at each stage
// boundary, never compared across stages, since the L0 retrieval score, the L1
// linear score, and the L2 ordinal score are not on the same scale.
type Candidate struct {
	DocID uint32
	Score float64
}

// FeatureFunc returns the precomputed feature row for a document, the columns the
// L1 cut and the L2 model read. It abstracts the forward feature matrix so the
// cascade does not depend on the storage region.
type FeatureFunc func(docID uint32) []float64

// DefaultRRFk is the canon Reciprocal Rank Fusion constant.
const DefaultRRFk = 60

// Defaults are the canon cascade sizes: L0 retrieves about a thousand candidates,
// L1 cuts to about two hundred, L2 reranks to the final top-k.
const (
	DefaultL0Max  = 1000
	DefaultL1Keep = 200
)

// RRF merges ranked docID lists by Reciprocal Rank Fusion: a document's fused
// score is the sum over the lists it appears in of 1/(k+rank), with rank counted
// from one. It is order independent and needs no score calibration across planes,
// which is why it fuses the lexical and dense retrieval lists that carry
// incomparable scores. The result is sorted by fused score descending, ties to the
// smaller docID.
func RRF(k int, lists ...[]uint32) []Candidate {
	if k <= 0 {
		k = DefaultRRFk
	}
	score := map[uint32]float64{}
	for _, list := range lists {
		for rank, doc := range list {
			score[doc] += 1.0 / float64(k+rank+1)
		}
	}
	out := make([]Candidate, 0, len(score))
	for doc, s := range score {
		out = append(out, Candidate{DocID: doc, Score: s})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DocID < out[j].DocID
	})
	return out
}

// Cascade holds the ranking configuration: the stage sizes, the L1 linear cut, and
// the compiled L2 model. The sizes are fields rather than constants so the broker
// can shrink them under a tight deadline, degrading gracefully by ranking a
// smaller set precisely rather than skipping a stage.
type Cascade struct {
	L0Max  int
	L1Keep int
	K      int // RRF constant
	L1     *Linear
	L2     *Model
}

// NewCascade returns a cascade with the canon sizes and the given L1 cut and L2
// model.
func NewCascade(l1 *Linear, l2 *Model) *Cascade {
	return &Cascade{
		L0Max:  DefaultL0Max,
		L1Keep: DefaultL1Keep,
		K:      DefaultRRFk,
		L1:     l1,
		L2:     l2,
	}
}

// Rank runs the full cascade: fuse the lexical and dense plane rankings, cap the
// fused set to L0Max, cut it to L1Keep with the linear score, rerank the survivors
// with the L2 model, and return the final top-k by the model's ordinal score. The
// retrieval rank carried by the fusion is the retrieval-score feature the linear
// cut reads, so a strong dual-plane document survives the cut.
func (c *Cascade) Rank(lexical, dense []uint32, feat FeatureFunc, k int) []Candidate {
	fused := RRF(c.K, lexical, dense)
	if len(fused) > c.L0Max {
		fused = fused[:c.L0Max]
	}
	kept := c.L1.Cut(fused, feat, c.L1Keep)
	return c.rerank(kept, feat, k)
}

// rerank scores each survivor with the L2 model and returns the top-k by the
// ordinal score, ties to the smaller docID.
func (c *Cascade) rerank(cands []Candidate, feat FeatureFunc, k int) []Candidate {
	scored := make([]Candidate, len(cands))
	for i, cd := range cands {
		scored[i] = Candidate{DocID: cd.DocID, Score: c.L2.Score(feat(cd.DocID))}
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].DocID < scored[j].DocID
	})
	if k > 0 && len(scored) > k {
		scored = scored[:k]
	}
	return scored
}
