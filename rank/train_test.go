package rank

import (
	"bytes"
	"math"
	"testing"

	"github.com/tamnd/tsumugi/eval"
)

// lcg is a small deterministic generator so the synthetic data is reproducible
// without depending on package math/rand defaults across versions.
type lcg uint64

func (s *lcg) next() float64 {
	*s = *s*6364136223846793005 + 1442695040888963407
	return float64(*s>>11) / float64(1<<53)
}

// makeRankData builds a synthetic learning-to-rank set where the true relevance
// is a nonlinear function of two of the features plus noise, and the remaining
// features are distractors. A good ranker has to recover the two signal features
// to beat scoring by any single feature, which is the point of the gate.
func makeRankData(seed uint64, queries, docsPerQuery, numFeatures int) *Dataset {
	r := lcg(seed)
	d := &Dataset{NumFeatures: numFeatures}
	for q := 0; q < queries; q++ {
		d.Groups = append(d.Groups, docsPerQuery)
		for i := 0; i < docsPerQuery; i++ {
			row := make([]float64, numFeatures)
			for f := range row {
				row[f] = r.next()
			}
			// Relevance rises with feature 0 and with the product of 0 and 1, so
			// neither feature alone orders documents correctly.
			signal := row[0]*0.6 + row[0]*row[1]*0.8 + (r.next()-0.5)*0.1
			var label float64
			switch {
			case signal > 0.85:
				label = 4
			case signal > 0.65:
				label = 3
			case signal > 0.45:
				label = 2
			case signal > 0.25:
				label = 1
			default:
				label = 0
			}
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, label)
		}
	}
	return d
}

// rankedLabels orders one query's labels by a scoring function and returns the
// labels in that order, the input NDCG wants.
func rankedLabels(d *Dataset, rows, size int, score func(row []float64) float64) []float64 {
	type sl struct {
		s float64
		l float64
	}
	items := make([]sl, size)
	for i := 0; i < size; i++ {
		items[i] = sl{score(d.Features[rows+i]), d.Labels[rows+i]}
	}
	// insertion sort by score descending, stable enough for the test sizes.
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].s > items[j-1].s; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
	out := make([]float64, size)
	for i := range items {
		out[i] = items[i].l
	}
	return out
}

// meanNDCG scores a dataset under a per-document scoring function at cutoff k.
func meanNDCG(d *Dataset, k int, score func(row []float64) float64) float64 {
	var rankings, labels [][]float64
	base := 0
	for _, size := range d.Groups {
		rankings = append(rankings, rankedLabels(d, base, size, score))
		ls := make([]float64, size)
		copy(ls, d.Labels[base:base+size])
		labels = append(labels, ls)
		base += size
	}
	return eval.MeanNDCG(rankings, labels, k)
}

// TestTrainBeatsBaseline is the M9 gate: a model trained on a train split must
// rank a held-out split better than the best single raw feature and clear an
// absolute NDCG floor. It proves the trainer recovers signal the cascade can
// then serve through the same QuickScorer evaluator.
func TestTrainBeatsBaseline(t *testing.T) {
	const numFeatures = 8
	train := makeRankData(1, 200, 12, numFeatures)
	test := makeRankData(2, 80, 12, numFeatures)

	model := Train(train, DefaultParams()).Compile()
	trained := meanNDCG(test, 10, model.Score)

	// Best baseline among raw single features.
	best := 0.0
	for f := 0; f < numFeatures; f++ {
		f := f
		s := meanNDCG(test, 10, func(row []float64) float64 { return row[f] })
		if s > best {
			best = s
		}
	}

	t.Logf("trained NDCG@10 = %.4f, best single-feature = %.4f", trained, best)
	if trained <= best+0.02 {
		t.Fatalf("trained NDCG %.4f did not beat baseline %.4f by margin", trained, best)
	}
	if trained < 0.9 {
		t.Fatalf("trained NDCG %.4f below absolute floor 0.9", trained)
	}
}

// TestEnsembleSaveLoad round-trips a trained model and checks the loaded copy
// compiles to a scorer that agrees bit for bit with the original.
func TestEnsembleSaveLoad(t *testing.T) {
	d := makeRankData(3, 50, 10, 6)
	ens := Train(d, Params{Rounds: 40, LearningRate: 0.1, MaxDepth: 5, MinSamplesLeaf: 5, LambdaL2: 1, NDCGCutoff: 10})

	var buf bytes.Buffer
	if err := ens.Save(&buf); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadEnsemble(&buf)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.NumTrees() != ens.NumTrees() {
		t.Fatalf("tree count %d != %d", loaded.NumTrees(), ens.NumTrees())
	}
	a := ens.Compile()
	b := loaded.Compile()
	for i := 0; i < len(d.Features); i++ {
		sa := a.Score(d.Features[i])
		sb := b.Score(d.Features[i])
		if math.Float64bits(sa) != math.Float64bits(sb) {
			t.Fatalf("row %d: scores differ after round trip: %v vs %v", i, sa, sb)
		}
	}
}

// TestTrainReducesPairwiseLoss is a sanity check that boosting actually fits:
// training NDCG should rise from the untrained constant model.
func TestTrainReducesPairwiseLoss(t *testing.T) {
	d := makeRankData(4, 100, 12, 6)
	zero := meanNDCG(d, 10, func(row []float64) float64 { return 0 })
	model := Train(d, DefaultParams()).Compile()
	trained := meanNDCG(d, 10, model.Score)
	if trained <= zero {
		t.Fatalf("training did not improve NDCG: %.4f vs zero %.4f", trained, zero)
	}
}
