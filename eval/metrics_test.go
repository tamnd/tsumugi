package eval

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// TestDCGHandComputed checks DCG against a value worked by hand: labels 3,2,3,0,1,2
// at the standard gain and discount.
func TestDCGHandComputed(t *testing.T) {
	rel := []float64{3, 2, 3, 0, 1, 2}
	// gains: 7,3,7,0,1,3 ; discounts: 1/log2(2..7)
	var want float64
	gains := []float64{7, 3, 7, 0, 1, 3}
	for i, g := range gains {
		want += g / math.Log2(float64(i+2))
	}
	if got := DCG(rel, 0); !approx(got, want) {
		t.Fatalf("DCG = %v, want %v", got, want)
	}
}

// TestNDCGPerfectAndReversed checks the ideal ranking scores one and that a worse
// order scores less.
func TestNDCGPerfectAndReversed(t *testing.T) {
	labels := []float64{3, 2, 1, 0}
	perfect := []float64{3, 2, 1, 0}
	if got := NDCG(perfect, labels, 0); !approx(got, 1) {
		t.Fatalf("perfect NDCG = %v, want 1", got)
	}
	reversed := []float64{0, 1, 2, 3}
	if got := NDCG(reversed, labels, 0); got >= 1 || got <= 0 {
		t.Fatalf("reversed NDCG = %v, want in (0,1)", got)
	}
}

// TestNDCGMissedRelevant checks a ranking that omits relevant documents is
// penalized through the ideal normalizer.
func TestNDCGMissedRelevant(t *testing.T) {
	allLabels := []float64{3, 3, 2, 1}
	// The ranking only surfaced two of the four relevant docs.
	ranked := []float64{3, 2}
	full := NDCG([]float64{3, 3, 2, 1}, allLabels, 0)
	partial := NDCG(ranked, allLabels, 0)
	if partial >= full {
		t.Fatalf("partial NDCG %v should be below full %v", partial, full)
	}
}

func TestAveragePrecision(t *testing.T) {
	// ranked relevance: rel at positions 1,3,5 (1-based), total 3 relevant.
	ranked := []float64{1, 0, 1, 0, 1}
	// precisions at hits: 1/1, 2/3, 3/5 ; AP = (1 + 0.6667 + 0.6)/3
	want := (1.0 + 2.0/3.0 + 3.0/5.0) / 3.0
	if got := AveragePrecision(ranked, 3); !approx(got, want) {
		t.Fatalf("AP = %v, want %v", got, want)
	}
}

func TestReciprocalRank(t *testing.T) {
	if got := ReciprocalRank([]float64{0, 0, 1, 0}); !approx(got, 1.0/3.0) {
		t.Fatalf("RR = %v, want 1/3", got)
	}
	if got := ReciprocalRank([]float64{0, 0, 0}); got != 0 {
		t.Fatalf("RR with no relevant = %v, want 0", got)
	}
}

func TestRecallAtK(t *testing.T) {
	ranked := []float64{1, 0, 1, 0, 0}
	if got := RecallAtK(ranked, 4, 3); !approx(got, 0.5) {
		t.Fatalf("recall@3 = %v, want 0.5", got)
	}
}

func TestMeanNDCG(t *testing.T) {
	rankings := [][]float64{{3, 2, 1}, {0, 1, 2}}
	labels := [][]float64{{3, 2, 1}, {2, 1, 0}}
	got := MeanNDCG(rankings, labels, 0)
	q0 := NDCG(rankings[0], labels[0], 0)
	q1 := NDCG(rankings[1], labels[1], 0)
	if !approx(got, (q0+q1)/2) {
		t.Fatalf("mean NDCG = %v, want %v", got, (q0+q1)/2)
	}
}
