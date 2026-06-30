package eval

import (
	"fmt"
	"testing"
)

// buildLadderFixture makes a query set where each query has one perfectly-relevant
// document graded three, and two runs: a baseline that ranks the relevant document at a
// middling position and an improved run that ranks it first. The improved run must beat
// the baseline on NDCG@10 by a clear, per-query-consistent margin, the shape a real
// ladder step has.
func buildLadderFixture(nQueries int) (qrels Qrels, baseline, improved Run) {
	qrels = Qrels{}
	baseline = Run{}
	improved = Run{}
	for i := 0; i < nQueries; i++ {
		q := fmt.Sprintf("q%d", i)
		rel := fmt.Sprintf("rel%d", i)
		qrels[q] = map[string]float64{rel: 3}
		// Baseline: ten documents, the relevant one fifth by score.
		var b []RankedDoc
		for r := 0; r < 10; r++ {
			doc := fmt.Sprintf("d%d_%d", i, r)
			score := float64(10 - r)
			if r == 4 {
				doc = rel
			}
			b = append(b, RankedDoc{Doc: doc, Score: score})
		}
		baseline[q] = b
		// Improved: the relevant document first.
		imp := []RankedDoc{{Doc: rel, Score: 100}}
		for r := 0; r < 9; r++ {
			imp = append(imp, RankedDoc{Doc: fmt.Sprintf("d%d_%d", i, r), Score: float64(9 - r)})
		}
		improved[q] = imp
	}
	return qrels, baseline, improved
}

func TestLadderGainAndSignificance(t *testing.T) {
	qrels, baseline, improved := buildLadderFixture(30)
	// Gold is a subset of the queries, reported separately.
	gold := Qrels{}
	for i := 0; i < 8; i++ {
		q := fmt.Sprintf("q%d", i)
		gold[q] = qrels[q]
	}
	rungs := Ladder([]Config{
		{Name: "baseline", Run: baseline},
		{Name: "improved", Run: improved},
	}, qrels, gold, []int{10, 20}, []int{100})

	if len(rungs) != 2 {
		t.Fatalf("got %d rungs, want 2", len(rungs))
	}
	if rungs[0].GainNDCG10 != 0 || rungs[0].Significant {
		t.Errorf("base rung must have no gain over nothing: gain %.4f sig %v", rungs[0].GainNDCG10, rungs[0].Significant)
	}
	if rungs[1].GainNDCG10 <= 0 {
		t.Fatalf("improved rung gain = %.4f, want positive", rungs[1].GainNDCG10)
	}
	if !rungs[1].Significant {
		t.Fatalf("a consistent per-query gain over 30 queries must be significant, p = %.4f", rungs[1].PValue)
	}
	// Bulk is scored over all queries; gold over the subset, reported separately.
	if rungs[1].Bulk.NumQueries != 30 {
		t.Errorf("bulk scored %d queries, want 30", rungs[1].Bulk.NumQueries)
	}
	if rungs[1].Gold.NumQueries != 8 {
		t.Errorf("gold scored %d queries, want the 8-query subset", rungs[1].Gold.NumQueries)
	}
	// The improved rung reaches a perfect NDCG@10 on both bulk and gold.
	if !approx(rungs[1].Bulk.MeanNDCG[10], 1) {
		t.Errorf("improved bulk NDCG@10 = %.4f, want 1.0", rungs[1].Bulk.MeanNDCG[10])
	}
}

func TestLadderFlatStepNotSignificant(t *testing.T) {
	qrels, baseline, _ := buildLadderFixture(30)
	// A second rung identical to the first: no gain, must not be significant.
	rungs := Ladder([]Config{
		{Name: "baseline", Run: baseline},
		{Name: "same", Run: baseline},
	}, qrels, Qrels{}, []int{10}, nil)
	if rungs[1].GainNDCG10 != 0 {
		t.Errorf("identical run gain = %.4f, want 0", rungs[1].GainNDCG10)
	}
	if rungs[1].Significant {
		t.Error("a zero gain must not be flagged significant; a step earns its place by a real gain")
	}
}

func TestPairedSignificanceNoSharedQueries(t *testing.T) {
	a := Evaluate(Run{"q1": {{Doc: "x", Score: 1}}}, Qrels{"q1": {"x": 1}}, []int{10}, nil)
	b := Evaluate(Run{"q2": {{Doc: "y", Score: 1}}}, Qrels{"q2": {"y": 1}}, []int{10}, nil)
	mean, p := PairedSignificance(a, b, 10)
	if mean != 0 || p != 1 {
		t.Fatalf("disjoint query sets should give no gain and p=1, got mean %.4f p %.4f", mean, p)
	}
}
