package eval

import (
	"math"
	"testing"
)

// TestEvaluatePerfectRanking checks a run already in ideal order scores NDCG 1, MRR
// 1, and full recall, the upper-bound sanity that the join and the normalization
// line up.
func TestEvaluatePerfectRanking(t *testing.T) {
	run := Run{"q1": {{Doc: "d1", Score: 3}, {Doc: "d2", Score: 2}, {Doc: "d3", Score: 1}}}
	qrels := Qrels{"q1": {"d1": 3, "d2": 2, "d3": 1}}
	rep := Evaluate(run, qrels, []int{10}, []int{10})
	if len(rep.PerQuery) != 1 {
		t.Fatalf("scored %d queries, want 1", len(rep.PerQuery))
	}
	approxEq(t, "ndcg@10", rep.PerQuery[0].NDCG[10], 1.0)
	approxEq(t, "mrr", rep.PerQuery[0].MRR, 1.0)
	approxEq(t, "recall@10", rep.PerQuery[0].Recall[10], 1.0)
}

// TestEvaluateClosedFormNDCG pins the gain-and-discount convention against an
// independent closed form: with the only relevant document (grade one) at rank two
// and an unjudged document at rank one, NDCG is the rank-two discount 1/log2(3) over
// the ideal DCG of 1, so the value is 1/log2(3) computed straight from math, not
// from the package's own DCG. This is the check that the gain is 2^g-1 and the
// discount is 1/log2(rank+1) the way trec_eval defines them.
func TestEvaluateClosedFormNDCG(t *testing.T) {
	run := Run{"q": {{Doc: "b", Score: 5}, {Doc: "a", Score: 4}}}
	qrels := Qrels{"q": {"a": 1, "b": 0}}
	rep := Evaluate(run, qrels, []int{10}, []int{10})
	approxEq(t, "ndcg@10", rep.PerQuery[0].NDCG[10], 1/math.Log2(3))
	approxEq(t, "mrr", rep.PerQuery[0].MRR, 0.5)              // first relevant at rank 2
	approxEq(t, "recall@10", rep.PerQuery[0].Recall[10], 1.0) // the one relevant retrieved
}

// TestEvaluateRecallAndMRRCutoffs checks recall is the judged-relevant fraction in
// the top k (unjudged documents in the run counting as irrelevant) and MRR is one
// over the first relevant rank, both at clean fractions.
func TestEvaluateRecallAndMRRCutoffs(t *testing.T) {
	run := Run{"q": {
		{Doc: "z", Score: 9}, {Doc: "u", Score: 8}, // u unjudged -> grade 0
		{Doc: "r1", Score: 7}, {Doc: "r2", Score: 6},
	}}
	qrels := Qrels{"q": {"r1": 1, "r2": 1, "r3": 1, "r4": 1, "z": 0}}
	rep := Evaluate(run, qrels, []int{2}, []int{2, 10})
	q := rep.PerQuery[0]
	if q.NumRel != 4 {
		t.Fatalf("numRel = %d, want 4", q.NumRel)
	}
	approxEq(t, "recall@2", q.Recall[2], 0.0)   // top two are z,u, neither relevant
	approxEq(t, "recall@10", q.Recall[10], 0.5) // r1,r2 of the four relevant
	approxEq(t, "mrr", q.MRR, 1.0/3.0)          // first relevant r1 at rank 3
	approxEq(t, "ndcg@2", q.NDCG[2], 0.0)       // no gain in the top two
}

// TestEvaluateSkipsNoRelevant checks a judged query with no relevant document is
// dropped from the means and counted in NumSkipped, not averaged in as a zero that
// would misrepresent the denominator.
func TestEvaluateSkipsNoRelevant(t *testing.T) {
	run := Run{"q1": {{Doc: "a", Score: 1}}, "q2": {{Doc: "x", Score: 1}}}
	qrels := Qrels{"q1": {"a": 2}, "q2": {"x": 0, "y": 0}}
	rep := Evaluate(run, qrels, []int{10}, []int{10})
	if rep.NumQueries != 1 || rep.NumSkipped != 1 {
		t.Fatalf("NumQueries=%d NumSkipped=%d, want 1 and 1", rep.NumQueries, rep.NumSkipped)
	}
}

// TestEvaluateMissingRunQueryScoresZero checks a judged query the run returned
// nothing for is scored as a zero, the honest penalty for an engine that dropped a
// query, rather than skipped.
func TestEvaluateMissingRunQueryScoresZero(t *testing.T) {
	run := Run{} // engine returned nothing for q
	qrels := Qrels{"q": {"m": 2}}
	rep := Evaluate(run, qrels, []int{10}, []int{10})
	if rep.NumQueries != 1 {
		t.Fatalf("NumQueries=%d, want 1 (scored, not skipped)", rep.NumQueries)
	}
	approxEq(t, "ndcg@10", rep.PerQuery[0].NDCG[10], 0.0)
	approxEq(t, "recall@10", rep.PerQuery[0].Recall[10], 0.0)
	if rep.PerQuery[0].Retrieved != 0 {
		t.Fatalf("Retrieved=%d, want 0", rep.PerQuery[0].Retrieved)
	}
}

// TestEvaluateMeanAveragesScoredQueries checks the means are the average over the
// scored queries and use the doc 14 default cutoffs when none are given.
func TestEvaluateMeanAveragesScoredQueries(t *testing.T) {
	run := Run{
		"q1": {{Doc: "d1", Score: 3}, {Doc: "d2", Score: 2}}, // perfect
		"q2": {{Doc: "b", Score: 5}, {Doc: "a", Score: 4}},   // relevant at rank 2
	}
	qrels := Qrels{
		"q1": {"d1": 3, "d2": 2},
		"q2": {"a": 1, "b": 0},
	}
	rep := Evaluate(run, qrels, nil, nil)
	if _, ok := rep.MeanNDCG[10]; !ok {
		t.Fatal("default NDCG cutoffs not applied")
	}
	if _, ok := rep.MeanRecall[1000]; !ok {
		t.Fatal("default recall cutoffs not applied")
	}
	wantNDCG := (1.0 + 1/math.Log2(3)) / 2
	approxEq(t, "mean ndcg@10", rep.MeanNDCG[10], wantNDCG)
	approxEq(t, "mean mrr", rep.MeanMRR, (1.0+0.5)/2)
}
