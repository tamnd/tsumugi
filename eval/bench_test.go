package eval

import (
	"fmt"
	"testing"
)

// benchRunQrels builds a synthetic run and qrels of nQueries queries, each with a run
// of depth documents and nRel relevant judgments scattered through it, the shape the
// harness scores over a real run: a few hundred ranked documents per query with a
// handful judged relevant. The scores descend so the run is already ordered, and the
// relevant documents sit at a fixed stride so recall and NDCG have something to find.
func benchRunQrels(nQueries, depth, nRel int) (Run, Qrels) {
	run := Run{}
	qrels := Qrels{}
	for q := 0; q < nQueries; q++ {
		qid := fmt.Sprintf("q%d", q)
		docs := make([]RankedDoc, depth)
		for i := 0; i < depth; i++ {
			docs[i] = RankedDoc{Doc: fmt.Sprintf("d%d", i), Score: float64(depth - i)}
		}
		run[qid] = docs
		labels := map[string]float64{}
		for r := 0; r < nRel; r++ {
			labels[fmt.Sprintf("d%d", r*depth/nRel)] = float64(1 + r%3)
		}
		qrels[qid] = labels
	}
	return run, qrels
}

// BenchmarkEvaluate measures the per-call cost of scoring a run against qrels at the
// doc 14 cutoffs, the work the quality gate does once per evaluation. It is the
// substrate the reproducibility check and the offline NDCG report run on, so its cost
// is the floor of how fast a corpus's quality can be measured.
func BenchmarkEvaluate(b *testing.B) {
	run, qrels := benchRunQrels(200, 1000, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Evaluate(run, qrels, DefaultNDCGCutoffs, DefaultRecallCutoffs)
		if rep.NumQueries != 200 {
			b.Fatalf("scored %d queries, want 200", rep.NumQueries)
		}
	}
}
