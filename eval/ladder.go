package eval

import (
	"math"
	"math/rand"
)

// ladderSeed fixes the randomization test's permutation stream so a ladder is
// reproducible: the same per-query differences yield the same p-value across runs,
// which a committed quality number and a CI gate both need. It is a build-time
// constant rather than a clock-seeded source for that reason.
const ladderSeed = 0x1e3779b97f4a7c15

// permutations is how many sign-flip permutations the randomization test draws. A few
// thousand is enough to resolve a p-value around the 0.05 gate without the cost of an
// exact enumeration, which is exponential in the query count.
const permutations = 10000

// Config is one rung of the NDCG ladder: the configuration's name and the run it
// produced over the evaluation query set. The ladder evaluates each against the shared
// qrels, so a rung is a configuration paired with its output, the BM25F baseline, the
// learned-sparse step, the dense step, the LambdaMART rerank, each a Config the caller
// runs the query set through and hands to Ladder.
type Config struct {
	Name string
	Run  Run
}

// Rung is one row of the ladder table: the configuration's scores over the bulk qrels
// and, reported separately, over the human-labeled gold subset, the gain in NDCG@10
// over the rung below it, and whether that gain is significant. The gold report is kept
// separate from the bulk rather than averaged in, the discipline that keeps the quality
// numbers honest when the labels are mostly LLM-generated: the trusted human-labeled
// metric is always visible on its own.
type Rung struct {
	Name string
	// Bulk is the configuration's report over the full qrels, the LLM-labeled pool.
	Bulk Report
	// Gold is the configuration's report over the human-labeled gold subset alone, the
	// trusted metric reported separately so it is never diluted into the bulk.
	Gold Report
	// GainNDCG10 is the mean per-query NDCG@10 gain over the rung below, zero for the base
	// rung. A ladder step earns its place only with a positive gain here.
	GainNDCG10 float64
	// PValue is the two-sided paired-randomization p-value that the NDCG@10 gain over the
	// rung below is zero, the significance test that says whether the gain would hold on a
	// different query sample.
	PValue float64
	// Significant reports whether the gain is both positive and significant at the 0.05
	// level, the gate a ladder step must pass: a gain that is not significant is noise, and
	// a step justified by noise does not earn its latency cost.
	Significant bool
}

// significanceAlpha is the level a ladder step's gain must be significant at, the
// conventional 0.05, so a step's gain is accepted only when a paired test rejects the
// null of no gain at that level.
const significanceAlpha = 0.05

// Ladder evaluates each configuration against the shared bulk qrels and the gold subset,
// computes each rung's NDCG@10 gain over the one below it, and tests that gain for
// significance, the NDCG ladder doc 14 makes the spine of the quality benchmark. The
// configurations are evaluated in the order given, which is the ladder's order from the
// baseline up, and the pool is judged once so every rung is scored against identical
// labels, which is what makes the row-to-row comparisons valid. Each step's gate is the
// gain over the step below being positive and significant; a step whose gain is not is
// dead weight that adds latency for nothing, and the rung records both the gain and the
// p-value so the caller can apply the gate.
func Ladder(configs []Config, bulk, gold Qrels, ndcgCutoffs, recallCutoffs []int) []Rung {
	if ndcgCutoffs == nil {
		ndcgCutoffs = DefaultNDCGCutoffs
	}
	rungs := make([]Rung, len(configs))
	var prev Report
	for i, c := range configs {
		b := Evaluate(c.Run, bulk, ndcgCutoffs, recallCutoffs)
		g := Evaluate(c.Run, gold, ndcgCutoffs, recallCutoffs)
		r := Rung{Name: c.Name, Bulk: b, Gold: g}
		if i > 0 {
			gain, p := PairedSignificance(prev, b, 10)
			r.GainNDCG10 = gain
			r.PValue = p
			r.Significant = gain > 0 && p < significanceAlpha
		}
		rungs[i] = r
		prev = b
	}
	return rungs
}

// PairedSignificance runs a paired randomization test over the per-query NDCG@k
// differences between two reports, the significance test doc 14 puts behind every ladder
// gate. It pairs the two reports' per-query NDCG@k by query, takes the difference per
// query, and asks whether the mean difference is distinguishable from zero by drawing
// sign-flip permutations of the differences and counting how often a permuted mean is at
// least as extreme as the observed one. It returns the observed mean difference and the
// two-sided p-value. The test is paired because the same queries are scored under both
// configurations, and it is a randomization test rather than a t-test because it makes no
// normality assumption about per-query NDCG, which is bounded and skewed. A gain that is
// large on average but driven by a few queries fails this test, which is the point: a step
// earns its place by a gain that holds across the query set, not by a gain that one query
// carries.
func PairedSignificance(below, above Report, k int) (float64, float64) {
	bm := perQueryNDCG(below, k)
	am := perQueryNDCG(above, k)
	var diffs []float64
	for _, q := range sortedKeys(am) {
		if bv, ok := bm[q]; ok {
			diffs = append(diffs, am[q]-bv)
		}
	}
	n := len(diffs)
	if n == 0 {
		return 0, 1
	}
	obs := mean(diffs)
	rng := rand.New(rand.NewSource(ladderSeed))
	atLeast := 0
	absObs := math.Abs(obs)
	for p := 0; p < permutations; p++ {
		var s float64
		for _, d := range diffs {
			if rng.Intn(2) == 0 {
				s += d
			} else {
				s -= d
			}
		}
		if math.Abs(s/float64(n)) >= absObs-1e-12 {
			atLeast++
		}
	}
	// The plus-one on both sides is the conventional randomization-test correction: the
	// observed arrangement is itself one of the permutations, so a p-value can never be
	// reported as exactly zero, which would overstate the evidence.
	return obs, float64(atLeast+1) / float64(permutations+1)
}

// perQueryNDCG indexes a report's per-query NDCG at cutoff k by query, the pairing the
// significance test joins the two reports on. A query absent from a report's per-query
// list, dropped for having no relevant judgment, is absent from the map, so the test
// pairs only the queries both reports scored.
func perQueryNDCG(rep Report, k int) map[string]float64 {
	m := make(map[string]float64, len(rep.PerQuery))
	for _, qr := range rep.PerQuery {
		m[qr.Query] = qr.NDCG[k]
	}
	return m
}

// mean is the arithmetic mean of a slice, the observed statistic the randomization test
// compares its permutations against.
func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}
