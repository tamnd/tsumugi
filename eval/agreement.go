package eval

import "math"

// UmbrelaTauGate is the label-trust threshold doc 14 pins: a judge's labels are trusted
// for the bulk only while they reproduce human judgment at a Kendall tau at or above this
// value, the UMBRELA result that justifies LLM labels at all. A drop below it fails the
// label validation, which means the prompt drifted, the judge model changed behavior, or
// the gold set surfaced a query the judge handles poorly, and any of those has to be
// fixed before the labels are trusted to train or evaluate.
const UmbrelaTauGate = 0.87

// Agreement is the gold-set agreement check's result: the Kendall tau between the judge's
// labels and the human labels on the gold subset, the per-grade confusion matrix that
// says not just how much the judge disagrees but how badly, and the counts the two are
// computed over. It is the M9 gate's trustworthiness condition, reported on every training
// run and every quality benchmark rather than once, since the judge model and the prompt
// can change.
type Agreement struct {
	// KendallTau is the tau-b rank correlation between the judge and human grades over the
	// compared pairs, tau-b because the zero-to-three scale ties heavily and tau-b is the
	// variant that corrects for ties.
	KendallTau float64
	// Confusion is the per-grade confusion matrix indexed [human][judge], each cell the
	// count of gold pairs the human gave the row grade and the judge gave the column grade.
	// The diagonal is agreement; an off-diagonal cell near the corner, grade zero against
	// grade three, is the catastrophic confusion NDCG cannot forgive, while an adjacent
	// off-diagonal cell is the forgivable kind.
	Confusion [][]int
	// N is the number of gold pairs compared, the pairs present in both the gold qrels and
	// the judge's labels.
	N int
	// Missing is the number of gold pairs the judge produced no label for, which should be
	// zero when the judge ran over the whole gold set and is surfaced rather than hidden so
	// a gap in the judge's coverage is visible instead of silently shrinking the agreement
	// base.
	Missing int
}

// Passes reports whether the judge clears the UMBRELA trust gate on the gold set, the
// condition the bulk labels are trusted under. An empty comparison does not pass, since a
// gate with no evidence behind it is not a gate.
func (a Agreement) Passes() bool {
	return a.N >= 2 && a.KendallTau >= UmbrelaTauGate
}

// CompareToGold scores a judge's labels against a human-labeled gold qrels subset by
// Kendall tau and the per-grade confusion matrix, the gold-set agreement check the label
// validation is. It compares only the pairs the gold set holds, since the gold set is the
// trusted subset the judge is checked against, and it pairs each gold pair's human grade
// with the judge's grade for the same query and document. A gold pair the judge has no
// label for is counted as missing rather than scored, so the agreement is measured over
// the pairs both labeled and the coverage gap is reported alongside.
func CompareToGold(judge, gold Qrels) Agreement {
	conf := make([][]int, MaxGrade+1)
	for i := range conf {
		conf[i] = make([]int, MaxGrade+1)
	}
	var human, model []float64
	missing := 0
	for _, q := range sortedKeys(gold) {
		jq := judge[q]
		for _, doc := range sortedKeys(gold[q]) {
			hg := clampGrade(int(gold[q][doc]))
			mgRaw, ok := jq[doc]
			if !ok {
				missing++
				continue
			}
			mg := clampGrade(int(mgRaw))
			human = append(human, float64(hg))
			model = append(model, float64(mg))
			conf[hg][mg]++
		}
	}
	return Agreement{
		KendallTau: KendallTauB(human, model),
		Confusion:  conf,
		N:          len(human),
		Missing:    missing,
	}
}

// KendallTauB computes the tau-b rank correlation between two equal-length grade slices,
// the tie-corrected Kendall coefficient. tau-b is the right variant for graded relevance
// because the zero-to-three scale produces many ties, and the plain tau-a would charge
// those ties against the correlation while tau-b divides them out, which is the
// correlation the UMBRELA result is reported at. It is computed by counting concordant and
// discordant pairs directly, which is quadratic in the number of pairs and fine for the
// gold set's few hundred, and it returns zero for a degenerate input, fewer than two
// points or a slice with no variation, since a correlation is undefined there.
func KendallTauB(a, b []float64) float64 {
	n := len(a)
	if n != len(b) || n < 2 {
		return 0
	}
	var nc, nd int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			da := sign(a[i] - a[j])
			db := sign(b[i] - b[j])
			switch p := da * db; {
			case p > 0:
				nc++
			case p < 0:
				nd++
			}
		}
	}
	n0 := float64(n*(n-1)) / 2
	denom := math.Sqrt((n0 - tiePairs(a)) * (n0 - tiePairs(b)))
	if denom == 0 {
		return 0
	}
	return float64(nc-nd) / denom
}

// tiePairs counts the tied pairs within a grade slice, the sum over each distinct grade
// of its count-choose-two, the correction tau-b subtracts so ties on one side do not
// count against the correlation.
func tiePairs(v []float64) float64 {
	counts := map[float64]int{}
	for _, x := range v {
		counts[x]++
	}
	var t float64
	for _, c := range counts {
		t += float64(c*(c-1)) / 2
	}
	return t
}

// sign returns the sign of a difference as minus one, zero, or one, the comparison the
// concordance count is built from.
func sign(x float64) int {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}
