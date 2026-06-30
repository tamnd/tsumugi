package eval

// This is the evaluation harness doc 14 pins: it reads an engine's run and the
// graded qrels, joins them per query by document identifier, and reports NDCG at the
// page cutoffs, MRR, and Recall at the deep cutoffs, per query and averaged. It is
// the Go counterpart of the pytrec_eval flow, computing the same metric definitions
// (the 2^grade-1 gain, the score-tie order, the unjudged-is-irrelevant pooling
// assumption) so a number it reports is comparable to the wider retrieval
// literature rather than to a house variant.

// DefaultNDCGCutoffs are the NDCG cutoffs doc 14 reports: ten, the page-one
// headline the LambdaMART objective optimizes, and twenty, which catches a model
// that orders the first ten well and the next ten poorly.
var DefaultNDCGCutoffs = []int{10, 20}

// DefaultRecallCutoffs are the recall cutoffs doc 14 reports: a hundred and a
// thousand, the depths the L0 retrieval and L1 cut are judged on, since their job is
// to keep the relevant documents the precision stage then orders rather than to
// order them.
var DefaultRecallCutoffs = []int{100, 1000}

// QueryReport is one query's scores: NDCG keyed by cutoff, the reciprocal rank of
// its first relevant document, and Recall keyed by cutoff, plus the judged-relevant
// count and how many documents the run returned for it, the context that says
// whether a low score is a ranking failure or an empty result.
type QueryReport struct {
	Query     string
	NDCG      map[int]float64
	MRR       float64
	Recall    map[int]float64
	NumRel    int
	Retrieved int
}

// Report is the harness output: every scored query and the means over them. The
// means average only the queries with at least one relevant judgment, the trec_eval
// convention, since a query with no relevant document has an undefined NDCG and a
// trivially zero recall that would only dilute the average; NumQueries is that
// scored count and NumSkipped is how many judged queries were dropped for having no
// relevant document, so the average is never silently over a different denominator.
type Report struct {
	PerQuery   []QueryReport
	MeanNDCG   map[int]float64
	MeanMRR    float64
	MeanRecall map[int]float64
	NumQueries int
	NumSkipped int
}

// Evaluate scores a run against qrels at the given NDCG and recall cutoffs. The
// qrels define the query set, since a query without judgments cannot be scored; a
// judged query the run returned nothing for scores zero, the honest penalty for an
// engine that dropped it. For each query it orders the run by score (the trec_eval
// tie rule), maps each returned document to its grade (an unjudged document is
// grade zero, the pooling assumption), and computes NDCG over the ranked grades
// against the query's full judged-label multiset, MRR over the ranked grades, and
// recall against the judged-relevant count. Passing nil cutoffs uses the doc 14
// defaults.
func Evaluate(run Run, qrels Qrels, ndcgCutoffs, recallCutoffs []int) Report {
	if ndcgCutoffs == nil {
		ndcgCutoffs = DefaultNDCGCutoffs
	}
	if recallCutoffs == nil {
		recallCutoffs = DefaultRecallCutoffs
	}
	rep := Report{
		MeanNDCG:   map[int]float64{},
		MeanRecall: map[int]float64{},
	}
	for _, q := range sortedQueries(qrels) {
		labels := qrels[q]
		allLabels := make([]float64, 0, len(labels))
		numRel := 0
		for _, g := range labels {
			allLabels = append(allLabels, g)
			if g > 0 {
				numRel++
			}
		}
		// A query with no relevant judgment is dropped from the means: its NDCG is
		// undefined (a zero ideal DCG) and its recall denominator is zero, so scoring
		// it would only add a zero that misrepresents the average.
		if numRel == 0 {
			rep.NumSkipped++
			continue
		}
		ranked := rankedGrades(run[q], labels)
		qr := QueryReport{
			Query:     q,
			NDCG:      map[int]float64{},
			Recall:    map[int]float64{},
			MRR:       ReciprocalRank(ranked),
			NumRel:    numRel,
			Retrieved: len(run[q]),
		}
		for _, k := range ndcgCutoffs {
			qr.NDCG[k] = NDCG(ranked, allLabels, k)
		}
		for _, k := range recallCutoffs {
			qr.Recall[k] = RecallAtK(ranked, numRel, k)
		}
		rep.PerQuery = append(rep.PerQuery, qr)
		rep.MeanMRR += qr.MRR
		for _, k := range ndcgCutoffs {
			rep.MeanNDCG[k] += qr.NDCG[k]
		}
		for _, k := range recallCutoffs {
			rep.MeanRecall[k] += qr.Recall[k]
		}
	}
	rep.NumQueries = len(rep.PerQuery)
	if rep.NumQueries > 0 {
		n := float64(rep.NumQueries)
		rep.MeanMRR /= n
		for k := range rep.MeanNDCG {
			rep.MeanNDCG[k] /= n
		}
		for k := range rep.MeanRecall {
			rep.MeanRecall[k] /= n
		}
	}
	return rep
}

// rankedGrades maps a query's scored run to the grade of each returned document in
// scored order, an unjudged document defaulting to grade zero, which is the per-rank
// label vector every metric reads.
func rankedGrades(docs []RankedDoc, labels map[string]float64) []float64 {
	ordered := sortedByScore(docs)
	grades := make([]float64, len(ordered))
	for i, d := range ordered {
		grades[i] = labels[d.Doc]
	}
	return grades
}
