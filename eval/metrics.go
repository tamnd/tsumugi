// Package eval is the ranking evaluation harness: the graded-relevance metrics
// that turn a ranked list and a set of judgments into a single comparable number.
// It is what every milestone gate that claims a quality gain is measured with, and
// what the LambdaMART training optimizes against, so the two agree on the
// definition of better.
//
// The metrics are the standard information-retrieval ones over graded labels: DCG
// and its normalized form NDCG, mean average precision, mean reciprocal rank, and
// recall at a cutoff. They are defined to match pytrec_eval so a number measured
// here is comparable to the wider literature, which is the point of using the
// canonical definitions rather than a house variant.
package eval

import (
	"math"
	"sort"
)

// DCG is the discounted cumulative gain of a ranking at cutoff k. rel holds the
// graded relevance of each ranked position in rank order, position zero first. The
// gain of a label is 2^label - 1 and the discount of a position is 1/log2(rank+1)
// with rank counted from one, the standard formulation graded relevance uses. A
// cutoff at or below zero scores the whole list.
func DCG(rel []float64, k int) float64 {
	if k <= 0 || k > len(rel) {
		k = len(rel)
	}
	var dcg float64
	for i := 0; i < k; i++ {
		gain := math.Exp2(rel[i]) - 1
		dcg += gain / math.Log2(float64(i+2))
	}
	return dcg
}

// IDCG is the ideal DCG: the DCG of the same labels sorted into their best order,
// the normalizer NDCG divides by. The input is the set of relevant labels in any
// order; it is sorted descending here.
func IDCG(labels []float64, k int) float64 {
	ideal := append([]float64(nil), labels...)
	sort.Sort(sort.Reverse(sort.Float64Slice(ideal)))
	return DCG(ideal, k)
}

// NDCG is the normalized discounted cumulative gain at cutoff k: the ranking's DCG
// over the ideal DCG of the same label multiset. It is the headline ranking metric
// because it rewards putting the most relevant documents highest and is comparable
// across queries with different numbers of relevant documents. ranked holds the
// graded labels in rank order; allLabels is the full label multiset for the query,
// which may include relevant documents the ranking missed. A query with no relevant
// documents scores zero.
func NDCG(ranked []float64, allLabels []float64, k int) float64 {
	idcg := IDCG(allLabels, k)
	if idcg == 0 {
		return 0
	}
	return DCG(ranked, k) / idcg
}

// AveragePrecision is the average of the precision values taken at each relevant
// document in the ranking, with a label treated as relevant when it is positive.
// It is the per-query term mean average precision averages. total is the number of
// relevant documents for the query, which bounds the denominator so a ranking that
// misses relevant documents is penalized.
func AveragePrecision(ranked []float64, total int) float64 {
	if total <= 0 {
		return 0
	}
	var hits int
	var sum float64
	for i, r := range ranked {
		if r > 0 {
			hits++
			sum += float64(hits) / float64(i+1)
		}
	}
	return sum / float64(total)
}

// ReciprocalRank is one over the rank of the first relevant document, counted from
// one, or zero if the ranking holds no relevant document. Its mean over queries is
// MRR, the metric for tasks where only the first good result matters.
func ReciprocalRank(ranked []float64) float64 {
	for i, r := range ranked {
		if r > 0 {
			return 1 / float64(i+1)
		}
	}
	return 0
}

// RecallAtK is the fraction of a query's relevant documents that appear in the top
// k of the ranking, a label positive meaning relevant. It is the recall gate the
// early cascade stages are judged on, since their job is to not drop a relevant
// document rather than to order it.
func RecallAtK(ranked []float64, total, k int) float64 {
	if total <= 0 {
		return 0
	}
	if k <= 0 || k > len(ranked) {
		k = len(ranked)
	}
	var hits int
	for i := 0; i < k; i++ {
		if ranked[i] > 0 {
			hits++
		}
	}
	return float64(hits) / float64(total)
}

// MeanNDCG averages NDCG at cutoff k over a set of queries, each query a ranking of
// graded labels paired with its full label multiset. It is the aggregate number a
// model is reported at and the training tunes up.
func MeanNDCG(rankings [][]float64, labels [][]float64, k int) float64 {
	if len(rankings) == 0 {
		return 0
	}
	var sum float64
	for i := range rankings {
		sum += NDCG(rankings[i], labels[i], k)
	}
	return sum / float64(len(rankings))
}
