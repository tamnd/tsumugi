package eval

import (
	"context"
	"sort"
)

// Pool builds the TREC-style judging pool from a set of runs: per query, the union of
// the documents any run ranked in its top depth. It is the pooling doc 14 pins, the
// reason a few hundred judgments cover a corpus of billions: a document no system ranked
// highly cannot reach the top of any ranking, so its omitted label cannot move a
// top-cutoff metric, and judging the union of the top depth covers every document that
// can. The depth is chosen to cover the metric cutoffs with margin, so NDCG@20 is
// computed over judged documents rather than over pooling-assumed-irrelevant ones. The
// returned lists are sorted by document identifier so the pool, and the qrels judged
// from it, are deterministic across runs.
func Pool(runs []Run, depth int) map[string][]string {
	if depth <= 0 {
		depth = 100
	}
	seen := map[string]map[string]struct{}{}
	for _, run := range runs {
		for q, docs := range run {
			ordered := sortedByScore(docs)
			if len(ordered) > depth {
				ordered = ordered[:depth]
			}
			set := seen[q]
			if set == nil {
				set = map[string]struct{}{}
				seen[q] = set
			}
			for _, d := range ordered {
				set[d.Doc] = struct{}{}
			}
		}
	}
	pool := make(map[string][]string, len(seen))
	for q, set := range seen {
		docs := make([]string, 0, len(set))
		for d := range set {
			docs = append(docs, d)
		}
		sort.Strings(docs)
		pool[q] = docs
	}
	return pool
}

// PassageFunc resolves a document identifier to the passage a judge reads, returning
// false for an identifier the collection does not hold. The pooling path looks every
// pooled document up through it, so the judge grades the page's text and a document that
// has gone missing is skipped rather than judged on an empty passage.
type PassageFunc func(doc string) (Passage, bool)

// JudgePool runs the judge over every pooled query-document pair and returns the graded
// qrels, the labels every configuration on the ladder is then scored against. queries
// maps a query identifier to its text, pool is the output of Pool, and passage resolves
// each pooled document to its text. A document outside the pool is left absent from the
// qrels, the pooling-assumed-irrelevant convention Evaluate already applies, so the qrels
// hold only the judged documents and the harness defaults the rest to grade zero. The
// pool is judged once here and the resulting qrels are reused across every configuration,
// which is what makes the ladder's row-to-row comparisons valid: they are all measured
// against the identical labels. A judge error aborts, since a partial qrels would lower
// every metric for the queries it dropped, and a query with no text or a document with no
// passage is skipped rather than judged blind.
func JudgePool(ctx context.Context, j Judge, queries map[string]string, pool map[string][]string, passage PassageFunc) (Qrels, error) {
	qrels := Qrels{}
	for _, q := range sortedKeys(pool) {
		text, ok := queries[q]
		if !ok || text == "" {
			continue
		}
		for _, doc := range pool[q] {
			p, ok := passage(doc)
			if !ok {
				continue
			}
			g, err := j.Grade(ctx, text, p)
			if err != nil {
				return nil, err
			}
			if qrels[q] == nil {
				qrels[q] = map[string]float64{}
			}
			qrels[q][doc] = float64(g)
		}
	}
	return qrels, nil
}

// MergeQrels overlays the trusted judgments in gold onto a copy of base, so the
// human-labeled gold pairs win wherever they overlap the bulk labels. It is how a build
// would fold an audited gold subset into the bulk qrels for the metrics that report the
// combined set, while CompareToGold keeps the gold-only metric reported separately. base
// is not mutated, so the bulk qrels stay available for the separate report.
func MergeQrels(base, gold Qrels) Qrels {
	out := make(Qrels, len(base))
	for q, docs := range base {
		m := make(map[string]float64, len(docs))
		for d, g := range docs {
			m[d] = g
		}
		out[q] = m
	}
	for q, docs := range gold {
		if out[q] == nil {
			out[q] = map[string]float64{}
		}
		for d, g := range docs {
			out[q][d] = g
		}
	}
	return out
}

// sortedKeys returns a map's keys in ascending order, the deterministic iteration the
// pool judging and the agreement comparison both walk in so their output does not depend
// on map order.
func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
