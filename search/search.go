// Package search is the online half of tsumugi: the per-shard retrieval that runs
// the ranking cascade against one .tsumugi file, and the broker that fans a query
// across many shards and merges a fleet-wide top-k.
//
// The split mirrors the spec's serving design. A Shard owns one immutable file and
// turns a query into a model-ranked top-k over the documents in that file: it runs
// the lexical and dense and learned-sparse retrieval planes, attaches each
// candidate's feature row, and runs the cascade. A Broker owns many shards, routes
// a query to the ones that can contribute through a routing index, gathers their
// candidates, and runs one global rerank against fleet-wide statistics so the merged
// top-k is the same result a single index over every shard would give.
//
// The exactness the broker guarantees is at the rerank-and-merge layer, the place
// the spec's "exact, not best-effort" requirement lives: a document's final score is
// the global L2 model over its per-document feature row, which is identical whether
// the document sits in a shard or in a single combined index, so given a
// recall-complete candidate set the broker reproduces the monolith's top-k exactly.
package search

import "github.com/tamnd/tsumugi/lexical"

// Query is one search request across the planes a shard carries. Text drives the
// lexical plane, Sparse the learned-sparse plane as a term-to-weight map, Vector the
// dense plane, and K is the number of results wanted. A plane is skipped when its
// field is empty and the shard lacks that region, so the same Query runs against a
// lexical-only shard and a full multi-plane shard.
type Query struct {
	Text   string
	Sparse map[string]int
	Vector []float32
	K      int

	// Terms, when set, is the analyzed lexical term set the broker computed once at the
	// front, in query order. A shard with Terms set scores those terms directly instead
	// of re-running the analysis chain over Text, the spec's analyze-once-at-broker rule:
	// the chain runs one time per query, not once per shard the fan-out visits. It is nil
	// on the raw-string path, where each shard analyzes Text itself, and the two paths
	// produce identical results when Terms is the analysis of Text.
	Terms []string

	// TermIDF, when set, overrides the lexical idf of each query term with a value
	// computed from outside the shard, the collection-wide idf the broker pushes down
	// so every shard scores a term against the same df and N. It is nil on the
	// single-shard path, where the shard's local idf is already the collection idf.
	TermIDF map[string]float64

	// AvgFieldLen, when set, overrides the per-field average lengths the broker
	// normalizes BM25F by during the global rerank, the fleet-wide averages an
	// aggregator pushes down so every broker beneath it normalizes a field's length
	// against the same denominator. It is the length-normalization counterpart to
	// TermIDF: idf makes a term's weight comparable across brokers, AvgFieldLen makes a
	// field's length normalization comparable, and together they make a partitioned
	// deployment's L2 scores land on one scale so the aggregator's merge stays exact (doc
	// 11, "Exactness up the tree", the partitioned-GlobalStats case). It is nil on the
	// single-broker path, where the broker's own fleet averages are already the
	// collection averages, and the field order is the online extractor's: title, body, url.
	AvgFieldLen *[3]float64

	// L0, when positive, overrides the shard's default L0 candidate width per plane, the
	// first rung of the broker's degradation ladder: under budget pressure the broker
	// shrinks L0 so each shard retrieves and ranks a smaller candidate set precisely
	// rather than skipping a stage (doc 11, "The degradation order"). It is zero on the
	// full-quality path, where each shard uses its built-in L0.
	L0 int
}

// lexTerms returns the analyzed lexical term set for the query: the broker's
// pre-analyzed Terms when set, otherwise the analysis of Text, so a caller that did
// not pre-analyze still gets the right terms. It is the one place the two paths join.
func (q Query) lexTerms() []string {
	if q.Terms != nil {
		return q.Terms
	}
	if q.Text == "" {
		return nil
	}
	return lexical.Analyze(q.Text)
}

// Hit is one ranked result: the global document id and its final model score. The
// id is fleet-wide, the shard's node base plus the in-shard id, so a hit is
// meaningful after the broker has merged results from many shards.
type Hit struct {
	DocID uint32
	Score float64
}

// scored is an in-flight candidate carrying its global id and a retrieval-plane
// score, used to merge per-shard ranked lists into a global ranked list before the
// cascade fuses them.
type scored struct {
	docID uint32
	score float64
}
