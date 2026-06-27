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
