package search

import "github.com/tamnd/tsumugi/lexical"

// RoutingIndex maps each lexical term to the shards that hold a posting for it, so
// the broker can skip the shards that cannot contribute to a query's lexical plane.
// It is the routing index of the serving design: a query touches only the shards
// whose vocabulary intersects it, which is what keeps fan-out sublinear in fleet
// size for the long-tail queries that dominate real traffic.
//
// The index is built once over the opened shards and is read-only thereafter. A
// query with no lexical terms, a pure dense query, routes to every shard, since the
// term index says nothing about which shards carry relevant vectors.
type RoutingIndex struct {
	postings map[string][]int
	always   []int // shards routed for every query, the ones with no enumerable vocabulary
	numShard int
}

// NewRoutingIndex wraps an already-built term-to-shards map as a routing index, the
// shape a persisted collection artifact loads into, so the broker can route from the
// stored index without rescanning the shards. The shard indices in the postings must
// line up with the order the shards are passed to the broker. The always list names
// shards that must see every query because their vocabulary was not enumerable, the
// impact-quantized shards, so routing never drops a candidate it cannot index.
func NewRoutingIndex(postings map[string][]int, always []int, numShard int) *RoutingIndex {
	return &RoutingIndex{postings: postings, always: always, numShard: numShard}
}

// BuildRoutingIndex scans every shard's term dictionary and records, per term, the
// shards that contain it. It walks each shard's terms by id through the lexical
// region's dictionary, the same vocabulary the shard scores against, so the routing
// is exact: a term routes to a shard if and only if that shard can score it.
func BuildRoutingIndex(shards []*Shard) *RoutingIndex {
	ri := &RoutingIndex{postings: make(map[string][]int), numShard: len(shards)}
	for si, s := range shards {
		if s.lex == nil {
			continue
		}
		n := s.lex.TermCount()
		for id := uint32(0); id < n; id++ {
			term, ok := s.lex.Term(id)
			if !ok {
				continue
			}
			lst := ri.postings[term]
			if len(lst) == 0 || lst[len(lst)-1] != si {
				ri.postings[term] = append(lst, si)
			}
		}
	}
	return ri
}

// Route returns the shard indices a query should fan out to. A lexical query routes
// to the union of the shards holding any of its terms; a query with no lexical terms
// routes to every shard, since only the lexical vocabulary is indexed for routing.
func (ri *RoutingIndex) Route(q Query) []int {
	if q.Text == "" {
		return ri.all()
	}
	terms := lexical.Analyze(q.Text)
	if len(terms) == 0 {
		return ri.all()
	}
	seen := make([]bool, ri.numShard)
	var out []int
	add := func(si int) {
		if !seen[si] {
			seen[si] = true
			out = append(out, si)
		}
	}
	for _, si := range ri.always {
		add(si)
	}
	for _, t := range terms {
		for _, si := range ri.postings[t] {
			add(si)
		}
	}
	return out
}

func (ri *RoutingIndex) all() []int {
	out := make([]int, ri.numShard)
	for i := range out {
		out[i] = i
	}
	return out
}
