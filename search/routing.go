package search

import "sort"

// RoutingIndex maps each lexical term to the shards that hold a posting for it, so
// the broker can skip the shards that cannot contribute to a query's lexical plane.
// It is the routing index of the serving design: a query touches only the shards
// whose vocabulary intersects it, which is what keeps fan-out sublinear in fleet
// size for the long-tail queries that dominate real traffic.
//
// The index is built once over the opened shards and is read-only thereafter. A
// query with no lexical terms, a pure dense query, routes to every shard, since the
// term index says nothing about which shards carry relevant vectors.
//
// The vocabulary is held as a front-coded term dictionary with a bloom-filter front
// rather than a Go map, because the fleet vocabulary is the scale-critical part of
// serving and a map of string keys to shard-id slices does not hold at 100,000 shards
// (see routingdict.go). The dictionary maps a term to its ordinal in sorted order, and
// the ordinal indexes a flat shard-set table, so a term costs only the bytes it does not
// share with its neighbor plus four bytes per shard it routes to, with no per-term map or
// string-header overhead.
type RoutingIndex struct {
	dict *routingDict

	// setOff and setIDs are the per-term shard sets in dictionary order: term ordinal i
	// routes to the shards setIDs[setOff[i]:setOff[i+1]], stored ascending and as int32 to
	// halve the resident cost of the largest array in the index.
	setOff []uint32
	setIDs []int32

	always   []int // shards routed for every query, the ones with no enumerable vocabulary
	numShard int
}

// NewRoutingIndex builds a routing index from an already-built term-to-shards map, the
// shape a persisted collection artifact loads into, so the broker can route from the
// stored index without rescanning the shards. The shard indices in the postings must
// line up with the order the shards are passed to the broker. The always list names
// shards that must see every query because their vocabulary was not enumerable, the
// impact-quantized shards, so routing never drops a candidate it cannot index.
//
// The map is the load-time input; the index converts it to the front-coded, bloom-fronted
// form at construction, so the on-disk artifact and this in-memory structure are
// independent: the artifact stays a sorted, delta-encoded map and this builds the compact
// resident form from it once at startup.
func NewRoutingIndex(postings map[string][]int, always []int, numShard int) *RoutingIndex {
	terms := make([]string, 0, len(postings))
	for t := range postings {
		terms = append(terms, t)
	}
	sort.Strings(terms)
	ri := &RoutingIndex{
		dict:     newRoutingDict(terms),
		setOff:   make([]uint32, len(terms)+1),
		always:   append([]int(nil), always...),
		numShard: numShard,
	}
	for i, t := range terms {
		ri.setOff[i] = uint32(len(ri.setIDs))
		ri.setIDs = appendSortedShards(ri.setIDs, postings[t])
	}
	ri.setOff[len(terms)] = uint32(len(ri.setIDs))
	return ri
}

// appendSortedShards appends the shard ids in ascending, deduplicated order, so the routed
// set per term is deterministic regardless of the order the source map listed them in. The
// built and persisted forms both already list them ascending, so this is usually a copy,
// but it makes the index independent of its source's ordering.
func appendSortedShards(dst []int32, ids []int) []int32 {
	if len(ids) == 0 {
		return dst
	}
	tmp := make([]int, len(ids))
	copy(tmp, ids)
	sort.Ints(tmp)
	var prev int = -1
	for _, id := range tmp {
		if id != prev {
			dst = append(dst, int32(id))
			prev = id
		}
	}
	return dst
}

// BuildRoutingIndex scans every shard's term dictionary and records, per term, the
// shards that contain it. It walks each shard's terms by id through the lexical
// region's dictionary, the same vocabulary the shard scores against, so the routing
// is exact: a term routes to a shard if and only if that shard can score it. It builds
// the term-to-shards map and hands it to NewRoutingIndex, so the scan path and the load
// path produce the same front-coded resident form.
func BuildRoutingIndex(shards []*Shard) *RoutingIndex {
	postings := make(map[string][]int)
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
			lst := postings[term]
			if len(lst) == 0 || lst[len(lst)-1] != si {
				postings[term] = append(lst, si)
			}
		}
	}
	return NewRoutingIndex(postings, nil, len(shards))
}

// Route returns the shard indices a query should fan out to. A lexical query routes
// to the union of the shards holding any of its terms; a query with no lexical terms
// routes to every shard, since only the lexical vocabulary is indexed for routing.
func (ri *RoutingIndex) Route(q Query) []int {
	return ri.RouteTerms(q.lexTerms())
}

// RouteTerms returns the shard indices an already-analyzed term set should fan out to,
// the analyze-once path: a lexical query routes to the union of the shards holding any
// of its terms, an empty set routes everywhere. It is the routing the broker calls
// after analyzing the query once at the front.
func (ri *RoutingIndex) RouteTerms(terms []string) []int {
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
		ord := ri.dict.lookup(t)
		if ord < 0 {
			continue // term absent from the fleet vocabulary, rejected at the bloom front
		}
		for _, si := range ri.setIDs[ri.setOff[ord]:ri.setOff[ord+1]] {
			add(int(si))
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

// sizeBytes estimates the index's resident size, the front-coded dictionary plus the
// shard-set table plus the always list, the number the scale measurement reports against a
// Go map of the same vocabulary to show the compaction the front-coded form buys.
func (ri *RoutingIndex) sizeBytes() int {
	return ri.dict.sizeBytes() + len(ri.setOff)*4 + len(ri.setIDs)*4 + len(ri.always)*8
}
