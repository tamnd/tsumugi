package search

import (
	"math"
	"sort"
)

// The incremental publish path keeps a running broker's per-publish cost proportional to
// one shard's vocabulary rather than the whole fleet's. The base broker state holds the
// routing index built over the shards present at the last full build (a prefix of the shard
// slice); shards published since sit in a small overlay, each carrying a front-coded
// dictionary of only its own terms. A query routes against the base index and then probes
// the overlay's per-shard dictionaries, so adding a shard never rescans the fleet vocabulary
// the way a full BuildRoutingIndex does.
//
// The overlay is bounded: once it grows past overlayLimit shards, the next publish folds it
// back into the base with one full rebuild. With the limit set near the square root of the
// base size, a full rebuild amortizes to roughly a square-root fraction of its cost per
// publish, the standard memtable-then-merge shape, so per-publish cost grows with the
// square root of the fleet rather than linearly. A retire keeps the simple full rebuild
// (retires are rare and a rebuild also folds the overlay back in), so only the publish path,
// the one a continuous ingest hammers at scale, is incremental.

// overlayEntry is one published-since-base shard's routing contribution: its absolute index
// in the broker's shard slice and a front-coded dictionary of its own terms. The dictionary
// answers "does this shard hold this term" with a bloom-fronted lookup, so probing the
// overlay for a query term rejects the shards that do not hold it in close to constant time.
type overlayEntry struct {
	si   int
	dict *routingDict
}

// shardDict builds a front-coded dictionary over a single shard's sorted term set, the
// overlay's per-shard routing structure. It returns nil for a shard with no enumerable
// lexical vocabulary, which therefore contributes nothing to a lexical route, matching
// BuildRoutingIndex's treatment of such a shard. Building it costs one shard's vocabulary,
// the per-publish work the incremental path trades the fleet rescan for.
func shardDict(s *Shard) *routingDict {
	if s.lex == nil {
		return nil
	}
	n := s.lex.TermCount()
	if n == 0 {
		return nil
	}
	terms := make([]string, 0, n)
	for id := uint32(0); id < n; id++ {
		if t, ok := s.lex.Term(id); ok {
			terms = append(terms, t)
		}
	}
	if len(terms) == 0 {
		return nil
	}
	sort.Strings(terms)
	return newRoutingDict(terms)
}

// overlayLimit is the number of published-since-base shards the overlay holds before a
// publish folds it into the base. It is about the square root of the base size, with a floor
// so a small collection does not rebuild on nearly every publish (a rebuild over a handful
// of shards is cheap anyway). At the square root the amortized per-publish cost of the
// periodic full rebuild matches the per-publish cost of carrying the overlay, the balance
// that makes incremental publish scale with the square root of the fleet.
func overlayLimit(baseN int) int {
	lim := int(math.Sqrt(float64(baseN)))
	if lim < 16 {
		return 16
	}
	return lim
}

// appendOverlay returns the overlay extended by one shard's routing contribution, without
// mutating the input: a stored broker state is immutable and read by concurrent queries, so
// a publish copies the small overlay slice rather than appending in place. A shard with no
// enumerable vocabulary adds no entry (it is still in the shard slice and so still answers a
// term-free query through the all-shards route). The copy is bounded by overlayLimit, so it
// is the cheap part of a publish.
func appendOverlay(prev []overlayEntry, si int, s *Shard) []overlayEntry {
	d := shardDict(s)
	if d == nil {
		// No vocabulary to route on; the shard is carried by the shard slice alone.
		next := make([]overlayEntry, len(prev))
		copy(next, prev)
		return next
	}
	next := make([]overlayEntry, len(prev), len(prev)+1)
	copy(next, prev)
	return append(next, overlayEntry{si: si, dict: d})
}

// withShard returns the broker state that results from publishing one shard. When the
// overlay is below its limit the shard joins the overlay incrementally, sharing the base
// routing index and folding its statistics into the running totals, so the publish costs one
// shard's vocabulary plus a copy of the small overlay. When the overlay is at its limit the
// publish triggers a full rebuild over the whole set, folding the overlay back into a fresh
// base. Either way the returned state is independent of the receiver, so the atomic swap that
// stores it never disturbs a query reading the old state.
func (st *brokerState) withShard(s *Shard) *brokerState {
	all := make([]*Shard, len(st.shards)+1)
	copy(all, st.shards)
	all[len(st.shards)] = s

	published := len(st.shards) - st.baseN
	if published+1 > overlayLimit(st.baseN) {
		return newState(all)
	}

	si := len(st.shards)
	ns := &brokerState{
		shards:  all,
		baseN:   st.baseN,
		routing: st.routing,
		overlay: appendOverlay(st.overlay, si, s),
		sums:    st.sums.add(shardSums(s)),
	}
	ns.stats = ns.sums.global()
	return ns
}

// routeTerms returns the shard indices a query should fan out to over the whole served set,
// base plus overlay. An empty term set routes everywhere, since routing indexes only the
// lexical vocabulary and a term-free query (a pure dense query) says nothing about which
// shards carry relevant vectors. Otherwise it unions the base index's route with the overlay
// shards whose own dictionary holds any query term. The base route covers indices below
// baseN and the overlay covers indices at or above it, so the two ranges never overlap and
// the overlay only needs to dedup within itself.
func (st *brokerState) routeTerms(terms []string) []int {
	if len(terms) == 0 {
		out := make([]int, len(st.shards))
		for i := range out {
			out[i] = i
		}
		return out
	}
	out := st.routing.RouteTerms(terms)
	if len(st.overlay) == 0 {
		return out
	}
	seen := make([]bool, len(st.shards)-st.baseN)
	for _, e := range st.overlay {
		idx := e.si - st.baseN
		if seen[idx] {
			continue
		}
		for _, t := range terms {
			if e.dict.lookup(t) >= 0 {
				seen[idx] = true
				out = append(out, e.si)
				break
			}
		}
	}
	return out
}

// overlaySize is the resident cost of the overlay's per-shard dictionaries, reported next to
// the base routing index's size so a scale measurement accounts for the whole routing
// footprint, base plus the not-yet-folded overlay.
func (st *brokerState) overlaySize() int {
	total := 0
	for _, e := range st.overlay {
		total += e.dict.sizeBytes() + 8
	}
	return total
}
