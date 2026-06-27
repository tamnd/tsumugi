package graph

import (
	"math"
	"sort"
)

// This file is the node reordering doc 06 calls the heart of the graph plane.
// The Boldi-Vigna codec is fixed; the ordering is the lever that makes it spend
// three bits an edge instead of twenty. A random order scatters a node's
// out-neighbors uniformly across the id space, so the gaps are large and a gap
// code spends close to log2(N) bits an edge. The right order puts a node's
// out-neighbors at nearby ids, so the gaps collapse and the lists of nearby
// nodes become near-identical, which is what the gap, interval, and reference
// layers all reward.
//
// The order is built in two passes, exactly as the spec pins them: host and
// domain grouping first (a cheap sort that captures the dominant regularity,
// that most links are internal to a host), then Recursive Graph Bisection, which
// spends the freedom the grouping leaves to squeeze out the last few bits. The
// same order assigns the dense docID space, so one pass serves graph
// compression, identity assignment, and posting impact locality at once.

// BPConfig tunes the Recursive Graph Bisection pass.
type BPConfig struct {
	// MaxIters caps the swap iterations at each split, the paper's default 20,
	// which usually converges in fewer.
	MaxIters int
	// MinLeaf stops the recursion when a segment is this small. The spec pins
	// MaxDepth = log2(N) - 5, leaves of about 2^5 = 32 nodes, where the within-
	// leaf order is left as the host-grouping order because the per-edge savings
	// of splitting an already-small, already-similar block no longer pay for the
	// recursion.
	MinLeaf int
}

// DefaultBPConfig returns the spec's defaults: 20 swap iterations and a 32-node
// leaf, the log2(N) - 5 depth written as a leaf size.
func DefaultBPConfig() BPConfig {
	return BPConfig{MaxIters: 20, MinLeaf: 32}
}

// Reorder computes the host-grouping-then-Recursive-Graph-Bisection order over
// the out-adjacency and returns it as order, where order[i] is the OLD node id
// placed at NEW dense id i. The inverse (Invert) gives the new id of an old node,
// the relabeling the build applies to assign dense docIDs.
//
// group, when non-nil, is the first-pass grouping key per old node: nodes sharing
// a key (a host, with hosts of one domain sharing a domain prefix in the key's
// order) are made contiguous before bisection. A nil group skips the grouping
// pass and bisects the identity order, which is the pure-BP path the unit gate
// exercises.
//
// The out-lists are read but never mutated, so the same slices the caller built
// the graph from can be passed straight in.
func Reorder(out [][]int32, group []int, cfg BPConfig) []int {
	n := len(out)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if n <= 1 {
		return order
	}
	// First pass: host and domain grouping. A stable sort by group key then old
	// id makes a host's pages contiguous while preserving their input order
	// within the host, so the grouping never fights the host+url order the build
	// already cut the stream in.
	if group != nil {
		sort.SliceStable(order, func(a, b int) bool {
			ga, gb := group[order[a]], group[order[b]]
			if ga != gb {
				return ga < gb
			}
			return order[a] < order[b]
		})
	}
	// Second pass: Recursive Graph Bisection refines the grouped order in place.
	if cfg.MaxIters > 0 {
		bisect(order, out, cfg)
	}
	return order
}

// Invert turns an order (new id -> old id) into its inverse (old id -> new id),
// the map the build uses to relabel every edge and every per-document signal.
func Invert(order []int) []int {
	inv := make([]int, len(order))
	for newID, oldID := range order {
		inv[oldID] = newID
	}
	return inv
}

// OutLists materializes a region's forward adjacency as plain int32 slices, the
// form Reorder reads. It decodes each node's out-list once; the build that has
// the lists in hand already passes them directly and skips this.
func OutLists(g *Region) [][]int32 {
	out := make([][]int32, g.NodeCount())
	for v := range out {
		nb := g.OutNeighbors(v)
		if len(nb) == 0 {
			continue
		}
		s := make([]int32, len(nb))
		for i, u := range nb {
			s[i] = int32(u)
		}
		out[v] = s
	}
	return out
}

// bisect recursively splits seg (a contiguous window of the order) into two
// balanced halves, refining which nodes land on each side so nodes that share
// out-neighbors end up together, then recurses into each half. The in-order walk
// of the recursion is the final order, because left is always kept before right.
func bisect(seg []int, out [][]int32, cfg BPConfig) {
	n := len(seg)
	if n <= cfg.MinLeaf {
		return
	}
	mid := n / 2
	refine(seg, mid, out, cfg.MaxIters)
	bisect(seg[:mid], out, cfg)
	bisect(seg[mid:], out, cfg)
}

// refine runs the swap iterations on one split. Each iteration recomputes the
// per-term degree on each side, computes every node's move gain (the bits saved
// by flipping its side), sorts each side by gain, and swaps the best-to-move
// pairs across the cut while the combined gain is positive. Swapping pairs keeps
// the halves exactly balanced, which is what bounds the recursion depth at
// log2(n).
func refine(seg []int, mid int, out [][]int32, maxIters int) {
	left := seg[:mid]
	right := seg[mid:]
	n1, n2 := len(left), len(right)
	if n1 == 0 || n2 == 0 {
		return
	}
	for it := 0; it < maxIters; it++ {
		deg1 := termDegrees(left, out)
		deg2 := termDegrees(right, out)
		lg := sideGains(left, out, deg1, deg2, n1, n2, true)
		rg := sideGains(right, out, deg1, deg2, n1, n2, false)
		sort.Slice(lg, func(a, b int) bool { return lg[a].gain > lg[b].gain })
		sort.Slice(rg, func(a, b int) bool { return rg[a].gain > rg[b].gain })
		k := 0
		for k < len(lg) && k < len(rg) && lg[k].gain+rg[k].gain > 1e-9 {
			k++
		}
		if k == 0 {
			break
		}
		// Each position appears once in its sorted list, so the k swaps touch
		// distinct slots and never collide.
		for i := 0; i < k; i++ {
			lp, rp := lg[i].pos, rg[i].pos
			left[lp], right[rp] = right[rp], left[lp]
		}
	}
}

// nodeGain is a node's index within its side and the gain of moving it across
// the cut.
type nodeGain struct {
	pos  int
	gain float64
}

// termDegrees counts, per out-neighbor term, how many nodes on the side link to
// it. This is deg1(t) or deg2(t) in the cost function.
func termDegrees(side []int, out [][]int32) map[int32]int {
	d := make(map[int32]int)
	for _, v := range side {
		for _, t := range out[v] {
			d[t]++
		}
	}
	return d
}

// sideGains computes the move gain of every node on a side. fromLeft says which
// direction the move is, which flips which degree counter increments and which
// decrements.
func sideGains(side []int, out [][]int32, deg1, deg2 map[int32]int, n1, n2 int, fromLeft bool) []nodeGain {
	res := make([]nodeGain, len(side))
	for i, v := range side {
		res[i] = nodeGain{pos: i, gain: moveGain(out[v], deg1, deg2, n1, n2, fromLeft)}
	}
	return res
}

// moveGain is the change in the split's log-gap cost from flipping node v's side,
// summed over v's out-neighbor terms. A positive gain means the move lowers the
// cost, so v wants to cross the cut. Only v's own terms change degree when v
// moves, so the gain is O(degree(v)) to compute, which is what makes an iteration
// linear in the edges touched.
//
// The cost a term contributes is the spec's surrogate deg * log2(n / (deg + 1)):
// a term concentrated on one side is cheap, a term split evenly is expensive,
// because a split term means edges that cross the cut and cross-cut edges are the
// long gaps the ordering is trying to avoid.
func moveGain(terms []int32, deg1, deg2 map[int32]int, n1, n2 int, fromLeft bool) float64 {
	var g float64
	for _, t := range terms {
		d1 := deg1[t]
		d2 := deg2[t]
		before := costTerm(d1, n1) + costTerm(d2, n2)
		var after float64
		if fromLeft {
			after = costTerm(d1-1, n1) + costTerm(d2+1, n2)
		} else {
			after = costTerm(d1+1, n1) + costTerm(d2-1, n2)
		}
		g += before - after
	}
	return g
}

// costTerm is one term's contribution to the split cost, deg * log2(n/(deg+1)),
// zero when the term has no nodes on the side.
func costTerm(deg, n int) float64 {
	if deg <= 0 {
		return 0
	}
	return float64(deg) * math.Log2(float64(n)/float64(deg+1))
}
