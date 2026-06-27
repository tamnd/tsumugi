package graph

import "math"

// This file is the out-of-core PageRank doc 07 specifies for the corpus graph,
// where the adjacency is hundreds of gigabytes but the rank vector is one float32 a
// node. It rests on that asymmetry: keep the rank vectors and the out-degree array
// resident and stream the transpose adjacency from disk, once an iteration, in
// dense-docID order. The in-core PageRank in signals.go decodes the whole transpose
// into a flat CSR once; this one re-streams it each iteration and never materializes
// it, so its working set is the resident vectors, not the graph.

// InNeighborSource streams in-neighbor lists in dense-docID order. A node's in-list
// is decoded on demand, so an implementation backed by an mmap'd region holds none
// of the adjacency resident; the iteration pulls one node's in-list at a time and
// treats the calls as a sequential scan. *Region satisfies it directly.
type InNeighborSource interface {
	NodeCount() int
	InNeighbors(v int) []int
}

// OutDegrees reads the out-degree of every node in one forward pass, the resident
// uint32-per-node array the out-of-core iteration divides each sender's rank by. The
// spec fills it once from the forward record degrees and never touches the forward
// adjacency again during the iteration.
func OutDegrees(g *Region) []uint32 {
	out := make([]uint32, g.nodeCount)
	for v := 0; v < g.nodeCount; v++ {
		out[v] = uint32(g.OutDegree(v))
	}
	return out
}

// StreamPageRank computes PageRank out of core: it streams the transpose adjacency
// one in-list at a time from src and keeps only the two rank vectors and the
// out-degree array resident, so it never holds the whole adjacency in memory. On the
// same graph it orders documents identically to the in-core PageRank and agrees with
// it to float32 precision; the difference is the working set, not the result.
//
// It follows the spec's doc 07 pseudocode exactly:
//   - the rank vectors are float32, half the resident memory of float64 and still
//     finer than the byte-quantized feature column ever keeps;
//   - the per-node gather accumulates in float64, so a long in-list does not lose
//     low-order bits as the accumulator grows;
//   - the dangling mass is a global sum over the current r, computed in a pre-scan
//     over the resident vector before the per-node loop, never over the in-progress
//     next vector;
//   - the teleport-plus-dangle base is one scalar an iteration, hoisted out of the
//     per-node loop so the only per-node work is the in-neighbor gather.
//
// A node that appears in some in-list has at least that one out-edge, so its
// out-degree is non-zero and the division is safe; dangling nodes, the ones with
// zero out-degree, never appear in any in-list and receive rank only through the
// base term.
func StreamPageRank(src InNeighborSource, outdeg []uint32, cfg PRConfig) []float32 {
	n := src.NodeCount()
	if n == 0 {
		return nil
	}
	r := make([]float32, n)
	rn := make([]float32, n)
	init := float32(1.0 / float64(n))
	for i := range r {
		r[i] = init
	}
	for it := 0; it < cfg.MaxIters; it++ {
		// Pre-scan the dangling mass from the resident vector, before the loop,
		// because it sums the current r, not the next one being written.
		var dm float64
		for v := 0; v < n; v++ {
			if outdeg[v] == 0 {
				dm += float64(r[v])
			}
		}
		base := (1-cfg.Alpha)/float64(n) + cfg.Alpha*dm/float64(n)

		var delta float64
		for v := 0; v < n; v++ {
			var acc float64
			for _, u := range src.InNeighbors(v) {
				acc += float64(r[u]) / float64(outdeg[u])
			}
			nv := float32(base + cfg.Alpha*acc)
			delta += math.Abs(float64(nv) - float64(r[v]))
			rn[v] = nv
		}
		r, rn = rn, r
		if delta < cfg.Tol {
			break
		}
	}
	return r
}
