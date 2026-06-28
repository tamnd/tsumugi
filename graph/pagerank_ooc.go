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

// OutDegreesFromSource fills the resident out-degree array from a callback that
// reports one node's out-degree, the same uint32-per-node array OutDegrees builds
// off a *Region. It exists so an mmap-backed InNeighborSource supplies the array
// without a *Region in hand and without materializing the forward adjacency: a
// disk-backed source reads only each node's degree code, never expanding the list.
func OutDegreesFromSource(n int, outDegree func(v int) int) []uint32 {
	out := make([]uint32, n)
	for v := 0; v < n; v++ {
		out[v] = uint32(outDegree(v))
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
// base term. It is the uniform-teleport case of StreamPageRankP.
func StreamPageRank(src InNeighborSource, outdeg []uint32, cfg PRConfig) []float32 {
	n := src.NodeCount()
	if n == 0 {
		return nil
	}
	return StreamPageRankP(src, outdeg, UniformTeleport(n), cfg)
}

// StreamPageRankP is the personalized out-of-core power iteration: the same streaming
// structure as StreamPageRank, but the random surfer teleports onto an arbitrary
// distribution p (the teleport vector, which must sum to one) instead of the uniform
// one. A uniform p gives plain PageRank; a p biased onto a seed set gives TrustRank
// over the forward graph or, over the reversed graph, the inverse and anti-trust
// ranks. It is the streaming twin of signals.go's powerIter: it computes the identical
// update
//
//	r'[v] = (1-alpha)*p[v] + alpha*(acc + dm*p[v]) = p[v]*((1-alpha)+alpha*dm) + alpha*acc
//
// where dm is the dangling mass and acc the in-neighbor gather, so it agrees with the
// in-core seed-biased ranks to float32 precision. The teleport-plus-dangle factor
// ((1-alpha)+alpha*dm) is one scalar an iteration, hoisted out of the per-node loop;
// the only per-node teleport work is the one multiply p[v]*factor. The resident set
// is the two float32 rank vectors, the out-degree array, and the teleport vector,
// node-proportional and independent of the edge count.
func StreamPageRankP(src InNeighborSource, outdeg []uint32, p []float32, cfg PRConfig) []float32 {
	n := src.NodeCount()
	if n == 0 {
		return nil
	}
	r := make([]float32, n)
	rn := make([]float32, n)
	// Start from the teleport vector, matching powerIter's copy(r, p): the first
	// iteration's gather then reads the seed-biased initial distribution.
	copy(r, p)
	for it := 0; it < cfg.MaxIters; it++ {
		// Pre-scan the dangling mass from the resident vector, before the loop,
		// because it sums the current r, not the next one being written.
		var dm float64
		for v := 0; v < n; v++ {
			if outdeg[v] == 0 {
				dm += float64(r[v])
			}
		}
		factor := (1 - cfg.Alpha) + cfg.Alpha*dm

		var delta float64
		for v := 0; v < n; v++ {
			var acc float64
			for _, u := range src.InNeighbors(v) {
				acc += float64(r[u]) / float64(outdeg[u])
			}
			nv := float32(float64(p[v])*factor + cfg.Alpha*acc)
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

// UniformTeleport is the teleport vector of plain PageRank: every node equally
// likely, summing to one. It is the float32 streaming twin of signals.go's
// uniformVec.
func UniformTeleport(n int) []float32 {
	p := make([]float32, n)
	u := float32(1.0 / float64(n))
	for i := range p {
		p[i] = u
	}
	return p
}

// SeedTeleport is the teleport vector biased onto seeds, each seed sharing the mass
// equally and every other node zero, the float32 streaming twin of signals.go's
// seedVec. With no seeds it falls back to uniform, so a seed-biased streamed rank
// with an empty seed set degenerates to plain PageRank exactly as the in-core variant
// does. Out-of-range seed ids are ignored.
func SeedTeleport(n int, seeds []int) []float32 {
	if len(seeds) == 0 {
		return UniformTeleport(n)
	}
	p := make([]float32, n)
	w := float32(1.0 / float64(len(seeds)))
	for _, s := range seeds {
		if s >= 0 && s < n {
			p[s] = w
		}
	}
	return p
}

// reverseSource adapts a *Region into the reversed graph as an InNeighborSource: a
// node's in-neighbors on the reverse are its out-neighbors on the original. It is the
// streaming equivalent of signals.go's buildCSRRev, and like that function it never
// copies the adjacency: each call decodes one out-list from the region on demand. The
// reverse out-degree a sender divides by is the original in-degree, which the caller
// supplies as the resident out-degree array (OutDegreesFromSource(n, g.InDegree)).
// Inverse PageRank and Anti-TrustRank are plain and seed-biased StreamPageRankP runs
// over this source.
type reverseSource struct{ g *Region }

// ReverseSource returns g viewed as its own transpose, an InNeighborSource whose
// in-lists are g's out-lists. The reversed view holds none of the adjacency resident:
// it forwards to the region, which decodes one list per call.
func ReverseSource(g *Region) InNeighborSource { return reverseSource{g} }

func (s reverseSource) NodeCount() int          { return s.g.nodeCount }
func (s reverseSource) InNeighbors(v int) []int { return s.g.OutNeighbors(v) }
