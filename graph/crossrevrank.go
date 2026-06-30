package graph

import "math"

// This file is the reversed-graph twin of crossrank.go's global-then-scatter loop. The
// build's inverse PageRank and Anti-TrustRank rank the reversed corpus graph: rank flows
// backward along links, a page scored by how well it reaches the rest of the graph
// (inverse PageRank, the inverse-PageRank trust-seed candidates) or by how much distrust
// reaches it from a spam seed set (Anti-TrustRank). In-core, graphbuild.go computes both
// as StreamPageRankP over ReverseSource(g) with OutDegreesFromSource(n, g.InDegree). At
// 2B docs across 100k shards the same merged-graph dependency that forced PageRank
// out-of-core forces these too, and for the same reason: almost every link that reaches a
// page crosses a shard boundary (doc 06), so a per-shard reversed rank in isolation would
// drop the cross-shard out-links the reverse flows along and misrank every page that links
// across a boundary. This is the cross-shard reversed rank, the missing primitive that
// lets the trust/anti-trust/inverse-PageRank family compute off the persisted per-shard
// regions instead of a second full-corpus in-core graph build (doc 07, the build-reorder
// the M15 work names).
//
// The reversal is exactly the one ReverseSource makes for the single-shard case, lifted
// across the shard set:
//
//   - the divisor is the reversed out-degree, which is the original in-degree: a node
//     splits its rank over the edges that point at it, intra and cross alike, so the
//     out-degree array is crossInDegrees (the intra in-degree plus the resolved cross
//     inbound count), the cross-shard twin of OutDegreesFromSource(n, g.InDegree);
//   - the intra gather reads each node's forward out-list (OutNeighbors), because a node's
//     in-neighbors on the reverse are its out-neighbors on the original, the same swap
//     reverseSource.InNeighbors makes;
//   - the cross exchange flows backward: a forward cross edge source->target carries rank
//     from target back to source on the reverse, so the message lands at the source's own
//     shard, the transpose of routeCrossMessages's forward routing;
//   - the dangling set is the nodes with zero reversed out-degree, the ones nothing links
//     to (original in-degree zero), whose rank routes through the global teleport-plus-
//     dangle base exactly as a forward dangling node's does.
//
// Everything else is crossrank.go's loop unchanged: global dangling mass in a pre-scan,
// the teleport-plus-dangle factor hoisted to one scalar an iteration, a Jacobi step
// reading the previous round's rank vectors, and a global L1 delta deciding convergence.

// crossInDegrees builds each shard's full in-degree array, the intra in-edges plus the
// resolved cross-shard inbound edges, the divisor a sender's rank-share uses on the
// reversed graph. A node's reversed out-degree is its original in-degree, and a page's
// in-links are mostly cross-shard, so the cross inbound count must be added: were it left
// out, a page with cross-shard in-links would divide its rank by too small a number on the
// reverse and over-contribute backward. The intra in-degree comes from each region's
// InDegree in one pass, then the resolved forward routing adds its per-target counts, the
// same messages routeCrossMessages walks (every forward cross edge is one inbound edge at
// its target), so the cross inbound count is read off that routing rather than re-walking
// the edge lists.
func crossInDegrees(shards []*Region) [][]uint32 {
	deg := make([][]uint32, len(shards))
	for si, s := range shards {
		if s == nil {
			continue
		}
		d := make([]uint32, s.nodeCount)
		for v := 0; v < s.nodeCount; v++ {
			d[v] = uint32(s.InDegree(v))
		}
		deg[si] = d
	}
	fwd := routeCrossMessages(shards)
	for ti := range fwd {
		for _, m := range fwd[ti] {
			deg[ti][m.dstDense]++
		}
	}
	return deg
}

// routeCrossMessagesReversed resolves every shard's cross-shard out-edges into inbound
// message buffers for the reversed flow. It walks the same forward cross edge list
// routeCrossMessages walks, but where the forward router lands a message at the target's
// shard (rank flowing source->target), the reversed router lands it at the source's own
// shard, carrying the target's rank back along the reversed edge (rank flowing
// target->source). The source the loop reads is the far target, named by its resolved
// (shard, dense) so the iteration reads rank[srcShard][srcDense] and divides by its
// reversed out-degree directly; the destination is this shard's local dense source. A
// target no shard in the set holds is dropped, the same as the forward router.
func routeCrossMessagesReversed(shards []*Region) [][]crossMsg {
	msgs := make([][]crossMsg, len(shards))
	for si, s := range shards {
		if s == nil || s.xs == nil {
			continue
		}
		s.xs.forEach(func(source int, targets []uint64) {
			for _, g := range targets {
				for ti, t := range shards {
					if ti == si || t == nil {
						continue
					}
					if d, ok := t.Dense(g); ok {
						msgs[si] = append(msgs[si], crossMsg{srcShard: ti, srcDense: d, dstDense: source})
						break
					}
				}
			}
		})
	}
	return msgs
}

// StreamCrossReversedPageRank computes PageRank over the reversed sharded graph out of
// core: every shard keeps only its own float32 rank and in-degree vectors resident and
// streams its own forward adjacency (its out-lists are the reverse's in-lists), and the
// cross-shard out-links arrive each iteration through the reversed inbound routing. On a
// graph that happens to fit one shard it agrees with StreamPageRank over ReverseSource to
// float32 precision, and on a graph split across shards it agrees with that reversed rank
// over the merged graph: the partition changes the working set, not the result. It is the
// inverse-PageRank case, the uniform-teleport form of StreamCrossReversedPageRankP, the
// cross-shard twin of graphbuild.go's streamInversePageRank.
func StreamCrossReversedPageRank(shards []*Region, cfg PRConfig) [][]float32 {
	return StreamCrossReversedPageRankP(shards, UniformCrossTeleport(shards), cfg)
}

// StreamCrossReversedPageRankP is the personalized global-then-scatter power iteration
// over the reversed sharded graph: the same cross-shard structure as
// StreamCrossReversedPageRank, but the surfer teleports onto an arbitrary distribution p
// (one slice a shard, summing to one across the set) instead of the uniform one. A
// uniform p gives inverse PageRank; a p biased onto a spam seed set gives sharded
// Anti-TrustRank, distrust flowing backward from the spam pages to the pages that link to
// them, the cross-shard twin of graphbuild.go's streamAntiTrustRank. It reuses
// StreamCrossPageRankP's flow with the three reversals this file's header lists: the
// divisor is crossInDegrees, the intra gather reads OutNeighbors, and the cross exchange
// uses the reversed routing. The result is one rank vector a shard, each over its own
// dense docID space, the scatter implicit because each shard accumulated its own next
// vector every round.
func StreamCrossReversedPageRankP(shards []*Region, p [][]float32, cfg PRConfig) [][]float32 {
	ns := len(shards)
	nc := make([]int, ns)
	total := 0
	for si, s := range shards {
		if s != nil {
			nc[si] = s.nodeCount
			total += s.nodeCount
		}
	}
	if total == 0 {
		return make([][]float32, ns)
	}

	// The reversed out-degree is the original in-degree, and the cross exchange flows
	// backward, so the only two structures that differ from the forward loop are the
	// divisor array and the inbound message routing.
	outdeg := crossInDegrees(shards)
	msgs := routeCrossMessagesReversed(shards)

	rank := make([][]float32, ns)
	rn := make([][]float32, ns)
	for si, s := range shards {
		if s == nil {
			continue
		}
		rank[si] = make([]float32, nc[si])
		rn[si] = make([]float32, nc[si])
		copy(rank[si], p[si])
	}

	for it := 0; it < cfg.MaxIters; it++ {
		// Global dangling mass: sum the current rank of every node with zero reversed
		// out-degree (original in-degree zero) across all shards, before any next vector
		// is written.
		var dm float64
		for si, s := range shards {
			if s == nil {
				continue
			}
			od := outdeg[si]
			rs := rank[si]
			for v := 0; v < nc[si]; v++ {
				if od[v] == 0 {
					dm += float64(rs[v])
				}
			}
		}
		factor := (1 - cfg.Alpha) + cfg.Alpha*dm

		var delta float64
		for si, s := range shards {
			if s == nil {
				continue
			}
			od := outdeg[si]
			own := rank[si]
			out := rn[si]
			ps := p[si]

			// Cross-shard inbound on the reverse: each arriving share is a far node's
			// rank divided by its reversed out-degree, read from the previous iteration's
			// rank vectors so the round is one Jacobi step.
			acc := make([]float64, nc[si])
			for _, m := range msgs[si] {
				acc[m.dstDense] += float64(rank[m.srcShard][m.srcDense]) / float64(outdeg[m.srcShard][m.srcDense])
			}

			// Intra-shard gather on the reverse: a node's in-neighbors on the reverse are
			// its forward out-neighbors, so stream this shard's out-lists. An out-neighbor
			// u has an in-edge here, so its in-degree (its reversed out-degree) is non-zero
			// and the division is safe.
			for v := 0; v < nc[si]; v++ {
				a := acc[v]
				for _, u := range s.OutNeighbors(v) {
					a += float64(own[u]) / float64(od[u])
				}
				nv := float32(float64(ps[v])*factor + cfg.Alpha*a)
				delta += math.Abs(float64(nv) - float64(own[v]))
				out[v] = nv
			}
		}

		rank, rn = rn, rank
		if delta < cfg.Tol {
			break
		}
	}
	return rank
}
