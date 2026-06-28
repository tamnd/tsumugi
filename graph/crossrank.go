package graph

import "math"

// This file is the global-then-scatter PageRank loop doc 07 specifies for the sharded
// corpus graph (the "Per-shard versus global" section, L954-994). PageRank needs the
// whole graph, cross-shard links included: a page's rank depends on the rank of
// everything that links to it, and at 2B docs across 100k shards almost all of those
// linkers live in other shards (doc 06). Computing PageRank shard by shard in isolation
// would drop the cross-shard in-links and underrate every page linked across a boundary,
// which is most well-linked pages. So the iteration is global, joined across shards
// through the cross-shard edge list and the inbound routing slice 41 built, even though
// every shard keeps only its own rank vector resident.
//
// The flow is the doc's: each round, every shard runs the local update over its own
// streamed transpose, the cross-shard rank-shares arrive through the precomputed inbound
// message buffers, the global L1 delta is the sum of the per-shard deltas, and the
// scatter is implicit because each shard accumulated its own next vector throughout, so
// when the global iteration converges every shard already holds the final rank for
// exactly its own nodes. The resident set is the per-shard rank vectors and out-degree
// arrays, node-proportional and independent of the edge count, the same asymmetry the
// single-shard StreamPageRank rests on, extended across the shard set.

// crossMsg is one resolved cross-shard rank-share for the per-iteration exchange. The
// source is named by its shard and dense docID, so the loop reads rank[srcShard][srcDense]
// and divides by outdeg[srcShard][srcDense] directly, with no global-id lookup on the hot
// path; the target is the local dense docID the share lands on in the receiving shard.
// It is the rank-vector twin of InboundEdge: RouteCrossEdges keeps the source as a global
// id because the offline edge set is corpus-stable and shard-independent, but the loop
// reads a rank vector indexed by shard and dense, so it resolves the source all the way to
// (shard, dense) once, before the iterations, and the exchange is then a plain array read.
type crossMsg struct {
	srcShard int
	srcDense int
	dstDense int
}

// routeCrossMessages resolves every shard's cross-shard out-edges into per-target-shard
// inbound message buffers, the precomputed routing the loop replays each iteration. It
// walks each shard's cross-shard edge list in source order and, for every far target,
// finds the shard that owns it through that shard's Dense lookup (the id table), recording
// a message there whose source is this shard and dense source and whose target is the
// resolved local dense docID. A target no shard in the set holds is dropped, the same
// way RouteCrossEdges leaves an edge to an absent shard for a later pass; here it simply
// does not contribute to the rank. The reference router scans the shard set per target,
// which the offline routing comment on RouteCrossEdges already notes the production path
// replaces with the global-id directory; the resolved messages are identical either way.
func routeCrossMessages(shards []*Region) [][]crossMsg {
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
						msgs[ti] = append(msgs[ti], crossMsg{srcShard: si, srcDense: source, dstDense: d})
						break
					}
				}
			}
		})
	}
	return msgs
}

// crossOutDegrees builds each shard's full out-degree array, the intra forward out-edges
// plus the cross-shard far out-edges, the divisor a sender's rank-share uses. A node's
// rank is split over all of its out-edges, near and far alike, so the cross targets must
// count: were the far edges left out, a page with cross-shard out-links would divide its
// rank by too small a number and over-contribute to its intra neighbors. The forward
// degree comes from OutDegrees in one pass, then the cross-shard list adds its per-source
// counts, the same list routeCrossMessages walks.
func crossOutDegrees(shards []*Region) [][]uint32 {
	deg := make([][]uint32, len(shards))
	for si, s := range shards {
		if s == nil {
			continue
		}
		d := OutDegrees(s)
		if s.xs != nil {
			s.xs.forEach(func(source int, targets []uint64) {
				d[source] += uint32(len(targets))
			})
		}
		deg[si] = d
	}
	return deg
}

// UniformCrossTeleport is the teleport vector of plain PageRank over the sharded graph:
// every node across every shard equally likely, the whole thing summing to one. It is the
// per-shard twin of UniformTeleport, split into one slice a shard, each entry one over the
// global node count, so the base term is globally normalized even though it is stored per
// shard. A shard the set leaves nil (a routing pass not holding it) gets a nil teleport.
func UniformCrossTeleport(shards []*Region) [][]float32 {
	total := 0
	for _, s := range shards {
		if s != nil {
			total += s.nodeCount
		}
	}
	p := make([][]float32, len(shards))
	if total == 0 {
		return p
	}
	u := float32(1.0 / float64(total))
	for si, s := range shards {
		if s == nil {
			continue
		}
		v := make([]float32, s.nodeCount)
		for i := range v {
			v[i] = u
		}
		p[si] = v
	}
	return p
}

// StreamCrossPageRank computes PageRank over a sharded graph out of core: every shard
// keeps only its own float32 rank and out-degree vectors resident and streams its own
// transpose, and the cross-shard in-links arrive each iteration through the precomputed
// inbound routing. On a graph that happens to fit one shard it agrees with StreamPageRank
// to float32 precision, and on a graph split across shards it agrees with StreamPageRank
// over the merged graph: the partition changes the working set, not the result. It is the
// uniform-teleport case of StreamCrossPageRankP.
func StreamCrossPageRank(shards []*Region, cfg PRConfig) [][]float32 {
	return StreamCrossPageRankP(shards, UniformCrossTeleport(shards), cfg)
}

// StreamCrossPageRankP is the personalized global-then-scatter power iteration: the same
// cross-shard structure as StreamCrossPageRank, but the random surfer teleports onto an
// arbitrary distribution p (one slice a shard, summing to one across the whole set)
// instead of the uniform one. A uniform p gives plain PageRank; a p biased onto a seed
// set gives sharded TrustRank, the cross-shard twin of StreamPageRankP. It follows doc
// 07's flow exactly:
//
//   - the dangling mass is a global sum over every shard's current rank vector, taken in
//     a pre-scan before the per-shard loop, never over the next vectors being written;
//   - the teleport-plus-dangle factor (1-alpha)+alpha*dm is one scalar an iteration,
//     hoisted out of every per-node loop, computed once from the global dangling mass;
//   - each shard gathers its intra-shard in-links from its own streamed transpose and its
//     cross-shard in-links from its inbound message buffer, both reading the previous
//     iteration's rank vectors, so the round is a Jacobi step over the whole corpus;
//   - the global L1 delta is the sum of the per-shard deltas, so convergence is decided
//     globally, correct because a slow-mixing region in one shard can keep the corpus rank
//     moving after fast regions have settled.
//
// The result is one rank vector a shard, each over its own dense docID space: the scatter
// is implicit because each shard accumulated its own next vector every round, so when the
// global iteration converges the rank already sits where it is needed.
func StreamCrossPageRankP(shards []*Region, p [][]float32, cfg PRConfig) [][]float32 {
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

	outdeg := crossOutDegrees(shards)
	msgs := routeCrossMessages(shards)

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
		// Global dangling mass: sum the current rank of every zero-out-degree node
		// across all shards, before any next vector is written.
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

			// Cross-shard inbound: scatter each arriving rank-share into the target's
			// accumulator. The shares read the previous iteration's rank vectors of the
			// source shards, so the whole round is one Jacobi step.
			acc := make([]float64, nc[si])
			for _, m := range msgs[si] {
				acc[m.dstDense] += float64(rank[m.srcShard][m.srcDense]) / float64(outdeg[m.srcShard][m.srcDense])
			}

			// Intra-shard gather: stream this shard's transpose, adding each in-neighbor's
			// rank-share. A node that is an in-neighbor has that out-edge, so its
			// out-degree is non-zero and the division is safe.
			for v := 0; v < nc[si]; v++ {
				a := acc[v]
				for _, u := range s.InNeighbors(v) {
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

// SeedCrossTeleport is the sharded teleport biased onto a seed set, each seed sharing the
// global teleport mass equally and every other node zero, the cross-shard twin of
// SeedTeleport. Seeds are named per shard, seeds[si] the local dense seed docIDs of shard
// si, so a TrustRank seed set spanning shards spreads its mass across them. With no seeds
// anywhere it falls back to the uniform teleport, so a seed-biased sharded rank with an
// empty seed set degenerates to plain PageRank, the same way the single-shard variant
// does. Out-of-range seed ids are ignored.
func SeedCrossTeleport(shards []*Region, seeds [][]int) [][]float32 {
	count := 0
	for _, s := range seeds {
		count += len(s)
	}
	if count == 0 {
		return UniformCrossTeleport(shards)
	}
	p := make([][]float32, len(shards))
	w := float32(1.0 / float64(count))
	for si, s := range shards {
		if s == nil {
			continue
		}
		v := make([]float32, s.nodeCount)
		if si < len(seeds) {
			for _, d := range seeds[si] {
				if d >= 0 && d < s.nodeCount {
					v[d] = w
				}
			}
		}
		p[si] = v
	}
	return p
}
