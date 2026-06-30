package graph

// This file is the cross-shard form of Reciprocity, the one in-edge-and-out-edge link signal
// signals.go computes over a single merged graph. Reciprocity is the fraction of a node's
// out-links that point at a node that links back: |Out(v) intersect In(v)| / |Out(v)|, the
// mild link-farm tell that two pages cite each other. Unlike the pure inbound aggregations
// (in-degree, linking-domains/hosts), it reads BOTH sides of a node: every out-neighbor and
// the whole in-neighbor set, then intersects them. The merged form reads both from one
// Region's adjacency and transpose, which needs the merged page graph resident; at 2B docs
// across 100k shards almost every link to a page crosses a shard boundary (doc 06), so each
// side is assembled from the shard set instead. A node's out-neighbors are its intra
// out-neighbors (its own shard's adjacency) plus its far out-neighbors (the cross-shard edge
// list, where the targets are corpus-stable global ids); its in-neighbors are its intra
// in-neighbors (its own transpose) plus its far in-neighbors (the cross-shard routing). The
// intersection is taken over global ids, the one id space both sides share across shards: an
// intra neighbor through its own shard's id table, a far neighbor through the corpus-stable
// global id the edge list and the routing already carry. So the back-link from a far
// out-neighbor u to v, which lives in u's shard as a cross edge u->v, is recovered as one of
// v's far in-neighbors, and the global-id match finds it whatever shard holds u.

// crossOutbound buckets each shard's far out-targets (corpus-stable global ids) by their
// source node, the out-edge mirror of crossInbound. It walks the cross-shard edge list once
// per shard, so the per-node reciprocity pass reads one node's far out-edges at a time. The
// targets stay global ids because that is the id space the in-neighbor set is intersected in,
// and a far target need not be held by any shard in the set: a back-link from it would arrive
// as a far in-neighbor carrying the same global id, so the match is by id, not by residency.
func crossOutbound(shards []*Region) [][][]uint64 {
	buckets := make([][][]uint64, len(shards))
	for si, s := range shards {
		if s == nil {
			continue
		}
		b := make([][]uint64, s.nodeCount)
		if s.xs != nil {
			s.xs.forEach(func(source int, targets []uint64) {
				b[source] = append(b[source], targets...)
			})
		}
		buckets[si] = b
	}
	return buckets
}

// CrossReciprocity returns, per node, the fraction of its out-links that point at a node that
// links back, the sharded form of Reciprocity. On a graph that fits one shard it equals
// Reciprocity, and over a partition it equals Reciprocity over the merged graph: the
// boundary-crossing back-links Reciprocity reads from the merged transpose are recovered here
// from the cross-shard routing, matched against the cross-shard out-edges in the shared global
// id space. Each node's in-neighbor set is gathered once into a reused set of global ids (its
// intra in-neighbors through its own id table, its far in-neighbors through their source
// shard's id table), then every out-neighbor (intra through its own id table, far as the
// global target id the edge list carries) is tested for membership; the denominator is the
// full out-degree, intra plus far, the same crossOutDegrees the rank loop divides by. A node
// with no out-links has reciprocity zero. The seen set is cleared and reused across nodes so
// the pass allocates once, not a set a node.
func CrossReciprocity(shards []*Region) [][]float64 {
	outdeg := crossOutDegrees(shards)
	inbound := crossInbound(shards)
	outbound := crossOutbound(shards)
	out := make([][]float64, len(shards))
	seen := map[uint64]struct{}{}
	for si, s := range shards {
		if s == nil {
			continue
		}
		n := s.nodeCount
		res := make([]float64, n)
		var farIn [][]crossMsg
		if inbound[si] != nil {
			farIn = inbound[si]
		}
		var farOut [][]uint64
		if outbound[si] != nil {
			farOut = outbound[si]
		}
		od := outdeg[si]
		for v := 0; v < n; v++ {
			total := int(od[v])
			if total == 0 {
				continue
			}
			clear(seen)
			for _, u := range s.InNeighbors(v) {
				seen[s.Global(u)] = struct{}{}
			}
			if farIn != nil {
				for _, m := range farIn[v] {
					seen[shards[m.srcShard].Global(m.srcDense)] = struct{}{}
				}
			}
			hits := 0
			for _, u := range s.OutNeighbors(v) {
				if _, ok := seen[s.Global(u)]; ok {
					hits++
				}
			}
			if farOut != nil {
				for _, g := range farOut[v] {
					if _, ok := seen[g]; ok {
						hits++
					}
				}
			}
			res[v] = float64(hits) / float64(total)
		}
		out[si] = res
	}
	return out
}
