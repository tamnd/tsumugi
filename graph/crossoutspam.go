package graph

// This file is the cross-shard form of OutboundSpamRatio, the last non-rank link signal
// signals.go computes over a single merged graph. OutboundSpamRatio is the fraction of a node's
// out-links that point at a page whose SpamMass exceeds the threshold: it is the local,
// per-page companion of AntiTrustRank, the immediate count of how much of a page's own linking
// aims at known spam, one hop rather than the propagated distrust signal. The two together catch
// both the page that links to spam and the page upstream of it.
//
// Like reciprocity it is an out-edge aggregation, but where reciprocity asks whether each
// out-neighbor links back, this asks whether each out-neighbor is spam. The merged form reads
// a node's out-neighbors from one Region's adjacency and looks each one up in the dense-indexed
// SpamMass vector, which needs the merged page graph resident; at 2B docs across 100k shards
// almost every out-link crosses a shard boundary (doc 06), so the out-neighbor set is assembled
// from the shard set instead: a node's intra out-neighbors from its own shard's adjacency and
// its far out-neighbors from the cross-shard edge list, the same two-source out-edge gather
// reciprocity uses. The SpamMass of a target comes from its corpus-stable global id through the
// caller's spam function, the cross-shard twin of the merged form's dense-indexed lookup; the
// SpamMass vectors themselves are the per-shard output the cross-shard reversed Anti-TrustRank
// (slice 86) now produces off the persisted regions, so the spam function reads a per-shard
// vector resolved by global id, never a merged one.

// CrossOutboundSpamRatio returns, per node, the fraction of its out-links that point at a page
// whose SpamMass exceeds the threshold, the sharded form of OutboundSpamRatio. spamOfGlobal maps
// a node's global id to its SpamMass, so every out-neighbor (intra resolved to a global id
// through its own shard's id table, far as the global target id the cross-shard edge list
// carries) is tested uniformly in the one id space the shards share; the denominator is the full
// out-degree, intra plus far, the same crossOutDegrees the rank loop and reciprocity divide by.
// On a graph that fits one shard it equals OutboundSpamRatio, and over a partition it equals
// OutboundSpamRatio over the merged graph: the boundary-crossing out-links the merged form reads
// from the resident adjacency are recovered here from the cross-shard edge list, each target's
// SpamMass found by global id whatever shard holds it. A node with no out-links has ratio zero.
func CrossOutboundSpamRatio(shards []*Region, spamOfGlobal func(uint64) float64, threshold float64) [][]float64 {
	outdeg := crossOutDegrees(shards)
	outbound := crossOutbound(shards)
	out := make([][]float64, len(shards))
	for si, s := range shards {
		if s == nil {
			continue
		}
		n := s.nodeCount
		res := make([]float64, n)
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
			hits := 0
			for _, u := range s.OutNeighbors(v) {
				if spamOfGlobal(s.Global(u)) > threshold {
					hits++
				}
			}
			if farOut != nil {
				for _, g := range farOut[v] {
					if spamOfGlobal(g) > threshold {
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
