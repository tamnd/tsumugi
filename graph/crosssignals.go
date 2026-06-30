package graph

// This file is the cross-shard form of the in-edge link signals signals.go computes over a
// single merged graph: the in-degree (InDegrees) and the two distinct-source-group counts
// (LinkingDomains, LinkingHosts). All three are inbound aggregations, a gather over the set
// of edges that point at a node, reducing either to a count (in-degree) or to the number of
// distinct groups the sources fall into (linking domains, linking hosts). The merged forms
// read that inbound set from one Region's transpose, which needs the merged page graph
// resident; at 2B docs across 100k shards almost every in-link to a page crosses a shard
// boundary (doc 06), so the inbound set is assembled from the shard set instead: a node's
// intra in-neighbors from its own shard's transpose and its far in-neighbors from the
// cross-shard routing, the same two-source inbound gather the cross-shard reversed rank
// divides by. The group of a far source comes from its corpus-stable global id through the
// caller's group function, exactly as StreamGroupRank resolves a far edge's endpoint, so the
// result is the same count the merged form gives, only assembled from the distributed shards.

// CrossInDegrees returns each node's total in-degree across the shard set, one slice a shard
// over that shard's dense docID space. It is the sharded form of InDegrees: on a graph that
// fits one shard it equals InDegrees, and over a partition it equals InDegrees over the
// merged graph, the cross-shard in-links InDegrees reads from the merged transpose recovered
// here from the routing. It is the int-typed signal twin of the uint32 crossInDegrees array
// the reversed rank divides by (the intra in-degree plus the resolved cross inbound count,
// the two disjoint edge sets that together are every in-edge), the same count widened for
// the feature column.
func CrossInDegrees(shards []*Region) [][]int {
	deg := crossInDegrees(shards)
	out := make([][]int, len(shards))
	for si, d := range deg {
		if d == nil {
			continue
		}
		o := make([]int, len(d))
		for i, v := range d {
			o[i] = int(v)
		}
		out[si] = o
	}
	return out
}

// crossInbound buckets the routed cross-shard inbound edges by their target node, so a
// per-node aggregation reads one node's far in-edges at a time instead of scanning the flat
// message list once a node. Each entry is the source's (shard, dense), the same pair the rank
// loop reads, so the aggregation resolves the source's group through shards[srcShard].Global
// without a global-id lookup repeated per node.
func crossInbound(shards []*Region) [][][]crossMsg {
	msgs := routeCrossMessages(shards)
	buckets := make([][][]crossMsg, len(shards))
	for si, s := range shards {
		if s == nil {
			continue
		}
		b := make([][]crossMsg, s.nodeCount)
		for _, m := range msgs[si] {
			b[m.dstDense] = append(b[m.dstDense], m)
		}
		buckets[si] = b
	}
	return buckets
}

// crossDistinctInboundGroups returns, per node, the number of distinct groups among its
// in-neighbors across the shard set, the shared body of CrossLinkingDomains and
// CrossLinkingHosts. groupOfGlobal maps a source's global id to its group id (the group ids
// need not be dense, only comparable, since the reducer counts distinct values). Each node's
// inbound set is its intra in-neighbors (resolved to a global id through its own shard's id
// table) plus its far in-neighbors (resolved through their source shard's id table), the same
// two disjoint edge sets CrossInDegrees counts; a set deduplicates them so a group that links
// in both near and far counts once, matching the merged distinct count over the whole inbound
// set. The seen map is cleared and reused across nodes so the pass allocates once, not a map
// a node.
func crossDistinctInboundGroups(shards []*Region, groupOfGlobal func(uint64) int) [][]int {
	buckets := crossInbound(shards)
	out := make([][]int, len(shards))
	seen := map[int]struct{}{}
	for si, s := range shards {
		if s == nil {
			continue
		}
		n := s.nodeCount
		res := make([]int, n)
		var far [][]crossMsg
		if buckets[si] != nil {
			far = buckets[si]
		}
		for v := 0; v < n; v++ {
			clear(seen)
			for _, u := range s.InNeighbors(v) {
				seen[groupOfGlobal(s.Global(u))] = struct{}{}
			}
			if far != nil {
				for _, m := range far[v] {
					seen[groupOfGlobal(shards[m.srcShard].Global(m.srcDense))] = struct{}{}
				}
			}
			res[v] = len(seen)
		}
		out[si] = res
	}
	return out
}

// CrossLinkingDomains returns, per node, the number of distinct domains among its
// in-neighbors across the shard set, the sharded form of LinkingDomains. domainOfGlobal maps
// a source's global id to its domain id, the cross-shard twin of LinkingDomains's
// dense-indexed domainOf: where the merged form reads the domain of a dense in-neighbor, this
// reads the domain of a global source id, near and far alike, so over a partition it returns
// the same distinct-domain count LinkingDomains returns over the merged graph. A farm can
// manufacture many links from a few domains but not from many independent ones, so this
// resists link spam far better than raw in-degree, the same robustness at corpus scale.
func CrossLinkingDomains(shards []*Region, domainOfGlobal func(uint64) int) [][]int {
	return crossDistinctInboundGroups(shards, domainOfGlobal)
}

// CrossLinkingHosts returns, per node, the number of distinct hosts among its in-neighbors
// across the shard set, the sharded form of LinkingHosts. hostOfGlobal maps a source's global
// id to its host id. It is the host-level companion of CrossLinkingDomains: a thousand
// in-links from one host count as one host's opinion repeated, not a thousand independent
// votes, so this resists the cheap single-host attack raw in-degree counts at face value,
// matching LinkingHosts over the merged graph whatever the partition.
func CrossLinkingHosts(shards []*Region, hostOfGlobal func(uint64) int) [][]int {
	return crossDistinctInboundGroups(shards, hostOfGlobal)
}
