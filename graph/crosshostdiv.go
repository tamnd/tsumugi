package graph

import "math"

// This file is the cross-shard form of HostLinkDiversity, the link signal signals.go computes
// over the contracted host graph rather than the page graph. HostLinkDiversity is the
// normalized entropy of a host's inbound source-host distribution: H(source host distribution
// of In(host)) / log(distinct source hosts). It refines the distinct-linking-host count
// (LinkingHosts says how many hosts link in, this says how evenly): a value near one means the
// inbound links spread evenly across many hosts, the natural pattern; near zero means they
// concentrate on one or two hosts, the farm pattern. It is computed on the same host graph the
// host-rank projection builds, where a host's weighted in-edges are exactly its source-host
// distribution, so the entropy is a sum over those weights.
//
// The merged form reads that contracted graph from one Region's adjacency through projectGroups,
// which needs the merged page graph resident; at 2B docs across 100k shards the merged page
// graph does not exist (doc 06), so the projection spans the shard set instead, the same
// cross-shard projection StreamGroupRank already runs for the host and domain ranks. The two
// signals share the contracted graph: the rank reduces it to a weighted PageRank, the diversity
// reduces it to per-host entropy, both over the identical inter-host edge weights, so the
// diversity agrees with the merged HostLinkDiversity whatever the partition.

// CrossHostLinkDiversity returns, per page, the normalized entropy of its host's inbound
// source-host distribution across the shard set, the sharded form of HostLinkDiversity.
// hostOfGlobal maps a node's global id to its host id, the same group function StreamGroupRank
// takes for host rank. It runs the cross-shard projection once to build the contracted host
// graph, computes each host's normalized entropy from its inbound inter-host edge weights, and
// scatters each page its host's value, the same inherit-from-host pass HostLinkDiversity makes.
// On a graph that fits one shard it equals HostLinkDiversity, and over a partition it equals
// HostLinkDiversity over the merged graph: the boundary-crossing inter-host links the merged
// projection reads from the resident page graph are recovered here from the cross-shard edge
// list. A host with one or zero distinct source hosts has diversity zero.
func CrossHostLinkDiversity(shards []*Region, hostOfGlobal func(uint64) int) [][]float64 {
	ns := len(shards)
	out := make([][]float64, ns)

	groups, m, inEdges, _ := crossGroupGraph(shards, hostOfGlobal)
	if m == 0 {
		return out
	}

	// Per-host normalized entropy of the inbound source-host weight distribution, the same
	// reduction HostLinkDiversity makes over projectGroups' in-edges. The sum is order
	// independent, so the cross-shard projection's sorted edge order gives the identical value.
	div := make([]float64, m)
	for v := 0; v < m; v++ {
		edges := inEdges[v]
		if len(edges) <= 1 {
			continue
		}
		var total float64
		for _, e := range edges {
			total += e.w
		}
		if total <= 0 {
			continue
		}
		var h float64
		for _, e := range edges {
			p := e.w / total
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		div[v] = h / math.Log(float64(len(edges)))
	}

	// Scatter: each page inherits its host's diversity, written into the page's own shard.
	for si, s := range shards {
		if s == nil {
			continue
		}
		gs := groups[si]
		v := make([]float64, len(gs))
		for d := range gs {
			v[d] = div[gs[d]]
		}
		out[si] = v
	}
	return out
}
