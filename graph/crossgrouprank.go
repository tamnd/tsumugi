package graph

import "sort"

// This file is the host and domain rank doc 07 specifies for the sharded corpus graph
// (the "Per-shard versus global" section, L985-994). The page-level link signals
// (PageRank, TrustRank) are global iterations with a per-iteration cross-shard exchange,
// the loop crossrank.go runs. The host and domain ranks are global too, but the doc is
// explicit that they are computed differently and more cheaply: the page graph is
// projected onto the host (or domain) graph across all shards, that small graph's
// PageRank runs once in RAM because the host and domain graphs are tiny next to the page
// graph, and then each page inherits its host's and domain's rank, scattered back to the
// page's shard. There is no per-iteration cross-shard exchange here: the projection spans
// shards once, the iteration is local to the small contracted graph, and the scatter is
// one pass.
//
// The projection is what makes this a sharded computation rather than a single-Region one.
// aggregateRank in signals.go projects the page graph onto the group graph by reading one
// Region's whole adjacency with a dense-docID-keyed group map; that needs the merged page
// graph resident in one address space, which at 2B docs does not exist. StreamGroupRank
// projects from the shard set instead: it walks each shard's own intra adjacency and its
// cross-shard edge list, mapping every edge's two endpoints to their group through the
// global node id, so the page graph is never materialized whole. The group of a node comes
// from its corpus-stable global id, which the host and domain partition (the build's
// AssignGlobalIDs) lays out so a host's pages share the high bits, so the caller's group
// function is a shift for host rank and a small per-host lookup for domain rank. The
// result is byte-for-byte the same group graph aggregateRank builds over the merged graph,
// only assembled from the distributed shards, so the ranks agree to floating-point
// precision whatever the partition; the partition changes the working set, not the result.

// StreamGroupRank projects the page graph across a shard set onto a group graph, ranks the
// small group graph in RAM, and scatters each group's rank back to its pages, returning one
// rank vector a shard over that shard's dense docID space. groupOfGlobal maps a node's
// global id to its group id (the group ids need not be dense, they are compacted here): for
// host rank it extracts the host group from the partition id, for domain rank it maps that
// host group to its domain. It is the sharded form of HostRank and DomainRank, computing the
// same contracted-graph PageRank over the same inter-group edge weights without ever holding
// the merged page graph, so a graph that fits one shard agrees with aggregateRank exactly and
// a graph split across shards agrees with aggregateRank over the merged graph.
//
// Like aggregateRank it drops intra-group edges (a host's internal links do not vote for the
// host, the property that makes host rank resist internal-link spam) and weights an
// inter-group edge by the number of page edges that cross between the two groups. A cross-shard
// page edge is one such edge whose target lives in another shard; it is read from the source
// shard's cross-shard list and counted exactly once, the same single count a merged-graph
// projection would give it, so intra and cross edges together reproduce every page edge once.
func StreamGroupRank(shards []*Region, groupOfGlobal func(uint64) int, cfg PRConfig) [][]float64 {
	ns := len(shards)
	out := make([][]float64, ns)

	groups, m, inEdges, outW := crossGroupGraph(shards, groupOfGlobal)
	if m == 0 {
		return out
	}

	gr := weightedPageRank(m, inEdges, outW, cfg)

	// Scatter: each page inherits its group's rank, written into the page's own shard, the
	// one cheap pass doc 07 calls for. The implicit per-shard layout falls out of writing
	// out[si][d] directly.
	for si, s := range shards {
		if s == nil {
			continue
		}
		gs := groups[si]
		v := make([]float64, len(gs))
		for d := range gs {
			v[d] = gr[gs[d]]
		}
		out[si] = v
	}
	return out
}

// crossGroupGraph projects the page graph across a shard set onto the contracted group graph,
// the shared cross-shard projection both StreamGroupRank and CrossHostLinkDiversity run before
// they reduce the group graph (the one to a weighted PageRank, the other to per-host entropy).
// It is the cross-shard twin of signals.go's projectGroups: it compacts the group ids to a
// dense range registering every page's group first, projects the page edges onto the group
// graph accumulating inter-group edge weights while dropping intra-group edges, and returns the
// per-shard cached dense group ids, the group count, and the weighted in-edges and out-weights
// in a deterministic edge order. It never materializes the merged page graph; an edge's two
// endpoints map to their group through the corpus-stable global id, the intra target through the
// shard's own cached group and the far target through the group closure on its global id.
func crossGroupGraph(shards []*Region, groupOfGlobal func(uint64) int) (groups [][]int, m int, inEdges [][]wedge, outW []float64) {
	ns := len(shards)

	// Compact the group ids to a dense range. Register every page's group first, in a full
	// pass over every shard's nodes, so an isolated group (a host with pages but no links
	// that cross its own boundary) is still a node in the group graph and still inherits a
	// teleport-only rank, matching aggregateRank's register-every-node-first projection.
	idOf := map[int]int{}
	groupDense := func(global uint64) int {
		g := groupOfGlobal(global)
		d, ok := idOf[g]
		if !ok {
			d = len(idOf)
			idOf[g] = d
		}
		return d
	}
	// Cache each shard's per-node dense group id while registering, so the projection and the
	// scatter do not call groupOfGlobal again (it is a user closure, and the cache also pins
	// the dense id a page scatters to even if its group is otherwise only seen via a far edge).
	groups = make([][]int, ns)
	for si, s := range shards {
		if s == nil {
			continue
		}
		n := s.nodeCount
		gs := make([]int, n)
		for d := 0; d < n; d++ {
			gs[d] = groupDense(s.Global(d))
		}
		groups[si] = gs
	}
	m = len(idOf)
	if m == 0 {
		return groups, 0, nil, nil
	}

	// Project the page edges onto the group graph, accumulating inter-group edge weights and
	// dropping intra-group edges. Both an intra edge's target (a local dense docID) and a
	// cross edge's target (a far global id) resolve to a group: the intra target through the
	// shard's own cached group, the far target through groupDense on its global id, which may
	// add a group with no page in this set (a host linked to but not loaded), a real node in
	// the group graph all the same.
	type key struct{ u, v int }
	w := map[key]float64{}
	for si, s := range shards {
		if s == nil {
			continue
		}
		gs := groups[si]
		n := s.nodeCount
		for d := 0; d < n; d++ {
			gu := gs[d]
			for _, t := range s.OutNeighbors(d) {
				if gv := gs[t]; gv != gu {
					w[key{gu, gv}]++
				}
			}
		}
		s.ForEachCrossEdge(func(source int, targets []uint64) {
			gu := gs[source]
			for _, g := range targets {
				if gv := groupDense(g); gv != gu {
					w[key{gu, gv}]++
				}
			}
		})
	}

	// Build the contracted graph in a fixed key order so the result is deterministic: a map
	// range would feed the reducer its in-edges in a different float summation order each run,
	// which drifts the result in the last bits and across platforms. The group graph is tiny,
	// so sorting its edges costs nothing.
	keys := make([]key, 0, len(w))
	for k := range w {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].v != keys[j].v {
			return keys[i].v < keys[j].v
		}
		return keys[i].u < keys[j].u
	})
	inEdges = make([][]wedge, m)
	outW = make([]float64, m)
	for _, k := range keys {
		weight := w[k]
		inEdges[k.v] = append(inEdges[k.v], wedge{u: k.u, w: weight})
		outW[k.u] += weight
	}
	return groups, m, inEdges, outW
}
