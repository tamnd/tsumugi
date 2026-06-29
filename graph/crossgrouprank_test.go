package graph

import (
	"math"
	"testing"
)

// hostGraph builds a graph with explicit host and domain structure: domains hold hosts,
// hosts hold pages, and pages link mostly to pages on other hosts (the inter-host links
// host rank votes with) plus a few inside their own host (the intra-host links host rank
// must drop). It returns the edges over node indices ordered (domain, host, page), and the
// per-node host group, domain, and partition global id, the id a sharded run derives the
// group from. The global id is the build's partition layout, a host group prefix in the
// high bits over a per-page sequence in the low bits, so the host group is the id shifted
// right by seqBits and the domain is that host group divided by the hosts per domain.
func hostGraph(domains, hostsPer, pagesPer, seqBits int) (edges []prEdge, hostOf, domainOf []int, gids []uint64) {
	n := domains * hostsPer * pagesPer
	hostOf = make([]int, n)
	domainOf = make([]int, n)
	gids = make([]uint64, n)
	idx := func(d, h, p int) int { return ((d*hostsPer+h)*pagesPer + p) }
	for d := 0; d < domains; d++ {
		for h := 0; h < hostsPer; h++ {
			hg := d*hostsPer + h
			for p := 0; p < pagesPer; p++ {
				i := idx(d, h, p)
				hostOf[i] = hg
				domainOf[i] = d
				gids[i] = uint64(hg)<<seqBits | uint64(p)
			}
		}
	}
	// Deterministic links: each page points at a few pages on other hosts and one page on
	// its own host, so the projection has both inter-group weight to rank and intra-group
	// edges to drop.
	for d := 0; d < domains; d++ {
		for h := 0; h < hostsPer; h++ {
			for p := 0; p < pagesPer; p++ {
				i := idx(d, h, p)
				for k := 1; k <= 3; k++ {
					j := (i + k*(pagesPer*hostsPer-1) + 1) % n
					if j != i {
						edges = append(edges, prEdge{i, j})
					}
				}
				// one intra-host edge to the next page in the same host
				if pagesPer > 1 {
					edges = append(edges, prEdge{i, idx(d, h, (p+1)%pagesPer)})
				}
			}
		}
	}
	return edges, hostOf, domainOf, gids
}

// buildGroupShards splits the edge set across s shards round-robin, carrying the partition
// global ids as each shard's id table, the same split buildShards does but with the host
// and domain partition ids rather than the identity ids, so a sharded group rank derives a
// node's group from its global id exactly as the build does.
func buildGroupShards(t *testing.T, n int, edges []prEdge, gids []uint64, s int) ([]*Region, []int, []int) {
	t.Helper()
	shardOf, denseOf, members := partition(n, s)
	builders := make([]*Builder, s)
	for sh := 0; sh < s; sh++ {
		ids := make([]uint64, len(members[sh]))
		for d, idx := range members[sh] {
			ids[d] = gids[idx]
		}
		builders[sh] = NewBuilder(len(members[sh])).WithNodeIDs(ids)
	}
	for _, e := range edges {
		sa, sb := shardOf[e.a], shardOf[e.b]
		if sa == sb {
			builders[sa].AddEdge(denseOf[e.a], denseOf[e.b])
		} else {
			builders[sa].AddCrossEdge(denseOf[e.a], gids[e.b])
		}
	}
	regions := make([]*Region, s)
	for sh := 0; sh < s; sh++ {
		r, err := Open(builders[sh].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", sh, err)
		}
		regions[sh] = r
	}
	return regions, shardOf, denseOf
}

// TestStreamGroupRankMatchesHostRank is the gate the slice rests on: the sharded
// projection must give every page the same host rank and the same domain rank as the
// single-Region HostRank and DomainRank over the merged graph, whatever the partition. The
// sharded run reads a node's host group from its global id (the id shifted past the page
// sequence) and its domain from that host group, the same grouping the merged run gets from
// the hostOf and domainOf arrays, so any divergence is the projection losing or
// double-counting an edge that crosses a shard boundary.
func TestStreamGroupRankMatchesHostRank(t *testing.T) {
	const domains, hostsPer, pagesPer, seqBits = 5, 4, 6, 3
	edges, hostOf, domainOf, gids := hostGraph(domains, hostsPer, pagesPer, seqBits)
	n := domains * hostsPer * pagesPer

	cfg := DefaultPRConfig()
	cfg.MaxIters = 500
	cfg.Tol = 1e-10

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	wantHost := HostRank(mono, hostOf, cfg)
	wantDomain := DomainRank(mono, domainOf, cfg)

	hostGroup := func(g uint64) int { return int(g >> seqBits) }
	domainGroup := func(g uint64) int { return int(g>>seqBits) / hostsPer }

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildGroupShards(t, n, edges, gids, s)

		gotHost := StreamGroupRank(regions, hostGroup, cfg)
		gotDomain := StreamGroupRank(regions, domainGroup, cfg)

		var maxHost, maxDomain float64
		for i := 0; i < n; i++ {
			h := gotHost[shardOf[i]][denseOf[i]]
			if d := math.Abs(h - wantHost[i]); d > maxHost {
				maxHost = d
			}
			dm := gotDomain[shardOf[i]][denseOf[i]]
			if d := math.Abs(dm - wantDomain[i]); d > maxDomain {
				maxDomain = d
			}
		}
		if maxHost > 1e-9 {
			t.Errorf("s=%d: host rank max error %g exceeds 1e-9", s, maxHost)
		}
		if maxDomain > 1e-9 {
			t.Errorf("s=%d: domain rank max error %g exceeds 1e-9", s, maxDomain)
		}
	}
}

// TestStreamGroupRankInheritsWithinGroup checks the scatter: every page of one host gets
// exactly that host's rank, so the per-page vector is constant within a host and the
// distinct values are the host ranks. It also confirms a higher-linked host outranks a
// barely-linked one, so the projection is ranking the host graph, not returning a flat
// vector.
func TestStreamGroupRankInheritsWithinGroup(t *testing.T) {
	const domains, hostsPer, pagesPer, seqBits = 3, 3, 5, 3
	edges, hostOf, _, gids := hostGraph(domains, hostsPer, pagesPer, seqBits)
	n := domains * hostsPer * pagesPer

	cfg := DefaultPRConfig()
	cfg.MaxIters = 500
	cfg.Tol = 1e-10

	regions, shardOf, denseOf := buildGroupShards(t, n, edges, gids, 3)
	got := StreamGroupRank(regions, func(g uint64) int { return int(g >> seqBits) }, cfg)

	// Every page in a host must carry the identical rank.
	byHost := map[int]float64{}
	for i := 0; i < n; i++ {
		r := got[shardOf[i]][denseOf[i]]
		if prev, ok := byHost[hostOf[i]]; ok {
			if math.Abs(prev-r) > 1e-12 {
				t.Fatalf("host %d: pages disagree on rank, %g vs %g", hostOf[i], prev, r)
			}
		} else {
			byHost[hostOf[i]] = r
		}
	}
	if len(byHost) != domains*hostsPer {
		t.Fatalf("got %d distinct host ranks, want %d", len(byHost), domains*hostsPer)
	}
	var sum, min, max float64
	min = math.Inf(1)
	for _, r := range byHost {
		sum += r
		if r < min {
			min = r
		}
		if r > max {
			max = r
		}
	}
	if max <= min {
		t.Fatalf("host ranks are flat (min %g, max %g), the projection is not ranking", min, max)
	}
}

// TestStreamGroupRankEmpty covers the degenerate shapes: a nil shard set, a set of nil
// shards, and a set whose shards hold no nodes all return without panicking and yield no
// ranks.
func TestStreamGroupRankEmpty(t *testing.T) {
	cfg := DefaultPRConfig()
	if got := StreamGroupRank(nil, func(uint64) int { return 0 }, cfg); len(got) != 0 {
		t.Fatalf("nil shards: got %v, want empty", got)
	}
	got := StreamGroupRank([]*Region{nil, nil}, func(uint64) int { return 0 }, cfg)
	if len(got) != 2 || got[0] != nil || got[1] != nil {
		t.Fatalf("nil shard entries: got %v, want two nils", got)
	}
}
