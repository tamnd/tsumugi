package graph

import (
	"math"
	"testing"
)

// TestCrossHostLinkDiversityMatchesMonolith is the gate the slice rests on: the sharded
// projection must give every page the same host-link diversity as the single-Region
// HostLinkDiversity over the merged graph, whatever the partition. Diversity is the normalized
// entropy of a host's inbound source-host weight distribution, computed over the same contracted
// host graph the host rank uses, so any divergence is the cross-shard projection losing or
// double-counting an inter-host edge. It reuses the host/domain test graph: the sharded run
// reads a node's host group from its global id (the id shifted past the page sequence), the same
// grouping the merged run gets from the hostOf array.
func TestCrossHostLinkDiversityMatchesMonolith(t *testing.T) {
	const domains, hostsPer, pagesPer, seqBits = 5, 4, 6, 3
	edges, hostOf, _, gids := hostGraph(domains, hostsPer, pagesPer, seqBits)
	n := domains * hostsPer * pagesPer

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := HostLinkDiversity(mono, hostOf)

	// Sanity: the test graph must carry real diversity, not an all-zero vector, or the gate
	// would pass on a broken implementation that always returns zero.
	var nonzero int
	for _, d := range want {
		if d > 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("test graph has no host with nonzero diversity; gate is vacuous")
	}

	hostGroup := func(g uint64) int { return int(g >> seqBits) }

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildGroupShards(t, n, edges, gids, s)
		got := CrossHostLinkDiversity(regions, hostGroup)

		var maxErr float64
		for i := 0; i < n; i++ {
			d := math.Abs(got[shardOf[i]][denseOf[i]] - want[i])
			if d > maxErr {
				maxErr = d
			}
		}
		if maxErr > 1e-12 {
			t.Errorf("s=%d: host-link diversity max error %g exceeds 1e-12", s, maxErr)
		}
	}
}

// TestCrossHostLinkDiversityInheritsWithinHost checks the scatter: every page of one host gets
// exactly that host's diversity, so the per-page vector is constant within a host.
func TestCrossHostLinkDiversityInheritsWithinHost(t *testing.T) {
	const domains, hostsPer, pagesPer, seqBits = 3, 3, 4, 3
	edges, _, _, gids := hostGraph(domains, hostsPer, pagesPer, seqBits)
	n := domains * hostsPer * pagesPer
	hostGroup := func(g uint64) int { return int(g >> seqBits) }

	regions, shardOf, denseOf := buildGroupShards(t, n, edges, gids, 3)
	got := CrossHostLinkDiversity(regions, hostGroup)

	// Group page values by host and assert each host's pages all carry one value.
	byHost := map[int]float64{}
	seen := map[int]bool{}
	for i := 0; i < n; i++ {
		hg := int(gids[i] >> seqBits)
		v := got[shardOf[i]][denseOf[i]]
		if seen[hg] {
			if math.Abs(byHost[hg]-v) > 1e-15 {
				t.Fatalf("host %d page %d diversity %g, want host value %g", hg, i, v, byHost[hg])
			}
		} else {
			byHost[hg] = v
			seen[hg] = true
		}
	}
}

// TestCrossHostLinkDiversityEmpty checks the degenerate shapes: no shards, all-nil shards, and
// a single empty shard all return without panicking.
func TestCrossHostLinkDiversityEmpty(t *testing.T) {
	zero := func(uint64) int { return 0 }
	if r := CrossHostLinkDiversity(nil, zero); len(r) != 0 {
		t.Fatalf("nil shards returned %d slots", len(r))
	}
	if r := CrossHostLinkDiversity([]*Region{nil, nil}, zero); len(r) != 2 {
		t.Fatalf("two nil shards returned %d slots", len(r))
	}
	b := NewBuilder(0)
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r := CrossHostLinkDiversity([]*Region{g}, zero); len(r) != 1 {
		t.Fatalf("one empty shard returned %d slots", len(r))
	}
}

// BenchmarkCrossHostLinkDiversity times one full diversity pass over a mid-size host graph split
// eight ways, the per-build cost the contracted-graph entropy pays once.
func BenchmarkCrossHostLinkDiversity(b *testing.B) {
	const domains, hostsPer, pagesPer, seqBits = 40, 10, 20, 5
	edges, _, _, gids := hostGraph(domains, hostsPer, pagesPer, seqBits)
	n := domains * hostsPer * pagesPer
	hostGroup := func(g uint64) int { return int(g >> seqBits) }

	shardOf, denseOf, members := partition(n, 8)
	builders := make([]*Builder, 8)
	for sh := 0; sh < 8; sh++ {
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
	regions := make([]*Region, 8)
	for sh := 0; sh < 8; sh++ {
		r, err := Open(builders[sh].Build())
		if err != nil {
			b.Fatalf("open shard %d: %v", sh, err)
		}
		regions[sh] = r
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CrossHostLinkDiversity(regions, hostGroup)
	}
}
