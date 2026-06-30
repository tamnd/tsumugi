package graph

import (
	"math"
	"testing"
)

// spreadSpamMass assigns each of the n nodes a SpamMass spread across [0,1) by a hash of its
// id, so a fixed threshold splits the nodes into spam and non-spam in a non-trivial mix and the
// out-spam ratio is a real spread of values, not all zero or all one. The merged dense-indexed
// vector and the sharded global-id function name the same mass for the same node because
// buildShards gives node i the global id i.
func spreadSpamMass(n int) []float64 {
	m := make([]float64, n)
	for i := 0; i < n; i++ {
		m[i] = float64((uint64(i)*2654435761+12345)%1000) / 1000.0
	}
	return m
}

// TestCrossOutboundSpamRatioMatchesMonolith is the gate the slice rests on: the sharded fraction
// of out-links that point at spam must equal OutboundSpamRatio over the merged graph, whatever
// the partition. For a fixed out-degree the value is a ratio of integer counts, so the agreement
// is exact: the cross-shard out-edge set must recover precisely the out-neighbor set the merged
// adjacency holds, and each target's spam mass must resolve to the same value through the
// global-id function as the merged dense-indexed lookup.
func TestCrossOutboundSpamRatioMatchesMonolith(t *testing.T) {
	const n = 400
	edges := prGraph(n)
	spam := spreadSpamMass(n)
	const thresh = 0.5

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := OutboundSpamRatio(mono, spam, thresh)

	// Sanity: the gate must see a real spread, both nonzero ratios (out-links to spam) and the
	// full out-degree denominator, or a broken all-zero implementation would pass.
	var nonzero int
	for _, r := range want {
		if r > 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("merged out-spam ratio is all zero; gate is vacuous")
	}

	spamOfGlobal := func(g uint64) float64 { return spam[g] }

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		got := CrossOutboundSpamRatio(regions, spamOfGlobal, thresh)
		for i := 0; i < n; i++ {
			sh, d := shardOf[i], denseOf[i]
			if math.Abs(got[sh][d]-want[i]) > 1e-12 {
				t.Errorf("s=%d node %d: out-spam ratio %g, want %g", s, i, got[sh][d], want[i])
			}
		}
	}
}

// TestCrossOutboundSpamRatioCrossTargets plants out-links to a known spam node across a shard
// boundary so the gate exercises the cross-edge path with a known answer. Node 0 links to nodes
// 1, 2, 3; only node 1 is spam (mass 1.0, the rest 0.0), so its out-spam ratio is 1/3, recovered
// whether the spam target lands in node 0's shard or another.
func TestCrossOutboundSpamRatioCrossTargets(t *testing.T) {
	const n = 6
	edges := []prEdge{{0, 1}, {0, 2}, {0, 3}, {3, 4}, {4, 5}}
	spam := make([]float64, n)
	spam[1] = 1.0
	const thresh = 0.5

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := OutboundSpamRatio(mono, spam, thresh)
	if math.Abs(want[0]-1.0/3.0) > 1e-12 {
		t.Fatalf("monolith node 0 out-spam ratio %g, want 1/3", want[0])
	}

	spamOfGlobal := func(g uint64) float64 { return spam[g] }
	for _, s := range []int{1, 2, 3, 6} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		got := CrossOutboundSpamRatio(regions, spamOfGlobal, thresh)
		for i := 0; i < n; i++ {
			sh, d := shardOf[i], denseOf[i]
			if math.Abs(got[sh][d]-want[i]) > 1e-12 {
				t.Errorf("s=%d node %d: out-spam ratio %g, want %g", s, i, got[sh][d], want[i])
			}
		}
	}
}

// TestCrossOutboundSpamRatioEmpty checks the degenerate shapes: no shards, all-nil shards, and a
// single empty shard all return without panicking.
func TestCrossOutboundSpamRatioEmpty(t *testing.T) {
	zero := func(uint64) float64 { return 0 }
	if r := CrossOutboundSpamRatio(nil, zero, 0.5); len(r) != 0 {
		t.Fatalf("nil shards returned %d slots", len(r))
	}
	if r := CrossOutboundSpamRatio([]*Region{nil, nil}, zero, 0.5); len(r) != 2 {
		t.Fatalf("two nil shards returned %d slots", len(r))
	}
	b := NewBuilder(0)
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r := CrossOutboundSpamRatio([]*Region{g}, zero, 0.5); len(r) != 1 {
		t.Fatalf("one empty shard returned %d slots", len(r))
	}
}

// BenchmarkCrossOutboundSpamRatio times one full out-spam pass over a mid-size graph split eight
// ways, the per-build cost the out-edge aggregation pays once.
func BenchmarkCrossOutboundSpamRatio(b *testing.B) {
	const n = 8000
	const s = 8
	var edges []prEdge
	for i := 0; i < n; i++ {
		if i%37 == 0 {
			continue
		}
		for k := 0; k < 8; k++ {
			j := (i*7 + k*1301 + 17) % n
			if j != i {
				edges = append(edges, prEdge{i, j})
			}
		}
	}
	spam := spreadSpamMass(n)
	spamOfGlobal := func(g uint64) float64 { return spam[g] }
	shardOf, denseOf, members := partition(n, s)
	builders := make([]*Builder, s)
	for sh := 0; sh < s; sh++ {
		ids := make([]uint64, len(members[sh]))
		for d, idx := range members[sh] {
			ids[d] = uint64(idx)
		}
		builders[sh] = NewBuilder(len(members[sh])).WithNodeIDs(ids)
	}
	for _, e := range edges {
		sa, sb := shardOf[e.a], shardOf[e.b]
		if sa == sb {
			builders[sa].AddEdge(denseOf[e.a], denseOf[e.b])
		} else {
			builders[sa].AddCrossEdge(denseOf[e.a], uint64(e.b))
		}
	}
	regions := make([]*Region, s)
	for sh := 0; sh < s; sh++ {
		r, err := Open(builders[sh].Build())
		if err != nil {
			b.Fatalf("open shard %d: %v", sh, err)
		}
		regions[sh] = r
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CrossOutboundSpamRatio(regions, spamOfGlobal, 0.5)
	}
}
