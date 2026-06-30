package graph

import (
	"math"
	"testing"
)

// TestCrossReciprocityMatchesMonolith is the gate the cross-shard reciprocity rests on: the
// sharded fraction of out-links that link back must equal Reciprocity over the merged graph,
// whatever the partition. Reciprocity is a ratio of integer counts (hits over out-degree), so
// for a fixed out-degree the value is exact: the cross-shard out-edge set and the cross-shard
// inbound gather must recover precisely the out-neighbor and in-neighbor sets the merged
// adjacency and transpose hold, matched in the shared global id space.
func TestCrossReciprocityMatchesMonolith(t *testing.T) {
	const n = 400
	edges := prGraph(n)

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := Reciprocity(mono)

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		got := CrossReciprocity(regions)
		for i := 0; i < n; i++ {
			sh, d := shardOf[i], denseOf[i]
			if math.Abs(got[sh][d]-want[i]) > 1e-12 {
				t.Errorf("s=%d node %d: reciprocity %g, want %g", s, i, got[sh][d], want[i])
			}
		}
	}
}

// TestCrossReciprocityMutualEdges plants explicit mutual links so the gate exercises a known
// nonzero answer, not only the synthetic graph's incidental reciprocity. Nodes 0 and 1 cite
// each other, 2 cites 0 one-directionally; over every partition node 0's reciprocity is the
// fraction of its out-links that come back, which the merged form and the sharded form must
// agree on with the back-link crossing a shard boundary in some partitions.
func TestCrossReciprocityMutualEdges(t *testing.T) {
	const n = 6
	edges := []prEdge{
		{0, 1}, {1, 0}, // mutual
		{2, 0}, // one-directional into 0
		{0, 3}, // one-directional out of 0
		{3, 4}, {4, 5},
	}
	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := Reciprocity(mono)
	// Node 0 has out-edges to {1,3}; only 1 links back, so reciprocity is 1/2.
	if math.Abs(want[0]-0.5) > 1e-12 {
		t.Fatalf("monolith node 0 reciprocity %g, want 0.5", want[0])
	}

	for _, s := range []int{1, 2, 3, 6} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		got := CrossReciprocity(regions)
		for i := 0; i < n; i++ {
			sh, d := shardOf[i], denseOf[i]
			if math.Abs(got[sh][d]-want[i]) > 1e-12 {
				t.Errorf("s=%d node %d: reciprocity %g, want %g", s, i, got[sh][d], want[i])
			}
		}
	}
}

// TestCrossReciprocityEmpty checks the degenerate shapes: no shards, all-nil shards, and a
// single empty shard all return without panicking.
func TestCrossReciprocityEmpty(t *testing.T) {
	if r := CrossReciprocity(nil); len(r) != 0 {
		t.Fatalf("nil shards returned %d slots", len(r))
	}
	if r := CrossReciprocity([]*Region{nil, nil}); len(r) != 2 {
		t.Fatalf("two nil shards returned %d slots", len(r))
	}
	b := NewBuilder(0)
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r := CrossReciprocity([]*Region{g}); len(r) != 1 {
		t.Fatalf("one empty shard returned %d slots", len(r))
	}
}

// BenchmarkCrossReciprocity times one full reciprocity pass over a mid-size graph split eight
// ways, the per-build cost the both-sides aggregation pays once.
func BenchmarkCrossReciprocity(b *testing.B) {
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
		CrossReciprocity(regions)
	}
}
