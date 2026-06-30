package graph

import "testing"

// crossSignalGroups assigns each of the n nodes a host and a domain id, hosts finer than
// domains, the shape the in-edge group counts are meaningful over. The exact assignment does
// not matter to the gate, only that the merged dense-indexed arrays and the sharded
// global-id functions name the same group for the same node, which holds because buildShards
// gives node i the global id i.
func crossSignalGroups(n int) (hostOf, domainOf []int) {
	hostOf = make([]int, n)
	domainOf = make([]int, n)
	for i := 0; i < n; i++ {
		hostOf[i] = i % 41
		domainOf[i] = i % 13
	}
	return hostOf, domainOf
}

// TestCrossInEdgeSignalsMatchMonolith is the gate the in-edge family rests on: the sharded
// in-degree and distinct-linking-domain and distinct-linking-host counts must equal the
// single-graph signals over the merged graph, whatever the partition. These are integer
// counts, so the agreement is exact, not float-tolerant: the cross-shard inbound gather
// recovers precisely the in-neighbor set the merged transpose holds.
func TestCrossInEdgeSignalsMatchMonolith(t *testing.T) {
	const n = 400
	edges := prGraph(n)
	hostOf, domainOf := crossSignalGroups(n)

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	wantDeg := InDegrees(mono)
	wantDom := LinkingDomains(mono, domainOf)
	wantHost := LinkingHosts(mono, hostOf)

	domainOfGlobal := func(g uint64) int { return domainOf[g] }
	hostOfGlobal := func(g uint64) int { return hostOf[g] }

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		gotDeg := CrossInDegrees(regions)
		gotDom := CrossLinkingDomains(regions, domainOfGlobal)
		gotHost := CrossLinkingHosts(regions, hostOfGlobal)

		for i := 0; i < n; i++ {
			sh, d := shardOf[i], denseOf[i]
			if gotDeg[sh][d] != wantDeg[i] {
				t.Errorf("s=%d node %d: in-degree %d, want %d", s, i, gotDeg[sh][d], wantDeg[i])
			}
			if gotDom[sh][d] != wantDom[i] {
				t.Errorf("s=%d node %d: linking-domains %d, want %d", s, i, gotDom[sh][d], wantDom[i])
			}
			if gotHost[sh][d] != wantHost[i] {
				t.Errorf("s=%d node %d: linking-hosts %d, want %d", s, i, gotHost[sh][d], wantHost[i])
			}
		}
	}
}

// TestCrossInEdgeSignalsEmpty checks the degenerate shapes: no shards, all-nil shards, and a
// single empty shard all return without panicking.
func TestCrossInEdgeSignalsEmpty(t *testing.T) {
	zero := func(uint64) int { return 0 }
	if r := CrossInDegrees(nil); len(r) != 0 {
		t.Fatalf("nil shards returned %d slots", len(r))
	}
	if r := CrossLinkingDomains([]*Region{nil, nil}, zero); len(r) != 2 {
		t.Fatalf("two nil shards returned %d slots", len(r))
	}
	b := NewBuilder(0)
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r := CrossLinkingHosts([]*Region{g}, zero); len(r) != 1 {
		t.Fatalf("one empty shard returned %d slots", len(r))
	}
}

// BenchmarkCrossLinkingDomains times one full distinct-domain pass over a mid-size graph
// split eight ways, the per-build cost the in-edge aggregation pays once.
func BenchmarkCrossLinkingDomains(b *testing.B) {
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
	domainOf := make([]int, n)
	for i := range domainOf {
		domainOf[i] = i % 13
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
	domainOfGlobal := func(g uint64) int { return domainOf[g] }

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CrossLinkingDomains(regions, domainOfGlobal)
	}
}
