package graph

import (
	"math"
	"testing"
)

// TestCrossReversedPageRankMatchesMonolith is the gate the reversed primitive rests on:
// the sharded reversed loop must produce the same rank as the single-graph streamed
// PageRank over the reversed merged graph, whatever the partition. The merged reversed
// rank is StreamPageRank over ReverseSource with the original in-degree as the reversed
// out-degree, exactly the form graphbuild.go's streamInversePageRank computes; the sharded
// form must match it across 1, 2, 3, and 5 shards, where the cross-shard edges flow
// backward through the reversed routing.
func TestCrossReversedPageRankMatchesMonolith(t *testing.T) {
	const n = 400
	edges := prGraph(n)

	cfg := DefaultPRConfig()
	cfg.MaxIters = 300
	cfg.Tol = 1e-8

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := StreamPageRank(ReverseSource(mono), OutDegreesFromSource(n, mono.InDegree), cfg)

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		got := StreamCrossReversedPageRank(regions, cfg)

		var maxErr, sum float64
		for i := 0; i < n; i++ {
			g := float64(got[shardOf[i]][denseOf[i]])
			sum += g
			if d := math.Abs(g - float64(want[i])); d > maxErr {
				maxErr = d
			}
		}
		if maxErr > 1e-5 {
			t.Errorf("s=%d: max reversed rank error %g exceeds 1e-5", s, maxErr)
		}
		if math.Abs(sum-1) > 1e-4 {
			t.Errorf("s=%d: reversed rank sums to %g, want ~1", s, sum)
		}
	}
}

// TestCrossReversedPageRankSeedTeleport checks the personalized variant against the
// anti-trust form: a spam-seed-biased sharded reversed run must match the spam-seed-biased
// streamed rank over the reversed monolith, the cross-shard twin of graphbuild.go's
// streamAntiTrustRank, so sharded Anti-TrustRank lands on the same ranks as the in-core
// reversed seed-biased rank.
func TestCrossReversedPageRankSeedTeleport(t *testing.T) {
	const n = 300
	const s = 3
	edges := prGraph(n)
	spam := []int{2, 49, 111, 240}

	cfg := DefaultPRConfig()
	cfg.MaxIters = 300
	cfg.Tol = 1e-8

	mb := NewBuilder(n)
	for _, e := range edges {
		mb.AddEdge(e.a, e.b)
	}
	mono, err := Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := StreamPageRankP(ReverseSource(mono), OutDegreesFromSource(n, mono.InDegree), SeedTeleport(n, spam), cfg)

	regions, shardOf, denseOf := buildShards(t, n, edges, s)
	perShard := make([][]int, s)
	for _, sd := range spam {
		perShard[shardOf[sd]] = append(perShard[shardOf[sd]], denseOf[sd])
	}
	got := StreamCrossReversedPageRankP(regions, SeedCrossTeleport(regions, perShard), cfg)

	var maxErr float64
	for i := 0; i < n; i++ {
		g := float64(got[shardOf[i]][denseOf[i]])
		if d := math.Abs(g - float64(want[i])); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("anti-trust max rank error %g exceeds 1e-5", maxErr)
	}
}

// TestCrossReversedPageRankEmpty checks the degenerate shapes: no shards, all-nil shards,
// and a single empty shard all return without panicking.
func TestCrossReversedPageRankEmpty(t *testing.T) {
	cfg := DefaultPRConfig()
	if r := StreamCrossReversedPageRank(nil, cfg); len(r) != 0 {
		t.Fatalf("nil shards returned %d slots", len(r))
	}
	if r := StreamCrossReversedPageRank([]*Region{nil, nil}, cfg); len(r) != 2 {
		t.Fatalf("two nil shards returned %d slots", len(r))
	}
	b := NewBuilder(0)
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r := StreamCrossReversedPageRank([]*Region{g}, cfg); len(r) != 1 {
		t.Fatalf("one empty shard returned %d slots", len(r))
	}
}

// BenchmarkCrossReversedPageRank times one fixed-iteration sharded reversed run over a
// mid-size graph split eight ways, the per-iteration cost the corpus-scale reversed loop
// pays a round.
func BenchmarkCrossReversedPageRank(b *testing.B) {
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

	cfg := DefaultPRConfig()
	cfg.MaxIters = 50
	cfg.Tol = 0 // run the full iteration count every time

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		StreamCrossReversedPageRank(regions, cfg)
	}
}
