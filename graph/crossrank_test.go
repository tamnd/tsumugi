package graph

import (
	"math"
	"testing"
)

type prEdge struct{ a, b int }

// prGraph builds a deterministic directed graph with a mix of in-degrees and a
// scattering of dangling nodes, the shape a PageRank test needs to be meaningful.
func prGraph(n int) []prEdge {
	var edges []prEdge
	for i := 0; i < n; i++ {
		// Nodes divisible by 37 are dangling: they get no out-edges, so the loop must
		// route their rank through the global teleport-plus-dangle base.
		if i%37 == 0 {
			continue
		}
		for k := 0; k < 5; k++ {
			j := (i*7 + k*131 + 17) % n
			if j != i {
				edges = append(edges, prEdge{i, j})
			}
		}
	}
	return edges
}

// partition assigns nodes to shards round-robin, so a shard owns a non-contiguous set
// of global ids and its id table is the explicit sorted-globals path, not the fast
// contiguous one. It returns each node's shard, its local dense docID within that shard,
// and the per-shard member lists in dense order.
func partition(n, s int) (shardOf, denseOf []int, members [][]int) {
	shardOf = make([]int, n)
	denseOf = make([]int, n)
	members = make([][]int, s)
	for i := 0; i < n; i++ {
		sh := i % s
		shardOf[i] = sh
		denseOf[i] = len(members[sh])
		members[sh] = append(members[sh], i)
	}
	return shardOf, denseOf, members
}

// buildShards splits the global edge set across s shards: an edge whose endpoints land
// in the same shard becomes an intra AddEdge, one that crosses becomes an AddCrossEdge
// named by the target's global id, which equals the global node index here. Each shard
// carries an explicit id table mapping its dense docIDs to those global ids.
func buildShards(t *testing.T, n int, edges []prEdge, s int) ([]*Region, []int, []int) {
	t.Helper()
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
			t.Fatalf("open shard %d: %v", sh, err)
		}
		regions[sh] = r
	}
	return regions, shardOf, denseOf
}

// TestCrossPageRankMatchesMonolith is the gate the whole slice rests on: the sharded
// global-then-scatter loop must produce the same rank as the single-graph streamed
// PageRank over the merged graph, whatever the partition. It builds one monolith and
// compares it against the same edge set split across 1, 2, 3, and 5 shards, where the
// cross-shard edges and the id-table resolution carry every link that crosses a
// boundary.
func TestCrossPageRankMatchesMonolith(t *testing.T) {
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
	want := StreamPageRank(mono, OutDegrees(mono), cfg)

	for _, s := range []int{1, 2, 3, 5} {
		regions, shardOf, denseOf := buildShards(t, n, edges, s)
		got := StreamCrossPageRank(regions, cfg)

		var maxErr, sum float64
		for i := 0; i < n; i++ {
			g := float64(got[shardOf[i]][denseOf[i]])
			sum += g
			if d := math.Abs(g - float64(want[i])); d > maxErr {
				maxErr = d
			}
		}
		if maxErr > 1e-5 {
			t.Errorf("s=%d: max rank error %g exceeds 1e-5", s, maxErr)
		}
		if math.Abs(sum-1) > 1e-4 {
			t.Errorf("s=%d: rank sums to %g, want ~1", s, sum)
		}
	}
}

// TestCrossPageRankSeedTeleport checks the personalized variant: a seed-biased sharded
// run must match the seed-biased streamed PageRank over the monolith, the cross-shard
// twin of StreamPageRankP, so sharded TrustRank lands on the same ranks as the in-core
// seed-biased rank.
func TestCrossPageRankSeedTeleport(t *testing.T) {
	const n = 300
	const s = 3
	edges := prGraph(n)
	seeds := []int{1, 50, 123, 270}

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
	want := StreamPageRankP(mono, OutDegrees(mono), SeedTeleport(n, seeds), cfg)

	regions, shardOf, denseOf := buildShards(t, n, edges, s)
	perShard := make([][]int, s)
	for _, sd := range seeds {
		perShard[shardOf[sd]] = append(perShard[shardOf[sd]], denseOf[sd])
	}
	got := StreamCrossPageRankP(regions, SeedCrossTeleport(regions, perShard), cfg)

	var maxErr float64
	for i := 0; i < n; i++ {
		g := float64(got[shardOf[i]][denseOf[i]])
		if d := math.Abs(g - float64(want[i])); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("seed-biased max rank error %g exceeds 1e-5", maxErr)
	}
}

// TestCrossPageRankEmpty checks the degenerate shapes: no shards, all-nil shards, and a
// single empty shard all return without panicking.
func TestCrossPageRankEmpty(t *testing.T) {
	cfg := DefaultPRConfig()
	if r := StreamCrossPageRank(nil, cfg); len(r) != 0 {
		t.Fatalf("nil shards returned %d slots", len(r))
	}
	if r := StreamCrossPageRank([]*Region{nil, nil}, cfg); len(r) != 2 {
		t.Fatalf("two nil shards returned %d slots", len(r))
	}
	b := NewBuilder(0)
	g, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r := StreamCrossPageRank([]*Region{g}, cfg); len(r) != 1 {
		t.Fatalf("one empty shard returned %d slots", len(r))
	}
}

// BenchmarkCrossPageRank times one fixed-iteration sharded run over a mid-size graph
// split eight ways, the per-iteration cost the corpus-scale loop pays a round.
func BenchmarkCrossPageRank(b *testing.B) {
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
		StreamCrossPageRank(regions, cfg)
	}
}
