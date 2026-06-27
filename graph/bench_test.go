package graph

import (
	"math/rand"
	"testing"
)

// webLikeGraph builds a graph with the locality and similarity the compression
// assumes. Nodes are grouped into hosts of contiguous ids. Every page links to
// its host's shared nav block (a contiguous run at the host start), which is what
// makes adjacency lists across a host nearly identical, so the reference and
// interval codes win. A few extra intra-host and cross-host links add the
// residual gaps. This is the shape the build's host grouping produces.
func webLikeGraph(n, hostSize int, seed int64) *Builder {
	rng := rand.New(rand.NewSource(seed))
	b := NewBuilder(n)
	const navSize = 8
	for u := 0; u < n; u++ {
		host := u / hostSize
		lo := host * hostSize
		hi := lo + hostSize
		if hi > n {
			hi = n
		}
		// Shared nav block: the contiguous run every page on the host links to.
		for k := 0; k < navSize && lo+k < hi; k++ {
			b.AddEdge(u, lo+k)
		}
		// A handful of article links within the host and the odd cross-host link.
		extra := 3 + rng.Intn(10)
		for k := 0; k < extra; k++ {
			if rng.Intn(10) < 9 && hi > lo {
				b.AddEdge(u, lo+rng.Intn(hi-lo))
			} else {
				b.AddEdge(u, rng.Intn(n))
			}
		}
	}
	return b
}

// TestCompressionRatio reports the bits an edge the coder achieves on a
// host-local graph. The Boldi-Vigna target is a few bits an edge; a plain
// uint32 adjacency is 32. It is a report, not a hard gate, so ordering work in a
// later milestone can only improve it.
func TestCompressionRatio(t *testing.T) {
	b := webLikeGraph(50000, 200, 7)
	region := b.Build()
	g, err := Open(region)
	if err != nil {
		t.Fatal(err)
	}
	edges := g.EdgeCount()
	// Both planes plus offsets, over the directed edge count once (the transpose
	// holds the same edges, so bits/edge counts forward+transpose against edges).
	bitsPerEdge := float64(len(region)*8) / float64(edges)
	t.Logf("nodes=%d edges=%d region=%d bytes, %.2f bits/edge (forward+transpose+offsets)",
		g.NodeCount(), edges, len(region), bitsPerEdge)
	if bitsPerEdge > 32 {
		t.Fatalf("compression worse than raw uint32: %.2f bits/edge", bitsPerEdge)
	}
}

func BenchmarkOutNeighbors(b *testing.B) {
	g, _ := Open(webLikeGraph(50000, 200, 11).Build())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.OutNeighbors(i % g.NodeCount())
	}
}

func BenchmarkPageRank(b *testing.B) {
	g, _ := Open(webLikeGraph(50000, 200, 13).Build())
	cfg := DefaultPRConfig()
	cfg.MaxIters = 30
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = PageRank(g, cfg)
	}
}
