package graph

import (
	"math"
	"math/rand"
	"testing"
)

func buildGraph(n int, edges [][2]int) *Region {
	b := NewBuilder(n)
	for _, e := range edges {
		b.AddEdge(e[0], e[1])
	}
	g, err := Open(b.Build())
	if err != nil {
		panic(err)
	}
	return g
}

// TestPageRankSumsToOne checks the distribution stays normalized even with a
// large dangling set, the property the dangling-mass redistribution guarantees.
func TestPageRankSumsToOne(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 300
	var edges [][2]int
	for i := 0; i < 600; i++ {
		u, v := rng.Intn(n), rng.Intn(n)
		edges = append(edges, [2]int{u, v})
	}
	g := buildGraph(n, edges)
	pr := PageRank(g, DefaultPRConfig())
	var sum float64
	for _, v := range pr {
		sum += v
		if v < 0 {
			t.Fatalf("negative rank %g", v)
		}
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("pagerank sums to %g, not 1", sum)
	}
}

// TestPageRankChain checks the ordering on a simple chain plus a hub: the node
// every other node links to must rank highest.
func TestPageRankChain(t *testing.T) {
	// Nodes 0..4 all link to node 5, the authority.
	n := 6
	var edges [][2]int
	for i := 0; i < 5; i++ {
		edges = append(edges, [2]int{i, 5})
	}
	g := buildGraph(n, edges)
	pr := PageRank(g, DefaultPRConfig())
	for i := 0; i < 5; i++ {
		if pr[5] <= pr[i] {
			t.Fatalf("hub rank %g not above node %d rank %g", pr[5], i, pr[i])
		}
	}
}

// TestPageRankSymmetry checks a two-node mutual link splits rank evenly.
func TestPageRankSymmetry(t *testing.T) {
	g := buildGraph(2, [][2]int{{0, 1}, {1, 0}})
	pr := PageRank(g, DefaultPRConfig())
	if math.Abs(pr[0]-pr[1]) > 1e-9 || math.Abs(pr[0]-0.5) > 1e-9 {
		t.Fatalf("symmetric pair got %v", pr)
	}
}

// TestTrustRankBias checks trust concentrates on the seed and what it reaches,
// not on a disconnected node that draws only uniform PageRank.
func TestTrustRankBias(t *testing.T) {
	// 0 -> 1 -> 2 is the trusted component; 3 -> 4 is untrusted and separate.
	n := 5
	edges := [][2]int{{0, 1}, {1, 2}, {3, 4}}
	g := buildGraph(n, edges)
	tr := TrustRank(g, []int{0}, DefaultPRConfig())
	if tr[1] <= tr[4] || tr[2] <= tr[4] {
		t.Fatalf("trust did not concentrate on seed component: %v", tr)
	}
	if tr[3] > tr[1] {
		t.Fatalf("untrusted node outranked trusted reach: %v", tr)
	}
}

// TestSpamMass checks a page fed only by an untrusted farm scores near one while
// pages in the trusted cluster score low. The trusted cluster is an interlinked
// triangle whose members are all seeds, the regime where trust should fully
// explain rank; the farm is a separate cycle with no inbound trust.
func TestSpamMass(t *testing.T) {
	n := 7
	edges := [][2]int{
		{0, 1}, {1, 0}, {1, 2}, {2, 1}, {0, 2}, {2, 0}, // trusted triangle
		{3, 4}, {4, 5}, {5, 3}, {3, 6}, {4, 6}, {5, 6}, {6, 3}, // farm
	}
	g := buildGraph(n, edges)
	cfg := DefaultPRConfig()
	seeds := []int{0, 1, 2}
	pr := PageRank(g, cfg)
	tr := TrustRank(g, seeds, cfg)
	sm := SpamMass(pr, tr, seeds)
	for _, v := range []int{0, 1, 2} {
		if sm[v] > 0.3 {
			t.Fatalf("trusted node %d spam mass %g too high", v, sm[v])
		}
	}
	if sm[6] < 0.7 {
		t.Fatalf("spam target mass %g too low", sm[6])
	}
	if sm[6] <= sm[2] {
		t.Fatalf("spam target mass %g not above trusted page %g", sm[6], sm[2])
	}
}

// TestInversePageRank checks the reversed-graph rank scores a node by its
// out-reach: on a chain the head, which reaches every other node, must outrank the
// tail, which reaches nobody. These are the trust-seed candidates.
func TestInversePageRank(t *testing.T) {
	// 0 -> 1 -> 2 -> 3: node 0 reaches the most, node 3 the least.
	g := buildGraph(4, [][2]int{{0, 1}, {1, 2}, {2, 3}})
	inv := InversePageRank(g, DefaultPRConfig())
	if inv[0] <= inv[3] {
		t.Fatalf("inverse pagerank head %g not above tail %g", inv[0], inv[3])
	}
	var sum float64
	for _, v := range inv {
		sum += v
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("inverse pagerank sums to %g, not 1", sum)
	}
}

// TestAntiTrustRank checks distrust flows backward from a spam seed to the pages
// that link to it, not to an unrelated page. Nodes 0 and 1 link to the spam node
// 2; node 3 is unrelated, so 0 and 1 must score above 3.
func TestAntiTrustRank(t *testing.T) {
	g := buildGraph(4, [][2]int{{0, 2}, {1, 2}})
	at := AntiTrustRank(g, []int{2}, DefaultPRConfig())
	if at[0] <= at[3] || at[1] <= at[3] {
		t.Fatalf("distrust did not reach the spam linkers: %v", at)
	}
}

// TestInDegrees checks the count matches the transpose.
func TestInDegrees(t *testing.T) {
	edges := [][2]int{{0, 3}, {1, 3}, {2, 3}, {0, 1}}
	g := buildGraph(4, edges)
	id := InDegrees(g)
	want := []int{0, 1, 0, 3}
	for i := range want {
		if id[i] != want[i] {
			t.Fatalf("indegree[%d]=%d want %d", i, id[i], want[i])
		}
	}
}

// TestLinkingDomains checks distinct-domain counting collapses many links from
// one domain to a single vote.
func TestLinkingDomains(t *testing.T) {
	// 0,1,2 are domain 0; 3 is domain 1. All link to 4.
	domainOf := []int{0, 0, 0, 1, 2}
	edges := [][2]int{{0, 4}, {1, 4}, {2, 4}, {3, 4}}
	g := buildGraph(5, edges)
	ld := LinkingDomains(g, domainOf)
	if ld[4] != 2 {
		t.Fatalf("linking domains of 4 = %d, want 2 (domain 0 and domain 1)", ld[4])
	}
}

// TestHostRank checks aggregation drops intra-host links and ranks the host that
// receives a cross-host link above one that only links internally.
func TestHostRank(t *testing.T) {
	// Host 0: nodes 0,1 (link to each other, intra-host). Host 1: node 2 links
	// to node 0 (cross-host into host 0). So host 0 should outrank host 1.
	hostOf := []int{0, 0, 1}
	edges := [][2]int{{0, 1}, {1, 0}, {2, 0}}
	g := buildGraph(3, edges)
	hr := HostRank(g, hostOf, DefaultPRConfig())
	if hr[0] <= hr[2] {
		t.Fatalf("host 0 rank %g not above host 1 rank %g", hr[0], hr[2])
	}
	// Nodes in the same host share a rank.
	if math.Abs(hr[0]-hr[1]) > 1e-12 {
		t.Fatalf("same-host nodes differ: %g vs %g", hr[0], hr[1])
	}
}
