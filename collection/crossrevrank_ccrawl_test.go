package collection

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

// TestCrossReversedPageRankOnCCrawl runs the reversed global-then-scatter loop on the real
// crawl, the gate that proves the inverse-PageRank and Anti-TrustRank family can compute
// off the persisted per-shard regions. It uses the same setup as the forward cross-rank
// test, real canonical URLs, the spec's MPH global node ids, and the resulting explicit id
// tables split across two shards, then computes the reversed rank two ways: once with the
// sharded reversed loop, where every boundary-crossing link travels backward as a
// reversed-routed cross-shard edge, and once with the single-graph streamed PageRank over
// the reversed merged graph, the exact form graphbuild.go's streamInversePageRank computes.
// The two must agree to float32 precision, which proves the reversed cross-shard exchange,
// the in-degree accounting for far edges, and the global dangling-and-teleport base all
// behave on real ids and real id tables.
//
// As in the forward test the link structure is generated rather than read from the bodies,
// because in-corpus links almost never resolve in a broad crawl sample; the ids, the id
// tables, and the dense assignment are all real, only which node points at which is
// synthetic, the one thing the rank's correctness does not depend on.
func TestCrossReversedPageRankOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	seen := make(map[string]struct{})
	var cus []string
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		if _, dup := seen[cu]; dup {
			continue
		}
		seen[cu] = struct{}{}
		cus = append(cus, cu)
	}
	_ = src.Close()
	n := len(cus)
	if n == 0 {
		t.Skip("no canonical URLs in parquet")
	}

	// Global node id = the spec's MPH over every canonical URL.
	allURLs := make([][]byte, n)
	for i, cu := range cus {
		allURLs[i] = []byte(cu)
	}
	m := mph.Build(allURLs, mph.DefaultGamma)

	// Split across two shards interleaved; track each page's monolith index (its order
	// in cus), shard, local dense, and global id.
	const shards = 2
	type loc struct {
		page   int
		shard  int
		dense  int
		global uint64
	}
	denseGlobals := make([][]uint64, shards)
	locByGlobal := make(map[uint64]loc, n)
	for i, cu := range cus {
		s := i % shards
		g := m.Lookup([]byte(cu))
		d := len(denseGlobals[s])
		denseGlobals[s] = append(denseGlobals[s], g)
		locByGlobal[g] = loc{page: i, shard: s, dense: d, global: g}
	}

	globalsSorted := make([]uint64, 0, n)
	for g := range locByGlobal {
		globalsSorted = append(globalsSorted, g)
	}
	sort.Slice(globalsSorted, func(i, j int) bool { return globalsSorted[i] < globalsSorted[j] })

	// Build the same deterministic far-edge set into both the sharded builders and the
	// monolith. In the shards a boundary-crossing edge becomes a cross-shard edge; in
	// the monolith everything is one intra graph keyed by page index.
	sb := make([]*graph.Builder, shards)
	for s := 0; s < shards; s++ {
		sb[s] = graph.NewBuilder(len(denseGlobals[s])).WithNodeIDs(denseGlobals[s])
	}
	mb := graph.NewBuilder(n)

	const fanout = 4
	var crossCount int
	for i := 0; i < n; i++ {
		src := locByGlobal[globalsSorted[i]]
		for k := 1; k <= fanout; k++ {
			tgt := locByGlobal[globalsSorted[(i*7+k*1300)%n]]
			if tgt.global == src.global {
				continue
			}
			mb.AddEdge(src.page, tgt.page)
			if tgt.shard == src.shard {
				sb[src.shard].AddEdge(src.dense, tgt.dense)
				continue
			}
			sb[src.shard].AddCrossEdge(src.dense, tgt.global)
			crossCount++
		}
	}
	if crossCount == 0 {
		t.Fatal("expected cross edges from the synthetic far-edge set")
	}

	cfg := graph.DefaultPRConfig()
	cfg.MaxIters = 200
	cfg.Tol = 1e-8

	mono, err := graph.Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	// The reversed merged oracle: rank the transpose, dividing each sender by its original
	// in-degree, exactly graphbuild.go's streamInversePageRank.
	want := graph.StreamPageRank(graph.ReverseSource(mono), graph.OutDegreesFromSource(n, mono.InDegree), cfg)

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(sb[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	got := graph.StreamCrossReversedPageRank(regions, cfg)

	var maxErr, sum float64
	for _, l := range locByGlobal {
		g := float64(got[l.shard][l.dense])
		sum += g
		if d := math.Abs(g - float64(want[l.page])); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("sharded vs monolith max reversed rank error %g exceeds 1e-5", maxErr)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("sharded reversed rank sums to %g, want ~1", sum)
	}
	t.Logf("pages=%d shard0=%d shard1=%d cross-edges=%d maxRevRankErr=%.2e sum=%.6f",
		n, len(denseGlobals[0]), len(denseGlobals[1]), crossCount, maxErr, sum)
}
