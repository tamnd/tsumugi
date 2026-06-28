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

// TestCrossPageRankOnCCrawl runs the global-then-scatter PageRank loop on the real
// crawl. It splits the corpus across two shards carrying the real canonical URLs, the
// spec's MPH global node ids, and the resulting explicit id tables, exactly the setup
// the cross-shard edge test uses, then computes PageRank two ways over the same graph:
// once with the sharded loop, where every boundary-crossing link travels as a cross-shard
// edge resolved through the id tables each iteration, and once with the single-graph
// streamed PageRank over the merged graph. The two must agree to float32 precision, which
// proves the per-iteration cross-shard exchange, the out-degree accounting for far edges,
// and the global dangling-and-teleport base all behave on real ids and real id tables.
//
// As in the cross-edge test the link structure is generated rather than read from the
// bodies, because in-corpus links almost never resolve in a broad crawl sample; the ids,
// the id tables, and the dense assignment are all real, only which node points at which is
// synthetic, the one thing PageRank's correctness does not depend on.
func TestCrossPageRankOnCCrawl(t *testing.T) {
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
	want := graph.StreamPageRank(mono, graph.OutDegrees(mono), cfg)

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(sb[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	got := graph.StreamCrossPageRank(regions, cfg)

	var maxErr, sum float64
	for _, l := range locByGlobal {
		g := float64(got[l.shard][l.dense])
		sum += g
		if d := math.Abs(g - float64(want[l.page])); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("sharded vs monolith max rank error %g exceeds 1e-5", maxErr)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("sharded rank sums to %g, want ~1", sum)
	}
	t.Logf("pages=%d shard0=%d shard1=%d cross-edges=%d maxRankErr=%.2e sum=%.6f",
		n, len(denseGlobals[0]), len(denseGlobals[1]), crossCount, maxErr, sum)
}
