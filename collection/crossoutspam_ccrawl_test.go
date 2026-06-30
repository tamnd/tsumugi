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

// TestCrossOutboundSpamRatioOnCCrawl runs the cross-shard outbound-spam ratio on the real crawl,
// the gate that proves the out-edge-against-SpamMass aggregation can come off the persisted
// per-shard regions. It uses the same setup as the reciprocity test, real canonical URLs, the
// spec's MPH global node ids, and the resulting explicit id tables split across two shards, with
// a SpamMass spread across the pages by a hash of their global id (the SpamMass vector is the
// per-shard reversed Anti-TrustRank output at scale; here a deterministic spread stands in so the
// ratio is a real mix of spam and non-spam targets), then computes the ratio two ways: once with
// CrossOutboundSpamRatio over the shard regions, where every boundary-crossing out-link's target
// is resolved to its SpamMass by global id, and once with the single-graph OutboundSpamRatio over
// the merged graph. For a fixed out-degree the value is a ratio of integer counts, so the two
// must agree to floating-point exactness.
//
// As in the cross-rank test the link structure is generated rather than read from the bodies,
// because in-corpus links almost never resolve in a broad crawl sample; the ids, the id tables,
// and the dense assignment are all real, only which node points at which is synthetic.
func TestCrossOutboundSpamRatioOnCCrawl(t *testing.T) {
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

	const shards = 2
	const thresh = 0.5
	type loc struct {
		page  int
		shard int
		dense int
	}
	denseGlobals := make([][]uint64, shards)
	locByGlobal := make(map[uint64]loc, n)
	spamByGlobal := make(map[uint64]float64, n)
	spam := make([]float64, n) // dense by page index, for the merged oracle
	for i, cu := range cus {
		s := i % shards
		g := m.Lookup([]byte(cu))
		d := len(denseGlobals[s])
		denseGlobals[s] = append(denseGlobals[s], g)
		locByGlobal[g] = loc{page: i, shard: s, dense: d}
		mass := float64((g*2654435761+12345)%1000) / 1000.0
		spamByGlobal[g] = mass
		spam[i] = mass
	}

	globalsSorted := make([]uint64, 0, n)
	for g := range locByGlobal {
		globalsSorted = append(globalsSorted, g)
	}
	sort.Slice(globalsSorted, func(i, j int) bool { return globalsSorted[i] < globalsSorted[j] })

	sb := make([]*graph.Builder, shards)
	for s := 0; s < shards; s++ {
		sb[s] = graph.NewBuilder(len(denseGlobals[s])).WithNodeIDs(denseGlobals[s])
	}
	mb := graph.NewBuilder(n)

	const fanout = 4
	var crossCount int
	for i := 0; i < n; i++ {
		src := locByGlobal[globalsSorted[i]]
		srcGlobal := globalsSorted[i]
		for k := 1; k <= fanout; k++ {
			tg := globalsSorted[(i*7+k*1300)%n]
			tgt := locByGlobal[tg]
			if tg == srcGlobal {
				continue
			}
			mb.AddEdge(src.page, tgt.page)
			if tgt.shard == src.shard {
				sb[src.shard].AddEdge(src.dense, tgt.dense)
				continue
			}
			sb[src.shard].AddCrossEdge(src.dense, tg)
			crossCount++
		}
	}
	if crossCount == 0 {
		t.Fatal("expected cross edges from the synthetic far-edge set")
	}

	mono, err := graph.Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := graph.OutboundSpamRatio(mono, spam, thresh)

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(sb[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	got := graph.CrossOutboundSpamRatio(regions, func(g uint64) float64 { return spamByGlobal[g] }, thresh)

	var mis, nonzero int
	var maxErr float64
	for _, l := range locByGlobal {
		e := math.Abs(got[l.shard][l.dense] - want[l.page])
		if e > maxErr {
			maxErr = e
		}
		if e > 1e-12 {
			mis++
		}
		if want[l.page] > 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("merged out-spam ratio is all zero; gate is vacuous")
	}
	if mis != 0 {
		t.Fatalf("sharded vs monolith out-spam ratio mismatches: %d, max error %g", mis, maxErr)
	}
	t.Logf("pages=%d cross-edges=%d nonzero-ratio=%d max error %.2e", n, crossCount, nonzero, maxErr)
}
