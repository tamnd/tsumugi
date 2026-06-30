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

// TestCrossHostLinkDiversityOnCCrawl runs the cross-shard host-link diversity on the real crawl,
// the gate that proves it can come off the persisted per-shard regions. It uses the same setup
// as the in-edge and reciprocity tests, real canonical URLs, the spec's MPH global node ids, and
// the resulting explicit id tables split across two shards, with each page's host derived from
// its real URL (analyze.HostOf, the same function the build's grouping uses), then computes the
// diversity two ways: once with CrossHostLinkDiversity over the shard regions, where the
// contracted host graph is projected across shards and every boundary-crossing inter-host link
// is recovered from the cross-shard edge list, and once with the single-graph HostLinkDiversity
// over the merged graph. The diversity is a deterministic function of the same inter-host edge
// weights both forms accumulate, so over a partition they must agree to floating-point
// exactness.
//
// As in the cross-rank test the link structure is generated rather than read from the bodies,
// because in-corpus links almost never resolve in a broad crawl sample; the ids, the id tables,
// the dense assignment, and the real host groups are all real, only which node points at which
// is synthetic, the one thing the diversity's correctness does not depend on. The real host
// distribution gives the contracted host graph genuine asymmetry, so the diversity is a real
// spread of values, not a degenerate constant.
func TestCrossHostLinkDiversityOnCCrawl(t *testing.T) {
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

	// Real host group per page, ids assigned first-seen the way grouping does.
	hostOf := make([]int, n)
	hostIDs := map[string]int{}
	for i, cu := range cus {
		h := analyze.HostOf(cu)
		hid, ok := hostIDs[h]
		if !ok {
			hid = len(hostIDs)
			hostIDs[h] = hid
		}
		hostOf[i] = hid
	}

	// Global node id = the spec's MPH over every canonical URL.
	allURLs := make([][]byte, n)
	for i, cu := range cus {
		allURLs[i] = []byte(cu)
	}
	m := mph.Build(allURLs, mph.DefaultGamma)

	const shards = 2
	type loc struct {
		page  int
		shard int
		dense int
	}
	denseGlobals := make([][]uint64, shards)
	locByGlobal := make(map[uint64]loc, n)
	hostByGlobal := make(map[uint64]int, n)
	for i, cu := range cus {
		s := i % shards
		g := m.Lookup([]byte(cu))
		d := len(denseGlobals[s])
		denseGlobals[s] = append(denseGlobals[s], g)
		locByGlobal[g] = loc{page: i, shard: s, dense: d}
		hostByGlobal[g] = hostOf[i]
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
	want := graph.HostLinkDiversity(mono, hostOf)

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(sb[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	got := graph.CrossHostLinkDiversity(regions, func(g uint64) int { return hostByGlobal[g] })

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
		t.Fatal("merged diversity is all zero; gate is vacuous")
	}
	if mis != 0 {
		t.Fatalf("sharded vs monolith host-link diversity mismatches: %d, max error %g", mis, maxErr)
	}
	t.Logf("pages=%d hosts=%d cross-edges=%d nonzero-diversity=%d max error %.2e",
		n, len(hostIDs), crossCount, nonzero, maxErr)
}
