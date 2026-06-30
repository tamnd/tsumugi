package collection

import (
	"os"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

// TestCrossInEdgeSignalsOnCCrawl runs the cross-shard in-edge aggregations on the real crawl,
// the gate that proves in-degree and the distinct-linking-domain and distinct-linking-host
// counts can come off the persisted per-shard regions. It uses the same setup as the
// cross-rank test, real canonical URLs, the spec's MPH global node ids, and the resulting
// explicit id tables split across two shards, with each page's host and domain derived from
// its real URL (analyze.HostOf and analyze.RegisteredDomain, the same functions the build's
// groupings uses), then computes the three counts two ways: once with the sharded
// aggregations over the shard regions, where every boundary-crossing in-link is recovered
// through the cross-shard routing, and once with the single-graph signals over the merged
// graph. The counts are integers, so they must agree exactly.
//
// As in the cross-rank test the link structure is generated rather than read from the bodies,
// because in-corpus links almost never resolve in a broad crawl sample; the ids, the id
// tables, the dense assignment, and now the real host and domain groups are all real, only
// which node points at which is synthetic, the one thing the counts' correctness does not
// depend on.
func TestCrossInEdgeSignalsOnCCrawl(t *testing.T) {
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

	// Real host and domain group per page, ids assigned first-seen the way groupings does.
	hostOf := make([]int, n)
	domainOf := make([]int, n)
	hostIDs := map[string]int{}
	domainIDs := map[string]int{}
	for i, cu := range cus {
		h := analyze.HostOf(cu)
		hid, ok := hostIDs[h]
		if !ok {
			hid = len(hostIDs)
			hostIDs[h] = hid
		}
		hostOf[i] = hid
		dom := analyze.RegisteredDomain(h)
		did, ok := domainIDs[dom]
		if !ok {
			did = len(domainIDs)
			domainIDs[dom] = did
		}
		domainOf[i] = did
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
	domainByGlobal := make(map[uint64]int, n)
	for i, cu := range cus {
		s := i % shards
		g := m.Lookup([]byte(cu))
		d := len(denseGlobals[s])
		denseGlobals[s] = append(denseGlobals[s], g)
		locByGlobal[g] = loc{page: i, shard: s, dense: d}
		hostByGlobal[g] = hostOf[i]
		domainByGlobal[g] = domainOf[i]
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
	wantDeg := graph.InDegrees(mono)
	wantDom := graph.LinkingDomains(mono, domainOf)
	wantHost := graph.LinkingHosts(mono, hostOf)

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(sb[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	gotDeg := graph.CrossInDegrees(regions)
	gotDom := graph.CrossLinkingDomains(regions, func(g uint64) int { return domainByGlobal[g] })
	gotHost := graph.CrossLinkingHosts(regions, func(g uint64) int { return hostByGlobal[g] })

	var degMis, domMis, hostMis int
	for g, l := range locByGlobal {
		_ = g
		if gotDeg[l.shard][l.dense] != wantDeg[l.page] {
			degMis++
		}
		if gotDom[l.shard][l.dense] != wantDom[l.page] {
			domMis++
		}
		if gotHost[l.shard][l.dense] != wantHost[l.page] {
			hostMis++
		}
	}
	if degMis != 0 || domMis != 0 || hostMis != 0 {
		t.Fatalf("sharded vs monolith mismatches: in-degree %d, linking-domains %d, linking-hosts %d", degMis, domMis, hostMis)
	}
	t.Logf("pages=%d hosts=%d domains=%d cross-edges=%d all in-edge counts match monolith exactly",
		n, len(hostIDs), len(domainIDs), crossCount)
}
