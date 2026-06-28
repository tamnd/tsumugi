package collection

import (
	"os"
	"reflect"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

// TestCrossEdgeOnCCrawl exercises the cross-shard edge list and its resolution join
// on the real crawl. The corpus is split across two shards, each carrying the real
// canonical URLs, the spec's MPH global node ids, and the resulting explicit id
// table (the MPH ids are a permutation of the dense order, so the table path runs).
// A node's far out-edges are routed against the other shard's id table by
// RouteCrossEdges, and the recovered inbound edges are checked against an
// independently computed ground truth, so the gap-encode, the gap-decode, and the
// global-to-dense resolution all agree on real ids.
//
// The link structure itself is generated rather than read from the bodies: an
// intra-corpus link in a broad crawl sample almost never resolves (a page's links
// point overwhelmingly outside the sampled shard set), which the test logs from the
// real bodies. The edges the join is checked on are deterministic far edges between
// real corpus nodes, so the URLs, the global ids, the id tables, and the dense
// assignment under test are all real; only which node points at which is synthetic,
// the one part the cross-shard format is indifferent to.
func TestCrossEdgeOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	type page struct {
		cu    string
		links []string
	}
	seen := make(map[string]struct{})
	var pages []page
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
		pages = append(pages, page{cu: cu, links: analyze.Links(d)})
	}
	_ = src.Close()
	n := len(pages)
	if n == 0 {
		t.Skip("no canonical URLs in parquet")
	}

	// The global node id is the spec's MPH over every canonical URL.
	allURLs := make([][]byte, n)
	for i, p := range pages {
		allURLs[i] = []byte(p.cu)
	}
	m := mph.Build(allURLs, mph.DefaultGamma)
	globalOf := func(cu string) uint64 { return m.Lookup([]byte(cu)) }

	// Split the corpus across two shards, interleaved so each gets half. Each page's
	// owner shard, local dense docID, and global id form the directory the routing
	// join resolves against.
	const shards = 2
	type loc struct {
		shard  int
		dense  int
		global uint64
	}
	dir := make(map[string]loc, n)
	denseGlobals := make([][]uint64, shards) // per shard, dense docID -> global id
	for i, p := range pages {
		s := i % shards
		d := len(denseGlobals[s])
		g := globalOf(p.cu)
		denseGlobals[s] = append(denseGlobals[s], g)
		dir[p.cu] = loc{shard: s, dense: d, global: g}
	}

	// Sanity check on the real link structure: how many in-corpus links resolve, and
	// of those how many cross the split. This is logged, not asserted, because a broad
	// crawl sample resolves almost none.
	var realIntra, realCross int
	for _, p := range pages {
		srcLoc := dir[p.cu]
		for _, link := range p.links {
			if tgt, ok := dir[link]; ok {
				if tgt.shard == srcLoc.shard {
					realIntra++
				} else {
					realCross++
				}
			}
		}
	}

	// Build each shard with the real ids, then add deterministic far edges: each
	// dense source points at a handful of other corpus nodes chosen by a stride over
	// the global id space, so targets spread across both shards. Edges whose target
	// lands in the same shard are intra-shard (forward adjacency); the rest are the
	// cross-shard edges the join recovers.
	builders := make([]*graph.Builder, shards)
	for s := 0; s < shards; s++ {
		builders[s] = graph.NewBuilder(len(denseGlobals[s])).WithNodeIDs(denseGlobals[s])
	}
	// A global-id-to-(shard,dense) lookup over the real ids, the directory the build
	// would carry; here it is the map keyed by global id.
	byGlobal := make(map[uint64]loc, n)
	for _, l := range dir {
		byGlobal[l.global] = l
	}
	globalsSorted := make([]uint64, 0, n)
	for g := range byGlobal {
		globalsSorted = append(globalsSorted, g)
	}
	sort.Slice(globalsSorted, func(i, j int) bool { return globalsSorted[i] < globalsSorted[j] })

	wantInbound := make([][]graph.InboundEdge, shards)
	var crossCount int
	const fanout = 4
	for i := 0; i < n; i++ {
		src := byGlobal[globalsSorted[i]]
		for k := 1; k <= fanout; k++ {
			tgt := byGlobal[globalsSorted[(i*7+k*1300)%n]]
			if tgt.global == src.global {
				continue
			}
			if tgt.shard == src.shard {
				builders[src.shard].AddEdge(src.dense, tgt.dense)
				continue
			}
			builders[src.shard].AddCrossEdge(src.dense, tgt.global)
			wantInbound[tgt.shard] = append(wantInbound[tgt.shard],
				graph.InboundEdge{Source: src.global, Target: tgt.dense})
			crossCount++
		}
	}

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(builders[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	t.Logf("pages=%d shard0=%d shard1=%d real-links(intra=%d cross=%d) routed-cross=%d",
		n, len(denseGlobals[0]), len(denseGlobals[1]), realIntra, realCross, crossCount)
	if crossCount == 0 {
		t.Fatal("expected cross edges from the synthetic far-edge set")
	}

	gotInbound := graph.RouteCrossEdges(regions)
	for s := 0; s < shards; s++ {
		got := append([]graph.InboundEdge(nil), gotInbound[s]...)
		want := append([]graph.InboundEdge(nil), wantInbound[s]...)
		sortInbound(got)
		sortInbound(want)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("shard %d inbound mismatch: got %d edges, want %d", s, len(got), len(want))
		}
		for _, e := range got {
			if e.Target < 0 || e.Target >= regions[s].NodeCount() {
				t.Fatalf("shard %d inbound target %d out of range", s, e.Target)
			}
			// The source global must be a real corpus id this routing did not own.
			if _, ok := byGlobal[e.Source]; !ok {
				t.Fatalf("shard %d inbound source %d is not a corpus global id", s, e.Source)
			}
		}
	}
	t.Logf("routed inbound: shard0=%d shard1=%d (sums to %d cross edges)",
		len(gotInbound[0]), len(gotInbound[1]), crossCount)
}

func sortInbound(e []graph.InboundEdge) {
	sort.Slice(e, func(i, j int) bool {
		if e[i].Source != e[j].Source {
			return e[i].Source < e[j].Source
		}
		return e[i].Target < e[j].Target
	})
}
