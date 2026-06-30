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

// TestCrossReciprocityOnCCrawl runs the cross-shard reciprocity on the real crawl, the gate
// that proves the both-sides aggregation can come off the persisted per-shard regions. It uses
// the same setup as the cross-rank and in-edge tests, real canonical URLs, the spec's MPH
// global node ids, and the resulting explicit id tables split across two shards, then computes
// reciprocity two ways: once with CrossReciprocity over the shard regions, where every
// boundary-crossing out-edge and its possible back-link are matched in the global id space, and
// once with the single-graph Reciprocity over the merged graph. For a fixed out-degree the
// value is a ratio of integer counts, so the two must agree to floating-point exactness.
//
// As in the cross-rank test the link structure is generated rather than read from the bodies,
// because in-corpus links almost never resolve in a broad crawl sample; the ids, the id tables,
// and the dense assignment are all real, only which node points at which is synthetic. The
// far-edge generator deliberately seeds a back-link for some edges so the corpus carries
// genuine cross-shard mutual links, the case the merged transpose recovers and the sharded
// gather must too.
func TestCrossReciprocityOnCCrawl(t *testing.T) {
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
	type loc struct {
		page  int
		shard int
		dense int
	}
	denseGlobals := make([][]uint64, shards)
	locByGlobal := make(map[uint64]loc, n)
	for i, cu := range cus {
		s := i % shards
		g := m.Lookup([]byte(cu))
		d := len(denseGlobals[s])
		denseGlobals[s] = append(denseGlobals[s], g)
		locByGlobal[g] = loc{page: i, shard: s, dense: d}
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

	// addEdge mirrors one directed link into both the monolith and the shard builders.
	addEdge := func(src, tgt loc, tgGlobal uint64) {
		mb.AddEdge(src.page, tgt.page)
		if tgt.shard == src.shard {
			sb[src.shard].AddEdge(src.dense, tgt.dense)
		} else {
			sb[src.shard].AddCrossEdge(src.dense, tgGlobal)
		}
	}

	const fanout = 4
	var crossCount, mutualCount int
	for i := 0; i < n; i++ {
		src := locByGlobal[globalsSorted[i]]
		srcGlobal := globalsSorted[i]
		for k := 1; k <= fanout; k++ {
			tg := globalsSorted[(i*7+k*1300)%n]
			tgt := locByGlobal[tg]
			if tg == srcGlobal {
				continue
			}
			addEdge(src, tgt, tg)
			if tgt.shard != src.shard {
				crossCount++
			}
			// Seed a back-link on a third of the edges so the corpus carries genuine
			// mutual links, including across shard boundaries.
			if k%3 == 0 {
				addEdge(tgt, src, srcGlobal)
				if tgt.shard != src.shard {
					crossCount++
				}
				mutualCount++
			}
		}
	}
	if crossCount == 0 {
		t.Fatal("expected cross edges from the synthetic far-edge set")
	}
	if mutualCount == 0 {
		t.Fatal("expected seeded mutual links")
	}

	mono, err := graph.Open(mb.Build())
	if err != nil {
		t.Fatalf("open monolith: %v", err)
	}
	want := graph.Reciprocity(mono)

	regions := make([]*graph.Region, shards)
	for s := 0; s < shards; s++ {
		g, err := graph.Open(sb[s].Build())
		if err != nil {
			t.Fatalf("open shard %d: %v", s, err)
		}
		regions[s] = g
	}
	got := graph.CrossReciprocity(regions)

	var mis int
	var maxErr float64
	var nonzero int
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
	if mis != 0 {
		t.Fatalf("sharded vs monolith reciprocity mismatches: %d, max error %g", mis, maxErr)
	}
	t.Logf("pages=%d cross-edges=%d mutual-links=%d nonzero-reciprocity=%d max error %.2e",
		n, crossCount, mutualCount, nonzero, maxErr)
}
