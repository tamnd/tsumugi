package collection

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/graph"
)

// TestGlobalRanksResolvesCrossHostLinks proves the collection-wide pass resolves
// links between documents on different hosts, the edges a per-shard graph drops.
// Ten spoke pages on one host each link to a single hub page on another host. The
// hub draws all ten in-links and must rank strictly above a page that nothing
// links to, and the resolved edge count must be the ten cross-host links, not the
// zero a single shard would see.
func TestGlobalRanksResolvesCrossHostLinks(t *testing.T) {
	const spokes = 10
	var docs []convert.Document
	// The hub and an unlinked page share the hub's host.
	docs = append(docs,
		convert.Document{URL: "https://a.example/hub", Host: "a.example", Body: "# Hub"},
		convert.Document{URL: "https://a.example/lonely", Host: "a.example", Body: "# Lonely\nno links here"},
	)
	// Every spoke lives on a second host and links to the hub.
	for i := 0; i < spokes; i++ {
		docs = append(docs, convert.Document{
			URL:  fmt.Sprintf("https://b.example/s%d", i),
			Host: "b.example",
			Body: fmt.Sprintf("# Spoke %d\nsee <https://a.example/hub>", i),
		})
	}

	if got := globalEdgeCount(docs); got != spokes {
		t.Fatalf("resolved edges = %d, want %d cross-host links", got, spokes)
	}

	ranks := globalRanks(docs)
	if len(ranks) != len(docs) {
		t.Fatalf("ranks length %d != doc count %d", len(ranks), len(docs))
	}
	var sum float64
	for _, r := range ranks {
		sum += r
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("ranks sum to %g, want ~1", sum)
	}
	// docs[0] is the hub, docs[1] the lonely page on the same host.
	if ranks[0] <= ranks[1] {
		t.Fatalf("hub rank %g should beat the unlinked page %g", ranks[0], ranks[1])
	}
}

// TestBuildBakesCrossShardPageRank proves the cross-shard rank reaches the feature
// matrix the serving path actually reads. The hub sits on one host and the spokes
// on another, and a shard size cut at the host boundary lands the hub and the
// spokes in different shards, so every spoke-to-hub link is a genuine cross-shard
// edge. After the build, the hub's FeatPageRank read back from its shard must be
// non-zero and above the unlinked page in the same shard, which can only happen if
// the in-links were resolved across the shard boundary.
func TestBuildBakesCrossShardPageRank(t *testing.T) {
	const spokes = 10
	tmp := t.TempDir()
	src := filepath.Join(tmp, "crossshard.jsonl")
	f, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	// Host a.example: the hub first (url sorts before "lonely"), then the unlinked
	// page. Two documents, so a shard size of two cuts the shard boundary exactly
	// at the host boundary.
	writeRec(t, f, "https://a.example/hub", "a.example", "# Hub")
	writeRec(t, f, "https://a.example/lonely", "a.example", "# Lonely\nplain page")
	for i := 0; i < spokes; i++ {
		writeRec(t, f, fmt.Sprintf("https://b.example/s%d", i), "b.example",
			fmt.Sprintf("# Spoke %d\nsee <https://a.example/hub>", i))
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	out := filepath.Join(tmp, "col")
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 2}); err != nil {
		t.Fatalf("build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(shards) < 2 {
		t.Fatalf("want the hosts split across at least 2 shards, got %d", len(shards))
	}

	// The first shard holds host a.example: docID 0 is the hub, docID 1 the
	// unlinked page, because both share a host and the hub's url sorts first.
	r, err := tsumugi.Open(shards[0].Path)
	if err != nil {
		t.Fatalf("open shard 0: %v", err)
	}
	defer func() { _ = r.Close() }()
	fb, err := r.Region(tsumugi.RegionFeature)
	if err != nil {
		t.Fatalf("read feature region: %v", err)
	}
	fr, err := feature.Open(fb)
	if err != nil {
		t.Fatalf("open feature region: %v", err)
	}

	hubRank, ok := fr.Value(0, feature.FeatPageRank)
	if !ok {
		t.Fatal("hub has no PageRank feature")
	}
	lonelyRank, _ := fr.Value(1, feature.FeatPageRank)
	if hubRank <= 0 {
		t.Fatalf("hub PageRank baked into the feature matrix is %g, want > 0", hubRank)
	}
	if hubRank <= lonelyRank {
		t.Fatalf("hub rank %g should beat the unlinked page %g across the shard boundary", hubRank, lonelyRank)
	}
}

// refRanksViaMap recomputes the collection PageRank with a plain map directory, the
// straightforward reference the minimal-perfect-hash directory must match exactly.
func refRanksViaMap(docs []convert.Document) []float64 {
	urlToID := make(map[string]int, len(docs))
	for i, d := range docs {
		if cu, ok := analyze.CanonicalURL(d.URL); ok {
			if _, dup := urlToID[cu]; !dup {
				urlToID[cu] = i
			}
		}
	}
	gb := graph.NewBuilder(len(docs))
	for i, d := range docs {
		for _, tgt := range analyze.Links(d) {
			if j, ok := urlToID[tgt]; ok && j != i {
				gb.AddEdge(i, j)
			}
		}
	}
	g, err := graph.Open(gb.Build())
	if err != nil {
		panic(err)
	}
	return graph.PageRank(g, graph.DefaultPRConfig())
}

// TestMPHDirMatchesMapOnCCrawl proves the minimal-perfect-hash directory is a
// drop-in for the plain map: over the whole real collection the ranks it produces
// are bit-for-bit identical to the map's, so the few-bits-a-key directory changes
// the build's memory, not its output. The membership fingerprint must reject every
// one of the many link targets the crawl did not capture, or a spurious edge would
// shift a rank and the vectors would diverge.
func TestMPHDirMatchesMapOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	// Compare the in-core PageRank over the MPH directory's graph against the in-core
	// PageRank over the map's graph: same algorithm both sides, so any difference is
	// the directory resolving an edge the map does not (or missing one). globalRanks
	// itself now streams the rank in float32; it is checked against the in-core path
	// separately in TestGlobalRanksStreamMatchesInCore, so this directory equality
	// stays an in-core float64 comparison where a one-ULP streaming difference cannot
	// mask an edge divergence.
	want := refRanksViaMap(docs)
	got := graph.PageRank(buildGraph(docs, buildDir(docs)), graph.DefaultPRConfig())
	if len(got) != len(want) {
		t.Fatalf("length %d != reference %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rank[%d] = %v, map reference = %v; the MPH directory diverged", i, got[i], want[i])
		}
	}
	t.Logf("docs=%d ranks bit-for-bit identical to the plain-map directory", len(docs))
}

// TestGlobalRanksStreamMatchesInCore proves the build's production rank path, which
// streams PageRank out of core through graph.StreamPageRankP and widens to float64,
// agrees with the in-core graph.PageRank over the same collection graph to float32
// precision. The build uses the streamed path so the rank vector, not the whole
// transpose CSR, is the resident set at corpus scale; this gate is what lets that
// swap stay invisible to every downstream signal that reads the ranks.
func TestGlobalRanksStreamMatchesInCore(t *testing.T) {
	const spokes = 40
	var docs []convert.Document
	docs = append(docs,
		convert.Document{URL: "https://a.example/hub", Host: "a.example", Body: "# Hub"},
		convert.Document{URL: "https://a.example/lonely", Host: "a.example", Body: "# Lonely"},
	)
	for i := 0; i < spokes; i++ {
		// Spokes link to the hub and to the next spoke, so the graph has real
		// structure (cycles, varied in and out degree) rather than a pure star.
		docs = append(docs, convert.Document{
			URL:  fmt.Sprintf("https://b.example/s%d", i),
			Host: "b.example",
			Body: fmt.Sprintf("# Spoke %d\nsee <https://a.example/hub> and <https://b.example/s%d>", i, (i+1)%spokes),
		})
	}

	g := buildGraph(docs, buildDir(docs))
	want := graph.PageRank(g, graph.DefaultPRConfig())
	got := globalRanks(docs)
	if len(got) != len(want) {
		t.Fatalf("streamed length %d != in-core %d", len(got), len(want))
	}
	var worst float64
	for i := range want {
		d := want[i] - got[i]
		if d < 0 {
			d = -d
		}
		if d > worst {
			worst = d
		}
	}
	if worst > 1e-5 {
		t.Fatalf("streamed global ranks diverge from in-core by %g, want < 1e-5", worst)
	}
}

// TestGlobalGraphOnCCrawl records what the collection-wide resolution recovers on
// the real crawl. M15a found zero links resolve within a single shard on a broad
// crawl sample and attributed it to the web graph being cross-shard. This test
// resolves links over the whole collection, removing sharding from the picture
// entirely, and finds the graph still does not materialize: a broad Common Crawl
// sample is one page per host across many hosts, so a page's link targets are
// specific pages on other hosts that the crawl did not itself capture. The test
// quantifies the crux, that the target's host is often in the collection but the
// target's exact page almost never is, so the milestone's real payoff needs a
// crawl with depth (many pages per host, which ami's seed-and-recrawl produces)
// rather than this breadth-first sample. It asserts only the property the signal
// stage depends on, that the rank vector normalizes, and logs the yield.
func TestGlobalGraphOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	// Build the collection's URL set and host set so the resolution gap can be
	// split into "host present" versus "exact page present".
	urlSet := make(map[string]struct{}, len(docs))
	hostSet := map[string]struct{}{}
	for _, d := range docs {
		if cu, ok := analyze.CanonicalURL(d.URL); ok {
			urlSet[cu] = struct{}{}
			if u, err := url.Parse(cu); err == nil {
				hostSet[u.Host] = struct{}{}
			}
		}
	}

	var totalLinks, hostPresent, pagePresent int
	for _, d := range docs {
		for _, tgt := range analyze.Links(d) {
			totalLinks++
			if u, err := url.Parse(tgt); err == nil {
				if _, ok := hostSet[u.Host]; ok {
					hostPresent++
				}
			}
			if _, ok := urlSet[tgt]; ok {
				pagePresent++
			}
		}
	}

	ranks := globalRanks(docs)
	if len(ranks) != len(docs) {
		t.Fatalf("ranks length %d != doc count %d", len(ranks), len(docs))
	}
	var sum float64
	for _, r := range ranks {
		sum += r
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("collection ranks sum to %g, want ~1", sum)
	}
	t.Logf("docs=%d hosts=%d links=%d targetHostInCollection=%d targetPageInCollection=%d edges=%d",
		len(docs), len(hostSet), totalLinks, hostPresent, pagePresent, globalEdgeCount(docs))
	t.Log("a broad crawl carries the target hosts but not the target pages, so the graph needs crawl depth, not just collection-wide resolution")
}
