package collection

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/graph"
)

// ccrawlGraphParquet is the real crawl export the graph build is exercised against.
// The internal test package cannot see the external test package's constant, so it
// names the path here; the test skips when the file is absent.
const ccrawlGraphParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// writeLinkedJSONL writes a star-shaped crawl into a .jsonl file: a hub page that
// every spoke links to, and spokes that each link only to the hub. The bodies carry
// the links as markdown autolinks, the form the crawl export preserves, so the build
// recovers the graph the same way it would from a real crawl. All pages share a host
// so they land in one shard, which is what makes the edges intra-shard.
func writeLinkedJSONL(t *testing.T, dir, name string, spokes int) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	const host = "hub.example"
	hubURL := fmt.Sprintf("https://%s/hub", host)

	// The hub links out to every spoke.
	var hubBody string
	for i := 0; i < spokes; i++ {
		hubBody += fmt.Sprintf("a link <https://%s/s%d> here ", host, i)
	}
	writeRec(t, f, hubURL, host, "# Hub\n"+hubBody)

	// Each spoke links only to the hub.
	for i := 0; i < spokes; i++ {
		body := fmt.Sprintf("# Spoke %d\nback to <%s> the hub", i, hubURL)
		writeRec(t, f, fmt.Sprintf("https://%s/s%d", host, i), host, body)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeRec(t *testing.T, f *os.File, url, host, body string) {
	t.Helper()
	// Encode the body as a JSON string so the markdown newline and quotes are safe.
	line := fmt.Sprintf(`{"url":%q,"host":%q,"markdown":%q}`+"\n", url, host, body)
	if _, err := f.WriteString(line); err != nil {
		t.Fatal(err)
	}
}

// openOnlyShardGraph builds the collection, opens its single shard, and returns the
// parsed graph region.
func openOnlyShardGraph(t *testing.T, src, out string) *graph.Region {
	t.Helper()
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 100000}); err != nil {
		t.Fatalf("build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("want 1 shard, got %d", len(shards))
	}
	r, err := tsumugi.Open(shards[0].Path)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	t.Cleanup(func() { _ = r.Close() })
	if !r.HasRegion(tsumugi.RegionGraph) {
		t.Fatal("shard has no graph region")
	}
	gb, err := r.Region(tsumugi.RegionGraph)
	if err != nil {
		t.Fatalf("read graph region: %v", err)
	}
	g, err := graph.Open(gb)
	if err != nil {
		t.Fatalf("open graph region: %v", err)
	}
	return g
}

// TestShardGraphStarShape proves the build recovers the link graph from the page
// bodies and lands it in the shard's graph region. The star is the cleanest case to
// assert on: the hub has in-degree and out-degree equal to the spoke count, each
// spoke has in-degree one (from the hub) and out-degree one (to the hub), and
// PageRank ranks the hub strictly above every spoke because the spokes' mass all
// flows to it while the hub's mass splits back across them. Every page shares the
// host, so the build groups them in one shard and every edge is intra-shard, which
// is what lets the graph materialize here.
func TestShardGraphStarShape(t *testing.T) {
	const spokes = 20
	tmp := t.TempDir()
	src := writeLinkedJSONL(t, tmp, "star.jsonl", spokes)
	g := openOnlyShardGraph(t, src, filepath.Join(tmp, "col"))

	if g.NodeCount() != spokes+1 {
		t.Fatalf("node count = %d, want %d", g.NodeCount(), spokes+1)
	}
	if g.EdgeCount() != uint64(2*spokes) {
		t.Fatalf("edge count = %d, want %d (hub->spoke and spoke->hub)", g.EdgeCount(), 2*spokes)
	}

	// The build orders documents by url within the host, so the id is not fixed.
	// Find the node whose in-degree is the spoke count; it is the hub.
	hub := -1
	for v := 0; v < g.NodeCount(); v++ {
		if g.InDegree(v) == spokes {
			if hub != -1 {
				t.Fatalf("more than one node has in-degree %d", spokes)
			}
			hub = v
		}
	}
	if hub == -1 {
		t.Fatalf("no node has the hub's in-degree %d", spokes)
	}
	if g.OutDegree(hub) != spokes {
		t.Fatalf("hub out-degree = %d, want %d", g.OutDegree(hub), spokes)
	}

	pr := graph.PageRank(g, graph.DefaultPRConfig())
	for v := 0; v < g.NodeCount(); v++ {
		if v == hub {
			continue
		}
		if g.InDegree(v) != 1 {
			t.Fatalf("spoke %d in-degree = %d, want 1 (the hub's link)", v, g.InDegree(v))
		}
		if g.OutDegree(v) != 1 {
			t.Fatalf("spoke %d out-degree = %d, want 1 (back to the hub)", v, g.OutDegree(v))
		}
		if pr[hub] <= pr[v] {
			t.Fatalf("hub rank %g should beat spoke %d rank %g", pr[hub], v, pr[v])
		}
	}
}

// TestShardGraphOnCCrawl builds a shard from the real crawl and records what the
// link graph actually looks like on a broad crawl sample, which is the finding that
// motivates the cross-shard milestone. The build extracts tens of thousands of
// outbound links from the page bodies, but a broad crawl samples roughly one page
// per host across thousands of hosts, so almost no link target is itself in the
// sample and the intra-shard graph is near-empty. The real graph only materializes
// once links resolve across the whole collection through a collection-wide node id,
// which is the next milestone. This test asserts what holds today, that the graph
// region is present and valid, its node count matches the shard, and its edge_count
// stat agrees with the region, and logs the intra-shard edge yield so the
// cross-shard gap is on the record rather than asserted away.
func TestShardGraphOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "col")
	res, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 100000, Limit: 8000})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("want 1 shard for the limit, got %d", len(shards))
	}
	r, err := tsumugi.Open(shards[0].Path)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = r.Close() }()

	gb, err := r.Region(tsumugi.RegionGraph)
	if err != nil {
		t.Fatalf("read graph region: %v", err)
	}
	g, err := graph.Open(gb)
	if err != nil {
		t.Fatalf("open graph region: %v", err)
	}
	if g.NodeCount() != res.Docs {
		t.Fatalf("graph node count %d != shard doc count %d", g.NodeCount(), res.Docs)
	}

	stat, _ := r.Stat(tsumugi.StatEdgeCount)
	if uint64(stat) != g.EdgeCount() {
		t.Fatalf("edge_count stat %.0f != region edge count %d", stat, g.EdgeCount())
	}

	// PageRank runs on whatever graph the sample yields and must always return a
	// normalized distribution, the property the offline signal stage depends on,
	// edges or no edges.
	pr := graph.PageRank(g, graph.DefaultPRConfig())
	if len(pr) != g.NodeCount() {
		t.Fatalf("pagerank length %d != node count %d", len(pr), g.NodeCount())
	}
	var sum float64
	for _, x := range pr {
		sum += x
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("pagerank sums to %g, want ~1", sum)
	}
	t.Logf("docs=%d intraShardEdges=%d (a broad crawl sample resolves few links in-shard; cross-shard resolution is the next milestone)",
		res.Docs, g.EdgeCount())
}
