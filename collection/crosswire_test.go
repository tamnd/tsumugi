package collection

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
)

// writeCrossLinkedJSONL writes a deep, multi-host crawl where pages link across
// hosts. The build groups each host into a contiguous id range, so once the shards
// are cut at a small granularity a link to another host usually lands in another
// shard, which is the cross-shard edge the build now resolves and the shard's graph
// region carries. Every link points at a real page in the corpus, so the links
// resolve (unlike a broad crawl sample, which resolves almost none), which is what
// lets the cross-shard path be exercised at all.
func writeCrossLinkedJSONL(t testing.TB, dir, name string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	const domains, hostsPer, pagesPer = 6, 3, 8
	type pg struct{ host, url string }
	var pages []pg
	for d := 0; d < domains; d++ {
		for h := 0; h < hostsPer; h++ {
			host := fmt.Sprintf("h%d.dom%d.com", h, d)
			for p := 0; p < pagesPer; p++ {
				pages = append(pages, pg{host: host, url: fmt.Sprintf("https://%s/p%d", host, p)})
			}
		}
	}
	// Each page links to a few pages on other hosts, deterministic far targets.
	for i, p := range pages {
		body := fmt.Sprintf("# %s\n", p.url)
		for k := 1; k <= 4; k++ {
			tgt := pages[(i+k*13)%len(pages)]
			if tgt.host != p.host {
				body += fmt.Sprintf("see <%s> ", tgt.url)
			}
		}
		writeRec(t, f, p.url, p.host, body)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// buildLayout replays the build's deterministic ordering and id assignment over a
// source, returning the documents in their final dense order, the partition global
// id of each, and the canonical-URL directory. Every step (the host+url sort, the
// Recursive Graph Bisection order, AssignGlobalIDs, buildDir) is the same function
// the build calls, so the layout matches the shards on disk byte for byte, which is
// what lets a test know which shard each endpoint of a link lands in and what global
// id a source carries.
func buildLayout(t testing.TB, src string) ([]convert.Document, []uint64, func([]byte) (uint32, bool)) {
	t.Helper()
	docs, _, err := readSource(src, 0)
	if err != nil {
		t.Fatalf("read source: %v", err)
	}
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Host != docs[j].Host {
			return docs[i].Host < docs[j].Host
		}
		return docs[i].URL < docs[j].URL
	})
	order := collectionOrder(docs)
	reordered := make([]convert.Document, len(docs))
	for newID, oldID := range order {
		reordered[newID] = docs[oldID]
	}
	docs = reordered
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())
	dir := buildDir(docs)
	return docs, gids, dir.Lookup
}

// loadShardGraphs opens every shard's graph region in shard (NodeBase) order, the
// order RouteCrossEdges and StreamCrossPageRank index their results by.
func loadShardGraphs(t testing.TB, out string) ([]*graph.Region, func()) {
	t.Helper()
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	regions := make([]*graph.Region, len(shards))
	readers := make([]*tsumugi.Reader, 0, len(shards))
	for i, s := range shards {
		r, err := tsumugi.Open(s.Path)
		if err != nil {
			t.Fatalf("open shard: %v", err)
		}
		readers = append(readers, r)
		gb, err := r.Region(tsumugi.RegionGraph)
		if err != nil {
			t.Fatalf("read graph region: %v", err)
		}
		g, err := graph.Open(gb)
		if err != nil {
			t.Fatalf("open graph region: %v", err)
		}
		regions[i] = g
	}
	return regions, func() {
		for _, r := range readers {
			_ = r.Close()
		}
	}
}

// TestCrossShardEdgesWiredFromBuild proves the build now wires the cross-shard edges
// the milestone added (slices 40 and 41) into the per-shard graph regions: it builds
// a multi-shard collection over a corpus whose links cross hosts, then routes the
// shards' cross-shard edge lists through their partition-global-id tables with
// graph.RouteCrossEdges and checks the recovered inbound edge set against a ground
// truth computed independently from the same documents. Before this slice the build
// dropped every cross-shard link; now each shard carries them keyed by the target's
// global node id, and the join recovers exactly the inbound edges the link structure
// implies.
func TestCrossShardEdgesWiredFromBuild(t *testing.T) {
	tmp := t.TempDir()
	src := writeCrossLinkedJSONL(t, tmp, "web.jsonl")
	out := filepath.Join(tmp, "col")
	const shardSize = 40
	res, err := Build(Options{Source: src, Out: out, ShardSize: shardSize})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("want a multi-shard collection to exercise cross-shard edges, got %d shards", res.Shards)
	}

	docs, gids, lookup := buildLayout(t, src)
	shardOf := func(dense int) int { return dense / shardSize }
	shardStart := func(s int) int { return s * shardSize }

	// Ground truth: every resolved link whose endpoints land in different shards is an
	// inbound edge into the target's shard, keyed by the source's global id and the
	// target's dense docID within its shard. Dedupe per source target the way the
	// region's cross-shard list does.
	type ib struct {
		src uint64
		tgt int
	}
	want := make(map[int]map[ib]bool)
	for u, d := range docs {
		seen := make(map[int]bool)
		for _, tgt := range analyze.Links(d) {
			v, ok := lookup([]byte(tgt))
			if !ok || int(v) == u || seen[int(v)] {
				continue
			}
			seen[int(v)] = true
			su, sv := shardOf(u), shardOf(int(v))
			if su == sv {
				continue
			}
			if want[sv] == nil {
				want[sv] = make(map[ib]bool)
			}
			want[sv][ib{src: gids[u], tgt: int(v) - shardStart(sv)}] = true
		}
	}

	regions, closeAll := loadShardGraphs(t, out)
	defer closeAll()

	got := graph.RouteCrossEdges(regions)
	var total int
	for _, e := range got {
		total += len(e)
	}
	if total == 0 {
		t.Fatal("the build produced no cross-shard edges, so the corpus did not exercise the wired path")
	}

	for s := range regions {
		w := want[s]
		gotSet := make(map[ib]bool, len(got[s]))
		for _, e := range got[s] {
			k := ib{src: e.Source, tgt: e.Target}
			if gotSet[k] {
				t.Fatalf("shard %d: duplicate routed inbound edge %+v", s, k)
			}
			gotSet[k] = true
		}
		if len(gotSet) != len(w) {
			t.Fatalf("shard %d: routed %d inbound edges, want %d", s, len(gotSet), len(w))
		}
		for k := range w {
			if !gotSet[k] {
				t.Fatalf("shard %d: missing inbound edge %+v", s, k)
			}
		}
	}
	t.Logf("docs=%d shards=%d cross-shard edges routed=%d", res.Docs, res.Shards, total)
}

// TestCrossShardPageRankFromBuild proves the wired per-shard graphs rank correctly
// across shard boundaries: it runs the global-then-scatter cross-shard PageRank loop
// (slice 42) over the shard graph regions the build wrote and checks the result
// equals, node for node, the single-graph PageRank over the same collection graph.
// Slice 42 proved the loop equal to the monolith over hand-built shards; this proves
// it over shards produced by the real build pipeline, so the build's id tables,
// cross-shard edge lists, and dense layout all line up with what the loop expects.
func TestCrossShardPageRankFromBuild(t *testing.T) {
	tmp := t.TempDir()
	src := writeCrossLinkedJSONL(t, tmp, "web.jsonl")
	out := filepath.Join(tmp, "col")
	const shardSize = 40
	if _, err := Build(Options{Source: src, Out: out, ShardSize: shardSize}); err != nil {
		t.Fatalf("build: %v", err)
	}

	docs, _, _ := buildLayout(t, src)
	mono := buildGraph(docs, buildDir(docs))

	cfg := graph.PRConfig{Alpha: 0.85, MaxIters: 300, Tol: 1e-9}
	monoRank := graph.StreamPageRankP(mono, graph.OutDegrees(mono), graph.UniformTeleport(mono.NodeCount()), cfg)

	regions, closeAll := loadShardGraphs(t, out)
	defer closeAll()
	crossRank := graph.StreamCrossPageRank(regions, cfg)

	var maxErr float64
	for s := range regions {
		base := s * shardSize
		for d := 0; d < regions[s].NodeCount(); d++ {
			diff := math.Abs(float64(crossRank[s][d]) - float64(monoRank[base+d]))
			if diff > maxErr {
				maxErr = diff
			}
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("sharded cross-shard PageRank diverges from the monolith: maxErr %g", maxErr)
	}

	var sum float64
	for s := range crossRank {
		for _, v := range crossRank[s] {
			sum += float64(v)
		}
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("cross-shard PageRank sums to %g, want ~1", sum)
	}
	t.Logf("docs=%d shards=%d maxErr=%g sum=%g", len(docs), len(regions), maxErr, sum)
}
