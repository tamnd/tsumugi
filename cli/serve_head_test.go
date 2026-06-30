package cli

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/search"
)

// headOverLeaves wires a head node over the given leaf brokers exactly the way runHead does,
// but in process for a test: it stands an RPC server in front of each leaf, dials it as a
// RemoteSearcher, builds the head pipeline from the peers' vocabularies, fans across them with
// an aggregator, and wraps that in a result cache. It returns the head httpServer and a cleanup
// that closes the RPC servers, so a test drives the head's /search the same handler the command
// serves.
func headOverLeaves(t *testing.T, leaves []*search.Broker) (*httpServer, func()) {
	t.Helper()
	var servers []*httptest.Server
	var remotes []*search.RemoteSearcher
	var children []search.Searcher
	for _, b := range leaves {
		ts := httptest.NewServer(search.NewSearcherHandler(b))
		servers = append(servers, ts)
		rs, err := search.NewRemoteSearcher(context.Background(), ts.URL)
		if err != nil {
			ts.Close()
			t.Fatalf("dial leaf: %v", err)
		}
		remotes = append(remotes, rs)
		children = append(children, rs)
	}
	pl, verr := buildHeadPipeline(context.Background(), remotes)
	if len(verr) != 0 {
		t.Fatalf("head pipeline vocab errors: %v", verr)
	}
	agg := search.NewAggregator(children)
	backend := search.NewCachedSearcher(agg, search.NewResultCache(8))
	srv := &httpServer{backend: backend, pipeline: pl, timeout: 0}
	return srv, func() {
		for _, ts := range servers {
			ts.Close()
		}
	}
}

// tieGroups maps each distinct score in a ranked top-k to the sorted set of doc ids that hold
// it, so two rankings can be compared up to the order of documents that tie on a score. The
// merge is exact on which documents reach the top-k and on the score at each rank; the order
// within a tie is not pinned, so a comparison that wants exactness without flagging a reordered
// tie groups by score and compares the doc-id sets.
func tieGroups(hits []hitJSON) map[float64][]uint32 {
	g := map[float64][]uint32{}
	for _, h := range hits {
		g[h.Score] = append(g[h.Score], h.DocID)
	}
	for _, ids := range g {
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	}
	return g
}

// distinctScores reports whether every hit in a top-k carries a different score. Only then is a
// doc-id comparison sound: with no ties the top-k boundary cannot fall inside a tie group, so the
// set of documents returned is pinned rather than an arbitrary pick among equally scored ones.
func distinctScores(hits []hitJSON) bool {
	seen := map[float64]bool{}
	for _, h := range hits {
		if seen[h.Score] {
			return false
		}
		seen[h.Score] = true
	}
	return true
}

// equalTieGroups reports whether two tie-group maps hold the same doc ids at the same scores.
func equalTieGroups(a, b map[float64][]uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for score, ai := range a {
		bi, ok := b[score]
		if !ok || len(ai) != len(bi) {
			return false
		}
		for i := range ai {
			if ai[i] != bi[i] {
				return false
			}
		}
	}
	return true
}

// getSearchOn runs one query against a serve handler and decodes the response, standing a
// throwaway HTTP server in front of the handler so the request travels the real socket path.
func getSearchOn(t *testing.T, srv *httpServer, rawURL string) searchResponse {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	defer ts.Close()
	return getSearch(t, ts.URL+rawURL)
}

// TestServeHeadMatchesMonolith is the head-node correctness proof: a head fanning across two
// leaf nodes over the RPC seam returns, for every query, the same ranked top-k a single broker
// over all the shards returns. The corpus is split across the two leaves, so a correct answer
// has to merge candidates from both, and the head builds its corrector and dense plane from the
// peers' vocabularies rather than from any local shard, so this also proves the fleet-pipeline
// wiring reproduces the monolith's query understanding.
func TestServeHeadMatchesMonolith(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	// The same four shards in three layouts: all together for the monolith, and split two and
	// two across the leaves. Distinct node bases keep every document's global id unique across
	// the whole fleet, so a merged top-k never collides two leaves' ids.
	type shard struct {
		name  string
		texts []string
		base  uint32
	}
	shards := []shard{
		{"a.tsumugi", []string{"the quick brown fox", "lazy brown dog"}, 0},
		{"b.tsumugi", []string{"brown bear runs fast", "swift brown hare"}, 1000},
		{"c.tsumugi", []string{"green forest canopy", "brown owl at night"}, 2000},
		{"d.tsumugi", []string{"river runs brown", "fox and hare race"}, 3000},
	}

	monoDir := filepath.Join(dir, "mono")
	leaf1Dir := filepath.Join(dir, "leaf1")
	leaf2Dir := filepath.Join(dir, "leaf2")
	for _, d := range []string{monoDir, leaf1Dir, leaf2Dir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", d, err)
		}
	}
	for i, s := range shards {
		writeShard(t, filepath.Join(monoDir, s.name), s.texts, s.base)
		leafDir := leaf1Dir
		if i >= 2 {
			leafDir = leaf2Dir
		}
		writeShard(t, filepath.Join(leafDir, s.name), s.texts, s.base)
	}

	mono, monoPipe, err := openCollection(monoDir, modelPath)
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	defer func() { _ = mono.Close() }()
	monoSrv := &httpServer{backend: mono, pipeline: monoPipe, timeout: 0}

	leaf1, _, err := openCollection(leaf1Dir, modelPath)
	if err != nil {
		t.Fatalf("open leaf1: %v", err)
	}
	defer func() { _ = leaf1.Close() }()
	leaf2, _, err := openCollection(leaf2Dir, modelPath)
	if err != nil {
		t.Fatalf("open leaf2: %v", err)
	}
	defer func() { _ = leaf2.Close() }()

	headSrv, cleanup := headOverLeaves(t, []*search.Broker{leaf1, leaf2})
	defer cleanup()

	queries := []string{"?q=brown&k=8", "?q=fox&k=4", "?q=brown+fox&k=8", "?q=runs&k=4", "?q=hare&k=4"}
	nonTrivial := false
	for _, q := range queries {
		want := getSearchOn(t, monoSrv, q)
		got := getSearchOn(t, headSrv, q)
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("%s: head returned %d hits, mono %d", q, len(got.Hits), len(want.Hits))
		}
		if got.Shards != want.Shards {
			t.Fatalf("%s: head shards = %d, mono = %d", q, got.Shards, want.Shards)
		}
		if !got.Completeness.Complete {
			t.Fatalf("%s: head result not complete: %+v", q, got.Completeness)
		}
		var sawLeaf1, sawLeaf2 bool
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID || got.Hits[i].Score != want.Hits[i].Score {
				t.Fatalf("%s: hit %d head=%+v mono=%+v", q, i, got.Hits[i], want.Hits[i])
			}
			// Doc ids below 2000 are leaf1's two shards (base 0 and 1000), at or above are leaf2's
			// (base 2000 and 3000). A query whose merged top-k reaches into both id ranges proves the
			// head fanned across both leaves and merged their candidates rather than answering from one,
			// so the equality above is over a genuinely distributed result, not a single leaf's hits.
			if got.Hits[i].DocID < 2000 {
				sawLeaf1 = true
			} else {
				sawLeaf2 = true
			}
		}
		if sawLeaf1 && sawLeaf2 {
			nonTrivial = true
		}
	}
	if !nonTrivial {
		t.Fatal("no query merged hits from both leaves; the distributed comparison is vacuous")
	}
}

// TestServeHeadCachesRepeatedQuery checks the head's result cache serves a repeated query from
// cache, so the head-of-tree cache (slice 67) composes over the fleet: the first query fans
// across the leaves and the second is a cache hit reported as cached.
func TestServeHeadCachesRepeatedQuery(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	leaf1Dir := filepath.Join(dir, "leaf1")
	leaf2Dir := filepath.Join(dir, "leaf2")
	if err := os.MkdirAll(leaf1Dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.MkdirAll(leaf2Dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeShard(t, filepath.Join(leaf1Dir, "a.tsumugi"), []string{"brown fox", "brown dog"}, 0)
	writeShard(t, filepath.Join(leaf2Dir, "b.tsumugi"), []string{"brown bear", "brown hare"}, 1000)

	leaf1, _, err := openCollection(leaf1Dir, modelPath)
	if err != nil {
		t.Fatalf("open leaf1: %v", err)
	}
	defer func() { _ = leaf1.Close() }()
	leaf2, _, err := openCollection(leaf2Dir, modelPath)
	if err != nil {
		t.Fatalf("open leaf2: %v", err)
	}
	defer func() { _ = leaf2.Close() }()

	headSrv, cleanup := headOverLeaves(t, []*search.Broker{leaf1, leaf2})
	defer cleanup()

	first := getSearchOn(t, headSrv, "?q=brown&k=4")
	if first.Cached {
		t.Fatal("first query reported cached")
	}
	second := getSearchOn(t, headSrv, "?q=brown&k=4")
	if !second.Cached {
		t.Fatal("repeated query not served from cache")
	}
	if len(first.Hits) != len(second.Hits) {
		t.Fatalf("cached hit count %d != cold %d", len(second.Hits), len(first.Hits))
	}
	for i := range first.Hits {
		if first.Hits[i] != second.Hits[i] {
			t.Fatalf("cached hit %d = %+v, cold = %+v", i, second.Hits[i], first.Hits[i])
		}
	}
}
