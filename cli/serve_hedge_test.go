package cli

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/search"
)

// slowSearchHandler wraps an RPC handler and adds a fixed delay to its /search route only,
// leaving /meta and /vocab fast so construction and the head pipeline build are not slowed. It
// respects the request context so a hedge loser whose request is cancelled stops waiting, the
// test stand-in for a replica on a slow host reached over the real wire.
func slowSearchHandler(inner http.Handler, delay time.Duration) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/search") {
			select {
			case <-time.After(delay):
			case <-r.Context().Done():
				return
			}
		}
		inner.ServeHTTP(w, r)
	})
}

// headOverHedgedLeaves wires a head over the given leaf brokers the way runHead does for a
// multi-replica --peer group: each leaf is stood up behind two RPC servers, a slow one and a fast
// one, dialed as two RemoteSearchers and made one child through a HedgedSearcher, so the head fans
// across hedged replica sets. The pipeline is fed the first (slow) replica of each group, matching
// runHead, which is sound because /vocab is not slowed and the replicas share a vocabulary.
func headOverHedgedLeaves(t *testing.T, leaves []*search.Broker, hedgeDelay, slowDelay time.Duration) (*httpServer, func()) {
	t.Helper()
	var servers []*httptest.Server
	var remotes []*search.RemoteSearcher
	var children []search.Searcher
	for _, b := range leaves {
		slowSrv := httptest.NewServer(slowSearchHandler(search.NewSearcherHandler(b), slowDelay))
		fastSrv := httptest.NewServer(search.NewSearcherHandler(b))
		servers = append(servers, slowSrv, fastSrv)
		slowRS, err := search.NewRemoteSearcher(context.Background(), slowSrv.URL)
		if err != nil {
			t.Fatalf("dial slow replica: %v", err)
		}
		fastRS, err := search.NewRemoteSearcher(context.Background(), fastSrv.URL)
		if err != nil {
			t.Fatalf("dial fast replica: %v", err)
		}
		remotes = append(remotes, slowRS) // the first replica feeds the pipeline, as in runHead
		children = append(children, search.NewHedgedSearcher([]search.Searcher{slowRS, fastRS}, hedgeDelay))
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

// TestServeHeadHedgedMatchesMonolith is the end-to-end hedging proof at the serve layer: a head
// over two leaves, each leaf hedged across a slow replica and a fast one, returns for every query
// the exact ranked top-k a single broker over all the shards returns, and returns it well under
// the slow replica's latency. So hedging changes only which replica answers, never the answer, and
// the fast replica hides the slow one's tail across the whole distributed query.
func TestServeHeadHedgedMatchesMonolith(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

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

	const slowDelay = 300 * time.Millisecond
	headSrv, cleanup := headOverHedgedLeaves(t, []*search.Broker{leaf1, leaf2}, 5*time.Millisecond, slowDelay)
	defer cleanup()

	queries := []string{"?q=brown&k=8", "?q=fox&k=4", "?q=brown+fox&k=8", "?q=runs&k=4", "?q=hare&k=4"}
	nonTrivial := false
	start := time.Now()
	for _, q := range queries {
		want := getSearchOn(t, monoSrv, q)
		got := getSearchOn(t, headSrv, q)
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("%s: hedged head returned %d hits, mono %d", q, len(got.Hits), len(want.Hits))
		}
		if got.Shards != want.Shards {
			t.Fatalf("%s: hedged head shards = %d, mono = %d", q, got.Shards, want.Shards)
		}
		if !got.Completeness.Complete {
			t.Fatalf("%s: hedged head result not complete: %+v", q, got.Completeness)
		}
		var sawLeaf1, sawLeaf2 bool
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID || got.Hits[i].Score != want.Hits[i].Score {
				t.Fatalf("%s: hit %d hedged head=%+v mono=%+v", q, i, got.Hits[i], want.Hits[i])
			}
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
	elapsed := time.Since(start)
	if !nonTrivial {
		t.Fatal("no query merged hits from both hedged leaves; the distributed comparison is vacuous")
	}
	// Every query had to wait out the slow replica if hedging failed; the fast replica should have
	// answered each in a few milliseconds, so all five together stay well under one slow delay.
	if elapsed > slowDelay {
		t.Fatalf("five hedged queries took %v, longer than one slow replica's %v: the hedge did not fire", elapsed, slowDelay)
	}
	t.Logf("hedged head over two slow-plus-fast leaf replica sets reproduced the monolith on %d queries in %v (slow replica %v)",
		len(queries), elapsed, slowDelay)
}
