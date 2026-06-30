package cli

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/search"
)

// TestServeExposesRPCLeaf checks the serve mux wiring that makes a broker reachable as a node
// of a serving tree spanning machines: the /rpc/ routes a RemoteSearcher dials are mounted on
// the same mux that serves the human /search, and a RemoteSearcher over them reproduces the
// in-process broker's answer. This is the end-to-end seam, the broker a head-node aggregator
// fans across, run over a real HTTP round trip through the serve handler.
func TestServeExposesRPCLeaf(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox", "lazy brown dog"}, 0)
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs", "swift brown hare"}, 1000)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, _, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	// Mount the RPC handler exactly as the serve command does, under /rpc/, so the test exercises
	// the real route prefix a peer dials rather than a bare handler.
	mux := http.NewServeMux()
	mux.Handle("/rpc/", http.StripPrefix("/rpc", search.NewSearcherHandler(broker)))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx := context.Background()
	rs, err := search.NewRemoteSearcher(ctx, ts.URL+"/rpc")
	if err != nil {
		t.Fatalf("dial remote leaf: %v", err)
	}
	if rs.NumShards() != broker.NumShards() {
		t.Fatalf("remote NumShards = %d, local %d", rs.NumShards(), broker.NumShards())
	}
	if rs.NumDocs() != broker.NumDocs() {
		t.Fatalf("remote NumDocs = %d, local %d", rs.NumDocs(), broker.NumDocs())
	}

	q := search.Query{Terms: []string{"brown"}, K: 4}
	want := broker.SearchComplete(ctx, q)
	got := rs.SearchComplete(ctx, q)
	if len(got.Hits) == 0 {
		t.Fatal("remote leaf returned no hits")
	}
	if len(got.Hits) != len(want.Hits) {
		t.Fatalf("remote leaf returned %d hits, local %d", len(got.Hits), len(want.Hits))
	}
	for i := range want.Hits {
		if got.Hits[i].DocID != want.Hits[i].DocID || got.Hits[i].Score != want.Hits[i].Score {
			t.Fatalf("hit %d: remote {%d,%v}, local {%d,%v}", i,
				got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].DocID, want.Hits[i].Score)
		}
	}
}
