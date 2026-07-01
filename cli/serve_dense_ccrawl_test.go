package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
)

// TestServeDenseCCrawl is the end-to-end activation gate: a collection built with a kept
// dimension serves through the same openCollection path the CLI uses, and the dense plane
// turns on by itself. It builds real crawl shards with a vector region, opens them, and
// checks the broker reports the dense dimension the shards agree on, the query pipeline
// wires an encoder so a parsed query carries a dense vector, and search still returns hits
// with the dense plane live. Before the build emitted vectors this whole plane was inert;
// this proves emitting the region is all it took to light it up, no serving change.
func TestServeDenseCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000, DenseDim: 256})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards, got %d", res.Shards)
	}

	modelPath := filepath.Join(tmp, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(out, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	// The dense plane is on because every shard carries a vector region and they agree on
	// the dimension the build embedded at.
	dim, ok := broker.VectorDim()
	if !ok || dim != 256 {
		t.Fatalf("broker VectorDim() = %d, %v; want 256, true (the dense plane did not turn on)", dim, ok)
	}

	// The pipeline built its dense encoder from the shards' dimension, so a parsed query
	// carries a dense vector the shards can recall against.
	denseQueries := 0
	total := 0
	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	for _, q := range queries {
		pq := pl.parse(q)
		if pq.Empty() {
			continue
		}
		if len(pq.DenseVec) > 0 {
			denseQueries++
		}
		hits := broker.Search(t.Context(), toQuery(pq, 10))
		total += len(hits)
		for _, h := range hits {
			if int(h.DocID) >= res.Docs {
				t.Fatalf("hit %d outside the collection's %d docs", h.DocID, res.Docs)
			}
		}
	}
	if denseQueries == 0 {
		t.Fatal("no query carried a dense vector though the dense plane is on")
	}
	if total == 0 {
		t.Fatalf("no query returned a hit with the dense plane live over %d docs", res.Docs)
	}
}
