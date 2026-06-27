package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
)

// ccrawlParquet is the real crawl export the serve path is exercised against, the same
// fixture the collection tests build from. A run without it present skips rather than
// fails, so the suite stays green on a machine that has not fetched the corpus.
const ccrawlParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// TestServeCCrawlPipeline builds a real multi-shard collection from the crawl export,
// loads it through the full serve path, and runs queries through the broker's query
// pipeline. It is the end-to-end proof that the wiring works on real data: the build
// stamps the analyzer hash the broker accepts, the pipeline is built from the open
// shards, and a fanned-out query returns ranked hits. The query set is a handful of
// common English content words, so at least one is expected to match real documents.
func TestServeCCrawlPipeline(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards to exercise the fan-out, got %d", res.Shards)
	}

	modelPath := filepath.Join(tmp, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(out, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()
	if broker.NumShards() != res.Shards {
		t.Fatalf("broker shards = %d, want %d", broker.NumShards(), res.Shards)
	}

	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	total := 0
	for _, q := range queries {
		pq := pl.parse(q)
		if pq.Empty() {
			continue
		}
		hits := broker.Search(t.Context(), toQuery(pq, 10))
		total += len(hits)
		// Every returned global id must fall inside the collection's id space, the proof
		// the fan-out shifted local ids into the global space by each shard's node base.
		for _, h := range hits {
			if int(h.DocID) >= res.Docs {
				t.Fatalf("hit %d outside the collection's %d docs", h.DocID, res.Docs)
			}
		}
	}
	if total == 0 {
		t.Fatalf("no common English query returned a hit over %d real docs", res.Docs)
	}
}
