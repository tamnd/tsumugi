package cli

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
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
		// With the full per-request deadline every contributing shard responds, so the
		// fan-out over real shards is complete: the count reached equals the count routed,
		// and the back-compatible Search returns the same top-k the completeness path does.
		sc := broker.SearchComplete(t.Context(), toQuery(pq, 10))
		if !sc.Complete() {
			t.Fatalf("query %q flagged partial over real shards: %d of %d reached", q, sc.ShardsOK, sc.ShardsTotal)
		}
		if sc.ShardsOK < 1 {
			t.Fatalf("query %q reached no shard", q)
		}
		hits := broker.Search(t.Context(), toQuery(pq, 10))
		if len(hits) != len(sc.Hits) {
			t.Fatalf("query %q: Search returned %d hits, SearchComplete %d", q, len(hits), len(sc.Hits))
		}
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

// TestAggregatorCCrawl builds a real multi-shard collection and serves it through a two-tier
// tree, an aggregator over two brokers that split the real shards, the topology a deployment
// reaches for when a corpus outgrows one broker. It checks the properties that hold on real
// data: the tree reports its whole shard count, a query is complete over the real shards
// with every contributing shard reached, the merged top-k is score-ordered, and every id
// falls inside the collection. It does not assert bit-exact equality against a single broker
// over all shards, because each sub-broker recomputes idf against its own half's df and N;
// fleet-wide df in GlobalStats (the still-open serving gap) is what would make the
// cross-broker scores identical, and the synthetic TestAggregatorExactOverBrokers covers the
// exactness path with a corpus that scores on features rather than idf.
func TestAggregatorCCrawl(t *testing.T) {
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
	if res.Shards < 4 {
		t.Fatalf("need at least 4 shards to split across two brokers, got %d", res.Shards)
	}

	modelPath := filepath.Join(tmp, "model.bin")
	writeModel(t, modelPath)

	f, err := os.Open(modelPath)
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	ens, err := rank.LoadEnsemble(f)
	_ = f.Close()
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	model := ens.Compile()

	// openShards opens every shard and returns them with a broker over all of them; the
	// shards are what the aggregator's sub-brokers are built from. The all-shards broker
	// owns the file mappings, so it is the one closed, and the sub-brokers reference the
	// same read-only shards without closing them a second time.
	shards, all, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = all.Close() }()
	pl := buildPipeline(shards)

	half := len(shards) / 2
	b0 := search.NewBroker(shards[:half], newCascade(model))
	b1 := search.NewBroker(shards[half:], newCascade(model))
	agg := search.NewAggregator([]search.Searcher{b0, b1})

	if agg.NumShards() != len(shards) {
		t.Fatalf("aggregator NumShards = %d, want %d", agg.NumShards(), len(shards))
	}

	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	total := 0
	for _, q := range queries {
		pq := pl.parse(q)
		if pq.Empty() {
			continue
		}
		sc := agg.SearchComplete(context.Background(), toQuery(pq, 10))
		if !sc.Complete() {
			t.Fatalf("query %q flagged partial over the real tree: %d of %d reached", q, sc.ShardsOK, sc.ShardsTotal)
		}
		if sc.ShardsTotal != len(shards) || sc.ShardsOK != len(shards) {
			t.Fatalf("query %q: reached %d of %d shards, want all %d", q, sc.ShardsOK, sc.ShardsTotal, len(shards))
		}
		for i := 1; i < len(sc.Hits); i++ {
			if sc.Hits[i-1].Score < sc.Hits[i].Score {
				t.Fatalf("query %q: merged top-k not score-ordered at %d", q, i)
			}
		}
		for _, h := range sc.Hits {
			if int(h.DocID) >= res.Docs {
				t.Fatalf("hit %d outside the collection's %d docs", h.DocID, res.Docs)
			}
		}
		total += len(sc.Hits)
	}
	if total == 0 {
		t.Fatalf("no common English query returned a hit through the tree over %d real docs", res.Docs)
	}
}
