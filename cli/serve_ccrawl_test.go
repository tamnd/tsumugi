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

// TestDegradeCCrawl exercises the degradation ladder on real data: it builds a real
// multi-shard collection and runs every rung of the ladder, checking that the
// full-quality rung matches the plain completeness path bit for bit, that every rung
// returns a ranked top-k within the collection, that the chosen rung is reported back,
// and that the shard-dropping rung queries strictly fewer shards. This is the proof the
// degradation order holds on the real corpus the engine gates on, not just synthetic
// data, where the static-rank spread and the language distribution are the real ones.
func TestDegradeCCrawl(t *testing.T) {
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
		t.Fatalf("need at least 4 shards to see the shard-dropping rung, got %d", res.Shards)
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

	shards, b, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = b.Close() }()
	pl := buildPipeline(shards)

	levels := []search.DegradeLevel{
		search.DegradeNone,
		search.DegradeL0,
		search.DegradeDense,
		search.DegradeShards,
		search.DegradeL2,
	}
	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	hitsSeen := 0
	for _, qs := range queries {
		pq := pl.parse(qs)
		if pq.Empty() {
			continue
		}
		q := toQuery(pq, 10)
		plain := b.SearchComplete(context.Background(), q)

		for _, level := range levels {
			deg := b.SearchDegraded(context.Background(), q, level)
			if deg.Degraded != level {
				t.Fatalf("query %q level %v: reported %v", qs, level, deg.Degraded)
			}
			if !deg.Complete() {
				t.Fatalf("query %q level %v: flagged partial, %d of %d shards reached", qs, level, deg.ShardsOK, deg.ShardsTotal)
			}
			for i := 1; i < len(deg.Hits); i++ {
				if deg.Hits[i-1].Score < deg.Hits[i].Score {
					t.Fatalf("query %q level %v: hits not score-ordered at %d", qs, level, i)
				}
			}
			for _, h := range deg.Hits {
				if int(h.DocID) >= res.Docs {
					t.Fatalf("query %q level %v: hit %d outside the %d docs", qs, level, h.DocID, res.Docs)
				}
			}
			switch level {
			case search.DegradeNone:
				// The no-degradation rung is the plain completeness path exactly.
				if len(deg.Hits) != len(plain.Hits) {
					t.Fatalf("query %q: DegradeNone returned %d hits, SearchComplete %d", qs, len(deg.Hits), len(plain.Hits))
				}
				for i := range deg.Hits {
					if deg.Hits[i] != plain.Hits[i] {
						t.Fatalf("query %q rank %d: DegradeNone %+v, SearchComplete %+v", qs, i, deg.Hits[i], plain.Hits[i])
					}
				}
			case search.DegradeShards, search.DegradeL2:
				// The shard-dropping rung (and the L2 rung above it) never query more shards
				// than the routed set, and strictly fewer once the routed set is large enough
				// that a quarter rounds to at least one shard.
				if deg.ShardsTotal > plain.ShardsTotal {
					t.Fatalf("query %q level %v: queried %d shards, want no more than the routed %d", qs, level, deg.ShardsTotal, plain.ShardsTotal)
				}
				if plain.ShardsTotal >= 4 && deg.ShardsTotal >= plain.ShardsTotal {
					t.Fatalf("query %q level %v: queried %d of %d routed shards, want fewer", qs, level, deg.ShardsTotal, plain.ShardsTotal)
				}
			}
			hitsSeen += len(deg.Hits)
		}
	}
	if hitsSeen == 0 {
		t.Fatalf("no query returned a hit at any degradation level over %d real docs", res.Docs)
	}
}
