package cli

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
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
	shards, _, all, err := openShards(out, model)
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

// TestAggregatorCCrawlDocFreqsExact is the cross-broker exact-idf proof on real data: an
// aggregator over two brokers that split the real shards gathers the same fleet-wide df and
// document count a single broker over every shard reports, so the idf the aggregator pushes
// down is the one a monolith would compute. The real corpus is exactly where a broker's own
// df diverges from the fleet's, because a term's documents are spread unevenly across the
// shards two brokers hold, so this is the gap TestAggregatorCCrawl could not assert
// bit-exactness against: with the fleet df gathered at the aggregator the statistic is now
// identical to the monolith's, term for term, over real text.
func TestAggregatorCCrawlDocFreqsExact(t *testing.T) {
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

	// all is the broker over every shard, the monolith ground truth for the fleet statistics;
	// it owns the file mappings, so it is the one closed. The aggregator's two sub-brokers
	// reference the same read-only shards split in half.
	shards, _, all, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = all.Close() }()
	pl := buildPipeline(shards)

	half := len(shards) / 2
	b0 := search.NewBroker(shards[:half], newCascade(model))
	b1 := search.NewBroker(shards[half:], newCascade(model))
	agg := search.NewAggregator([]search.Searcher{b0, b1})

	ctx := context.Background()
	if agg.NumDocs() != all.NumDocs() {
		t.Fatalf("aggregator NumDocs = %d, monolith %d", agg.NumDocs(), all.NumDocs())
	}

	// Gather the real retrieval terms the analysis chain produces for a set of common content
	// words, the terms the aggregator would push idf down for, and check the fleet df matches
	// the monolith's for every one of them.
	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	checked, skewed := 0, 0
	for _, qs := range queries {
		pq := pl.parse(qs)
		if pq.Empty() {
			continue
		}
		terms := toQuery(pq, 10).Terms
		if len(terms) == 0 {
			continue
		}
		want := all.DocFreqs(ctx, terms)
		got := agg.DocFreqs(ctx, terms)
		for _, term := range terms {
			if got[term] != want[term] {
				t.Fatalf("query %q fleet df[%q] = %d, monolith %d", qs, term, got[term], want[term])
			}
			if want[term] == 0 {
				continue
			}
			checked++
			// A term whose documents do not all sit in one broker's shards has a broker-local df
			// below the fleet df, the skew the gather corrects; count how many real terms show it
			// so the test fails loudly if the corpus happened to land every term in one broker.
			d0 := b0.DocFreqs(ctx, terms)
			d1 := b1.DocFreqs(ctx, terms)
			if d0[term] != want[term] && d1[term] != want[term] {
				skewed++
			}
		}
	}
	if checked == 0 {
		t.Fatalf("no real term resolved to a non-zero df over %d docs", res.Docs)
	}
	if skewed == 0 {
		t.Fatalf("no term's df was split across both brokers; the corpus shows no cross-broker skew to correct")
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

	shards, _, b, err := openShards(out, model)
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

// TestCacheCCrawl exercises the result cache on real data: it builds a real multi-shard
// collection, wires a cache into the broker, and checks that a repeated query is served
// from cache returning the same ranked top-k the cold query computed, that the cache
// reports a hit only the second time, and that the cache holds the warmed queries. This
// is the proof the cache keys and serves correctly over the real corpus, where the query
// understanding (analysis, expansion) the key normalizes on is the real chain, not a
// hand-built term set.
func TestCacheCCrawl(t *testing.T) {
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

	shards, _, b, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = b.Close() }()
	b.SetResultCache(search.NewResultCache(64))
	pl := buildPipeline(shards)

	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	warmed := 0
	for _, qs := range queries {
		pq := pl.parse(qs)
		if pq.Empty() {
			continue
		}
		q := toQuery(pq, 10)

		cold, hit := b.SearchCached(context.Background(), q)
		if hit {
			t.Fatalf("query %q: first run reported a cache hit", qs)
		}
		if !cold.Complete() {
			t.Fatalf("query %q: cold run flagged partial, %d of %d shards", qs, cold.ShardsOK, cold.ShardsTotal)
		}
		for _, h := range cold.Hits {
			if int(h.DocID) >= res.Docs {
				t.Fatalf("query %q: hit %d outside the %d docs", qs, h.DocID, res.Docs)
			}
		}

		warm, hit := b.SearchCached(context.Background(), q)
		if !hit {
			t.Fatalf("query %q: repeated run did not hit the cache", qs)
		}
		if len(warm.Hits) != len(cold.Hits) {
			t.Fatalf("query %q: warm returned %d hits, cold %d", qs, len(warm.Hits), len(cold.Hits))
		}
		for i := range cold.Hits {
			if warm.Hits[i] != cold.Hits[i] {
				t.Fatalf("query %q rank %d: warm %+v, cold %+v", qs, i, warm.Hits[i], cold.Hits[i])
			}
		}
		if warm.ShardsTotal != cold.ShardsTotal {
			t.Fatalf("query %q: warm ShardsTotal %d, cold %d", qs, warm.ShardsTotal, cold.ShardsTotal)
		}
		warmed++
	}
	if warmed == 0 {
		t.Fatalf("no query warmed the cache over %d real docs", res.Docs)
	}
	if b.ResultCache().Len() != warmed {
		t.Fatalf("cache holds %d entries, want the %d warmed queries", b.ResultCache().Len(), warmed)
	}

	// A publish-style invalidation clears the warmed entries, so the next query is cold again.
	b.ResultCache().Clear()
	if b.ResultCache().Len() != 0 {
		t.Fatalf("cache not empty after Clear: %d entries", b.ResultCache().Len())
	}
}

// TestRoutingCCrawlEquivalence is the routing-index correctness gate on real data: it builds
// a real multi-shard collection, builds the front-coded, bloom-fronted routing index over the
// open shards, and checks that the index routes every real query term to exactly the shards
// that actually hold the term, the brute-force membership the index replaced a whole-vocabulary
// Go map with. The brute-force oracle is each shard's own lexical df: a term routes to a shard
// if and only if that shard reports a non-zero df for it. This proves the front-coded lookup is
// exact over the real fleet vocabulary, the property that lets the resident routing structure
// shrink without changing a single routing decision. A term absent from the whole fleet must
// route to no shard, the bloom-reject path over real terms.
func TestRoutingCCrawlEquivalence(t *testing.T) {
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
		t.Fatalf("need at least 4 shards to see routing prune across the fleet, got %d", res.Shards)
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

	shards, _, b, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = b.Close() }()
	pl := buildPipeline(shards)

	ri := search.BuildRoutingIndex(shards)

	// bruteRoute is the oracle: the shards whose own lexical df for the term is non-zero, the
	// shards the term genuinely lives in, sorted ascending so it compares against the index's
	// routed set directly.
	bruteRoute := func(term string) []int {
		var want []int
		for si, s := range shards {
			if s.LexDocFreqs([]string{term})[term] > 0 {
				want = append(want, si)
			}
		}
		return want
	}

	// Sample real fleet terms straight from the shards' own vocabulary, so the check covers the
	// rare long-tail terms that live on one or two shards (where routing actually prunes), not
	// just the common content words that saturate every shard. A bounded stride keeps the test
	// fast while still spanning the whole vocabulary, and the common query terms are folded in so
	// the saturating case is exercised too.
	sample := map[string]struct{}{}
	for _, qs := range []string{"data", "page", "home", "search", "news", "world", "time", "people"} {
		pq := pl.parse(qs)
		if pq.Empty() {
			continue
		}
		for _, term := range toQuery(pq, 10).Terms {
			sample[term] = struct{}{}
		}
	}
	for _, s := range shards {
		i := 0
		s.ForEachTerm(func(term string, _ uint32) {
			if i%97 == 0 { // a coarse stride across each shard's term ids
				sample[term] = struct{}{}
			}
			i++
		})
	}

	checkedTerms, prunedTerms := 0, 0
	for term := range sample {
		want := bruteRoute(term)
		if len(want) == 0 {
			continue // not a real fleet term; nothing to compare
		}
		got := ri.RouteTerms([]string{term})
		sort.Ints(got)
		if len(got) != len(want) {
			t.Fatalf("term %q routed to %v, brute-force shard membership %v", term, got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("term %q routed to %v, brute-force shard membership %v", term, got, want)
			}
		}
		checkedTerms++
		if len(want) < res.Shards {
			prunedTerms++ // the term does not live on every shard, so routing actually prunes
		}
	}
	if checkedTerms == 0 {
		t.Fatalf("no real query term resolved to a fleet term over %d docs", res.Docs)
	}
	if prunedTerms == 0 {
		t.Fatalf("every checked term lived on every one of %d shards; routing pruned nothing to verify", res.Shards)
	}

	// A term that is not in the fleet vocabulary at all must route to no shard, the bloom-reject
	// path exercised against a real-shaped token rather than a synthetic one.
	if got := ri.RouteTerms([]string{"zzqxnot_a_real_fleet_term_zzqx"}); len(got) != 0 {
		t.Fatalf("absent term routed to %v, want no shard", got)
	}
	t.Logf("routing equivalence: %d real terms matched brute-force membership, %d of them pruned shards over %d shards",
		checkedTerms, prunedTerms, res.Shards)
}

// TestServeAdmissionCCrawl drives the HTTP serve path over a real multi-shard collection
// with a bounded admission gate under concurrent load, the property admission control
// exists for: the broker stays at its true capacity and sheds the overflow rather than
// collapsing. It fires many concurrent real queries at a small gate and checks every
// response is either a served 200 or a fast 503, that at least one of each is seen so both
// arms are exercised, that the in-flight count returns to zero once the burst drains (no
// leaked slot, the tatami over-admission bug the held-for-the-whole-search discipline
// fixes), and that the served responses carry real hits.
func TestServeAdmissionCCrawl(t *testing.T) {
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

	const capacity = 4
	adm := search.NewAdmission(capacity)
	srv := &httpServer{broker: broker, pipeline: pl, timeout: 0, admission: adm}
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	defer ts.Close()

	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people"}
	const workers = 64
	var ok, busy, withHits int64
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			q := queries[i%len(queries)]
			resp, err := http.Get(ts.URL + "?q=" + q + "&k=10")
			if err != nil {
				return
			}
			defer func() { _ = resp.Body.Close() }()
			switch resp.StatusCode {
			case http.StatusOK:
				atomic.AddInt64(&ok, 1)
				var got searchResponse
				if json.NewDecoder(resp.Body).Decode(&got) == nil && len(got.Hits) > 0 {
					atomic.AddInt64(&withHits, 1)
				}
			case http.StatusServiceUnavailable:
				atomic.AddInt64(&busy, 1)
			default:
				t.Errorf("unexpected status %d", resp.StatusCode)
			}
		}(i)
	}
	wg.Wait()

	if ok == 0 {
		t.Fatal("no request was admitted under load")
	}
	if busy == 0 {
		t.Fatalf("no request was shed: %d workers against a capacity-%d gate should overflow", workers, capacity)
	}
	if ok+busy != workers {
		t.Fatalf("admitted %d + shed %d != %d workers", ok, busy, workers)
	}
	if withHits == 0 {
		t.Fatal("no admitted request returned hits over the real corpus")
	}
	// The held-for-the-whole-search discipline means every served request released its slot:
	// after the burst the broker is back to zero in-flight, no leaked slot.
	if got := adm.InFlight(); got != 0 {
		t.Fatalf("in-flight after the burst = %d, want 0 (leaked slot)", got)
	}
	t.Logf("admission under load: %d served, %d shed, capacity %d, %d workers", ok, busy, capacity, workers)
}
