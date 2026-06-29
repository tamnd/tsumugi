package search

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/rank"
)

// BenchmarkBrokerSearch measures broker query latency over a sharded collection, the
// path the ten-millisecond gate is taken on. It builds a mid-size collection across
// several shards and times a full route, fan-out, merge, and global rerank per query.
// The corpus is lexical and feature only, the broker hot path; the dense plane's HNSW
// build cost would dominate setup and is benchmarked on its own in the vector package.
func BenchmarkBrokerSearch(b *testing.B) {
	const n, parts = 50000, 16
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)

	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("s%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, prodCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	br := NewBroker(shards, prodCascade(model))
	defer func() { _ = br.Close() }()

	ctx := context.Background()
	q := Query{Text: "common document number", K: 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hits := br.Search(ctx, q)
		if len(hits) == 0 {
			b.Fatal("no hits")
		}
	}
}

// BenchmarkBrokerSearchComplete measures the completeness-carrying entry point over the
// same sharded collection, so the cost of returning the partial-results indicator can be
// read against the back-compatible Search above. The two share one fan-out, so this
// confirms the indicator rides along for free rather than adding a second pass over the
// shards.
func BenchmarkBrokerSearchComplete(b *testing.B) {
	const n, parts = 50000, 16
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)

	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("s%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, prodCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	br := NewBroker(shards, prodCascade(model))
	defer func() { _ = br.Close() }()

	ctx := context.Background()
	q := Query{Text: "common document number", K: 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := br.SearchComplete(ctx, q)
		if !res.Complete() || len(res.Hits) == 0 {
			b.Fatal("complete fan-out returned no hits")
		}
	}
}

// BenchmarkAggregatorSearch measures a two-tier query: an aggregator over four brokers,
// each over four shards, the same sixteen-shard corpus the broker benchmarks use, so the
// aggregator's merge overhead can be read against a single broker over all sixteen. The
// merge is a sort of the children's top-ks, cheap next to the fan-out, which is the
// property doc 11's projected aggregator p99 rests on: the tree fans out without the merge
// becoming the bottleneck.
func BenchmarkAggregatorSearch(b *testing.B) {
	const n, brokers, perBroker = 50000, 4, 4
	parts := brokers * perBroker
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)

	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("s%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, prodCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	children := make([]Searcher, brokers)
	for k := 0; k < brokers; k++ {
		children[k] = NewBroker(shards[k*perBroker:(k+1)*perBroker], prodCascade(model))
	}
	agg := NewAggregator(children)

	ctx := context.Background()
	q := Query{Text: "common document number", K: 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := agg.SearchComplete(ctx, q)
		if !res.Complete() || len(res.Hits) == 0 {
			b.Fatal("complete tree returned no hits")
		}
	}
}

// BenchmarkBrokerSearchDegraded measures each rung of the degradation ladder over the
// same sixteen-shard corpus the other broker benchmarks use, so the budget each rung
// recovers can be read against the full-quality path. The ladder is a budget tool, so
// the point is that a higher rung is cheaper: shrinking L0 and dropping shards cut the
// retrieval and fan-out work, which is what lets a query under pressure answer within
// budget. The shard built is lexical and feature only, so the dense-drop rung shows no
// gain here (there is no dense plane to drop); its win is measured where a vector region
// exists, and the rung is correctness-tested in the search suite.
func BenchmarkBrokerSearchDegraded(b *testing.B) {
	const n, parts = 50000, 16
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)

	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("s%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, prodCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	br := NewBroker(shards, prodCascade(model))
	defer func() { _ = br.Close() }()

	ctx := context.Background()
	q := Query{Text: "common document number", K: 10}
	levels := []struct {
		name  string
		level DegradeLevel
	}{
		{"none", DegradeNone},
		{"shrink-l0", DegradeL0},
		{"drop-dense", DegradeDense},
		{"drop-shards", DegradeShards},
		{"trim-l2", DegradeL2},
	}
	for _, lv := range levels {
		b.Run(lv.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				res := br.SearchDegraded(ctx, q, lv.level)
				if len(res.Hits) == 0 {
					b.Fatal("degraded fan-out returned no hits")
				}
			}
		})
	}
}

// BenchmarkBrokerSearchCached measures the cached search path against the cold cascade
// over the same sixteen-shard corpus, so the load the cache takes off the cascade can be
// read directly. The miss case runs the whole cascade and stores the result; the hit case
// is served from the cache, which is the head-of-distribution work the cache exists to
// absorb. The hit should be far cheaper than the miss, because it returns the stored
// ranked top-k without retrieving, fusing, or reranking anything.
func BenchmarkBrokerSearchCached(b *testing.B) {
	const n, parts = 50000, 16
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)

	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("s%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, prodCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	br := NewBroker(shards, prodCascade(model))
	defer func() { _ = br.Close() }()
	br.SetResultCache(NewResultCache(1024))

	ctx := context.Background()
	q := Query{Text: "common document number", K: 10}

	// The miss case never lets the cache warm, so every iteration pays the full cascade
	// and the store; clearing before each run keeps it a miss.
	b.Run("miss", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			br.ResultCache().Clear()
			res, _ := br.SearchCached(ctx, q)
			if len(res.Hits) == 0 {
				b.Fatal("cached miss returned no hits")
			}
		}
	})

	// The hit case warms once, then every iteration serves from cache.
	if _, _ = br.SearchCached(ctx, q); br.ResultCache().Len() == 0 {
		b.Fatal("cache failed to warm")
	}
	b.Run("hit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			res, hit := br.SearchCached(ctx, q)
			if !hit {
				b.Fatal("expected a cache hit")
			}
			if len(res.Hits) == 0 {
				b.Fatal("cached hit returned no hits")
			}
		}
	})
}

// prodCascade is the benchmark cascade at the canon production cut sizes, so the
// latency measured is the latency the gate cares about, not an inflated wide cut.
func prodCascade(model *rank.Model) *rank.Cascade {
	return rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
}
