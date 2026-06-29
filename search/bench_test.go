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

// prodCascade is the benchmark cascade at the canon production cut sizes, so the
// latency measured is the latency the gate cares about, not an inflated wide cut.
func prodCascade(model *rank.Model) *rank.Cascade {
	return rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
}
