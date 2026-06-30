package search

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
)

// benchAggregator builds an aggregator over groups brokers, each over partsPer shards, for the
// partitioned-stats benchmarks. It returns the aggregator and a flat list of the brokers so a
// benchmark can close them.
func benchAggregator(b *testing.B, n, groups, partsPer int) (*Aggregator, []*Broker) {
	b.Helper()
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)
	total := groups * partsPer
	size := n / total
	brokers := make([]*Broker, groups)
	children := make([]Searcher, groups)
	next := 0
	for g := 0; g < groups; g++ {
		shards := make([]*Shard, partsPer)
		for p := 0; p < partsPer; p++ {
			path := filepath.Join(dir, fmt.Sprintf("g%d_s%d.tsumugi", g, p))
			lo := next * size
			buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
			sh, err := OpenShard(path, newTestCascade(model))
			if err != nil {
				b.Fatalf("open shard g%d p%d: %v", g, p, err)
			}
			shards[p] = sh
			next++
		}
		br := NewBroker(shards, newTestCascade(model))
		brokers[g] = br
		children[g] = br
	}
	return NewAggregator(children), brokers
}

// BenchmarkAggregatorStatsFold measures the cost of folding the children's statistics into the
// deployment-wide field averages, the work the partitioned push-down adds to a query. It is a
// fold over the children, not a rescan of any shard, so it is constant in the corpus size and
// linear only in the child count, which is what keeps the push-down off the per-query budget.
func BenchmarkAggregatorStatsFold(b *testing.B) {
	agg, brokers := benchAggregator(b, 8000, 8, 10)
	defer func() {
		for _, br := range brokers {
			_ = br.Close()
		}
	}()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = agg.Stats()
	}
}

// BenchmarkAggregatorSearchPartitioned measures a full query through the aggregator over the
// partitioned brokers, the path that gathers idf, folds the field averages, pushes both down,
// fans out, and merges. It shows the end-to-end cost of the exact partitioned merge.
func BenchmarkAggregatorSearchPartitioned(b *testing.B) {
	agg, brokers := benchAggregator(b, 8000, 8, 10)
	defer func() {
		for _, br := range brokers {
			_ = br.Close()
		}
	}()
	q := Query{Text: "common document", K: 10}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = agg.SearchComplete(ctx, q)
	}
}
