package search

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
)

// benchBroker builds a broker over parts shards for the publish/retire benchmarks.
func benchBroker(b *testing.B, n, parts int) (*Broker, []*Shard) {
	b.Helper()
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("shard%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	return NewBroker(shards, newTestCascade(model)), shards
}

// BenchmarkPublishRetire measures one publish plus one retire cycle, the cost of a
// shard-set swap: rebuild the routing index and statistics over the new set, store the
// atomic pointer, and clear the cache. It is the control-plane path, not the query path,
// so it is not budget-gated, but it shows the swap is cheap relative to how rarely it
// fires.
func BenchmarkPublishRetire(b *testing.B) {
	br, shards := benchBroker(b, 4000, 40)
	defer func() { _ = br.Close() }()
	last := shards[len(shards)-1]
	// Start with the last shard retired so each iteration publishes then retires it,
	// netting back to the same served set.
	br.Retire(func(s *Shard) bool { return s.NodeBase() == last.NodeBase() })
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		br.Publish(last)
		br.Retire(func(s *Shard) bool { return s.NodeBase() == last.NodeBase() })
	}
}

// BenchmarkSearchDuringNoSwap measures the query path's loadState overhead by running the
// same query the concurrent test runs with no swap in flight, so the cost reported is the
// atomic load plus the search, confirming the snapshot indirection adds nothing to the hot
// path.
func BenchmarkSearchDuringNoSwap(b *testing.B) {
	br, _ := benchBroker(b, 4000, 40)
	defer func() { _ = br.Close() }()
	q := Query{Text: "common document", K: 10}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = br.SearchComplete(ctx, q)
	}
}
