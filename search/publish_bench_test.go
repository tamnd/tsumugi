package search

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/rank"
)

// benchBroker builds a broker over parts shards for the publish/retire benchmarks, returning
// the shards, their file paths, and the model so a benchmark that churns the served set can
// reopen a shard as a fresh mapping rather than republishing one the reclaim may have unmapped.
func benchBroker(b *testing.B, n, parts int) (*Broker, []*Shard, []string, *rank.Model) {
	b.Helper()
	docs := makeCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)
	size := n / parts
	shards := make([]*Shard, parts)
	paths := make([]string, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("shard%d.tsumugi", p))
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
		paths[p] = path
	}
	return NewBroker(shards, newTestCascade(model)), shards, paths, model
}

// BenchmarkPublishRetire measures one publish plus one retire cycle, the cost of a
// shard-set swap: rebuild the routing index and statistics over the new set, store the
// atomic pointer, clear the cache, and reclaim the retired mapping. It is the control-plane
// path, not the query path, so it is not budget-gated, but it shows the swap is cheap relative
// to how rarely it fires. Each cycle reopens the churned shard from its file, the way a real
// reload does, so the retire genuinely reclaims a mapping and the publish maps a fresh one.
func BenchmarkPublishRetire(b *testing.B) {
	br, shards, paths, model := benchBroker(b, 4000, 40)
	defer func() { _ = br.Close() }()
	last := shards[len(shards)-1]
	lastBase := last.NodeBase()
	lastPath := paths[len(paths)-1]
	// Start with the last shard retired so each iteration publishes a fresh copy then retires
	// it, netting back to the same served set.
	br.Retire(func(s *Shard) bool { return s.NodeBase() == lastBase })
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fresh, err := OpenShard(lastPath, newTestCascade(model))
		if err != nil {
			b.Fatalf("reopen churn shard: %v", err)
		}
		br.Publish(fresh)
		br.Retire(func(s *Shard) bool { return s.NodeBase() == lastBase })
	}
}

// BenchmarkSearchDuringNoSwap measures the query path's loadState overhead by running the
// same query the concurrent test runs with no swap in flight, so the cost reported is the
// atomic load plus the search, confirming the snapshot indirection adds nothing to the hot
// path.
func BenchmarkSearchDuringNoSwap(b *testing.B) {
	br, _, _, _ := benchBroker(b, 4000, 40)
	defer func() { _ = br.Close() }()
	q := Query{Text: "common document", K: 10}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = br.SearchComplete(ctx, q)
	}
}
