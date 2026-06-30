package search

import "testing"

// BenchmarkPublishIncremental measures the cost of publishing one shard onto a base through
// the overlay path: build a front-coded dictionary over the new shard's own vocabulary,
// copy the small overlay slice, and fold the shard's statistics in. It does not rescan the
// base vocabulary, so its cost is proportional to one shard, not the fleet. Compare it to
// BenchmarkPublishFullRebuild over the same set to see the gap the incremental path opens.
func BenchmarkPublishIncremental(b *testing.B) {
	_, shards := benchBroker(b, 20000, 200)
	base := newState(shards[:len(shards)-1])
	spare := shards[len(shards)-1]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = base.withShard(spare)
	}
}

// BenchmarkPublishFullRebuild measures the cost of adding one shard by the full-rebuild
// path, the old publish: rescan every shard's vocabulary into a new routing index and resum
// every shard's statistics. This is what a publish cost before the overlay, and what the
// overlay path falls back to only once per overlayLimit publishes.
func BenchmarkPublishFullRebuild(b *testing.B) {
	_, shards := benchBroker(b, 20000, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = newState(shards)
	}
}
