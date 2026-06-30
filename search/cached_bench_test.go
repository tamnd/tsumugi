package search

import (
	"context"
	"testing"
)

// BenchmarkCachedSearcherHit measures the head-cache hit path: a query that is already cached
// returns the ranked top-k without re-fanning the tree, the cost the head pays for the head of
// the heavy-tailed distribution. It is the lookup, the LRU bump, and the slice clone, no
// retrieval or rerank, so it sits far under a cold query's budget.
func BenchmarkCachedSearcherHit(b *testing.B) {
	const n, parts = 4000, 8
	docs := remoteRankCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)
	broker, shards := buildBrokerFromDocs(b, dir, "h", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	cs := NewCachedSearcher(broker, NewResultCache(64))
	ctx := context.Background()
	q := Query{Terms: []string{"common"}, K: 10}
	if _, hit := cs.SearchCached(ctx, q); hit {
		b.Fatalf("warm-up call was a hit on a cold cache")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, hit := cs.SearchCached(ctx, q)
		if !hit || len(res.Hits) == 0 {
			b.Fatalf("expected a warm hit with results, got hit=%v hits=%d", hit, len(res.Hits))
		}
	}
}
