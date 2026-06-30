package search

import "context"

// CachedSearcher wraps any Searcher with a result cache so the head of the heavy-tailed
// query distribution serves from cache without re-running the work beneath it. It is the
// broker's SearchCached discipline (cache.go, broker.go) lifted off the concrete broker and
// onto the Searcher interface, so the cache composes over an aggregator, the head of a
// serving tree that spans machines, exactly as it does over one broker: a head node fans a
// query across remote brokers, merges their top-k, and a CachedSearcher in front of that
// aggregator takes the head of the distribution off the whole tree, not just off one leaf.
//
// The embedded Searcher supplies NumShards, NumDocs, DocFreqs, Stats, and SearchComplete, so
// a CachedSearcher is itself a Searcher and a parent aggregator fans across it like any other
// child; SearchCached is the one method it adds, the cache-aware entry point a server calls.
//
// Only a complete, non-degraded result is cached, the load-bearing rule the broker's cache
// already follows: a partial answer (a child dropped at the deadline somewhere beneath the
// tree) or a degraded one is transient and lower quality, so caching it would serve a stale
// degraded top-k to a later query that had the budget to do better. A hit serves that good
// result even under a tight current budget, which is the point. A nil cache makes SearchCached
// a plain SearchComplete with a false hit flag, so a caller uses it uniformly whether or not a
// cache is wired.
type CachedSearcher struct {
	Searcher
	cache *ResultCache
}

// NewCachedSearcher wraps a Searcher with a result cache. A nil or zero-capacity cache leaves
// every query running through the wrapped Searcher, so a deployment turns the head cache off by
// sizing it to zero the same way a broker does.
func NewCachedSearcher(s Searcher, cache *ResultCache) *CachedSearcher {
	return &CachedSearcher{Searcher: s, cache: cache}
}

// SearchCached serves a query through the result cache: a hit returns the cached ranked top-k
// without re-fanning the tree, and a miss runs the wrapped SearchComplete and caches the result
// when it is complete and non-degraded. The bool reports whether the query was a cache hit, so
// a server can surface the hit rate. A hit reports the shard count the cold result was computed
// over, so its metadata stays honest rather than fabricated.
func (c *CachedSearcher) SearchCached(ctx context.Context, q Query) (Results, bool) {
	if c.cache != nil {
		if hits, total, ok := c.cache.Get(q); ok {
			return Results{Hits: hits, ShardsTotal: total, ShardsOK: total, Degraded: DegradeNone}, true
		}
	}
	res := c.SearchComplete(ctx, q)
	if c.cache != nil && res.Complete() && res.Degraded == DegradeNone {
		c.cache.Put(q, res.Hits, res.ShardsTotal)
	}
	return res, false
}

// ResultCache returns the wrapped cache, or nil if none is set, so a publish or retire path can
// clear it on a collection change the same way it clears a broker's.
func (c *CachedSearcher) ResultCache() *ResultCache { return c.cache }
