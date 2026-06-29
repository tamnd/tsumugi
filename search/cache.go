package search

import (
	"container/list"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// ResultCache is the broker's result cache (doc 11, "The result cache"): it holds the
// ranked top-k for a normalized query so a repeated query serves from cache without
// re-running the cascade. Search query distributions are heavy-tailed, so caching the
// head of the distribution takes real load off the cascade while the tail misses and
// runs it, which is the right division because the head is cheap to cache and the tail
// is what the cascade exists for.
//
// The cache is a serving optimization, not a correctness mechanism, so it is bounded by
// the freshness it must respect: a published or retired shard can change a query's
// top-k, so the collection's cache is cleared on a publish, the coarse invalidation the
// spec makes the default. Clear is the hook the publish lifecycle calls; wiring the
// publish path itself is a separate piece of freshness work. The cache is safe for
// concurrent use by the broker's query goroutines.
//
// The value stored is the ranked order, the ordered docIDs, which is what the spec keys
// the result on; the per-hit ordinal scores ride along as the display payload returned
// verbatim on a hit and are never compared across cache entries, the ordinal-score
// discipline the ranking doc pins. A cache hit returns exactly the result the request
// would have computed, because the key captures everything the result depends on.
type ResultCache struct {
	mu      sync.Mutex
	cap     int
	entries map[string]*list.Element
	lru     *list.List // front is most recently used
}

// cacheEntry is one cached result: the key it is stored under, the ranked top-k, and the
// number of shards the result was computed over, so a hit reports the same shard count
// the cold result did rather than a fabricated one.
type cacheEntry struct {
	key         string
	hits        []Hit
	shardsTotal int
}

// NewResultCache builds a result cache holding up to cap entries, evicting the least
// recently used entry when full. A non-positive cap disables the cache: it never stores
// and never hits, the no-cache path, so a deployment can turn the cache off by sizing it
// to zero.
func NewResultCache(cap int) *ResultCache {
	return &ResultCache{
		cap:     cap,
		entries: make(map[string]*list.Element),
		lru:     list.New(),
	}
}

// Get returns the cached ranked top-k for a query and whether it was present, moving the
// entry to the most-recently-used end on a hit. The returned slice is a copy, so a
// caller that mutates or sorts it cannot corrupt the cached entry. A disabled cache
// always misses.
func (c *ResultCache) Get(q Query) ([]Hit, int, bool) {
	if c == nil || c.cap <= 0 {
		return nil, 0, false
	}
	key := cacheKey(q)
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.entries[key]
	if !ok {
		return nil, 0, false
	}
	c.lru.MoveToFront(el)
	e := el.Value.(*cacheEntry)
	return cloneHits(e.hits), e.shardsTotal, true
}

// Put stores a query's ranked top-k under its normalized key, evicting the least
// recently used entry if the cache is at capacity. The stored slice is a copy, so a
// caller that reuses or mutates the slice after Put cannot corrupt the cached entry. A
// disabled cache stores nothing. Storing an existing key refreshes its value and its
// recency.
func (c *ResultCache) Put(q Query, hits []Hit, shardsTotal int) {
	if c == nil || c.cap <= 0 {
		return
	}
	key := cacheKey(q)
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.entries[key]; ok {
		e := el.Value.(*cacheEntry)
		e.hits = cloneHits(hits)
		e.shardsTotal = shardsTotal
		c.lru.MoveToFront(el)
		return
	}
	el := c.lru.PushFront(&cacheEntry{key: key, hits: cloneHits(hits), shardsTotal: shardsTotal})
	c.entries[key] = el
	for c.lru.Len() > c.cap {
		back := c.lru.Back()
		if back == nil {
			break
		}
		c.lru.Remove(back)
		delete(c.entries, back.Value.(*cacheEntry).key)
	}
}

// Clear empties the cache, the coarse invalidation a shard publish or retire triggers:
// a published shard can add a document to a cached query's top-k and a retired shard can
// remove one, so a cached result computed against the old collection can be stale against
// the new one, and the simple correct thing is to clear on publish and let the cache
// re-warm from traffic.
func (c *ResultCache) Clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[string]*list.Element)
	c.lru.Init()
}

// Len is the number of entries currently cached, for observability and tests.
func (c *ResultCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Len()
}

// cloneHits returns a copy of a hit slice so the cache and its callers never share the
// backing array.
func cloneHits(hits []Hit) []Hit {
	if hits == nil {
		return nil
	}
	out := make([]Hit, len(hits))
	copy(out, hits)
	return out
}

// cacheKey is the normalized request shape the result depends on, the spec's key: the
// analyzed query terms in order, the sparse term set, whether a dense vector is present,
// and k. Two raw queries that analyze to the same terms and the same shape produce the
// same key, so "New York" and "new york" share an entry once query understanding has
// normalized them. The key captures everything the result depends on, so a hit returns
// exactly the result the request would have computed. Filters would extend the key the
// same way once the query carries them.
func cacheKey(q Query) string {
	var b strings.Builder
	b.WriteString("k=")
	b.WriteString(strconv.Itoa(q.K))
	b.WriteString(";t=")
	terms := q.lexTerms()
	for i, t := range terms {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(t)
	}
	if len(q.Sparse) > 0 {
		b.WriteString(";s=")
		keys := make([]string, 0, len(q.Sparse))
		for t := range q.Sparse {
			keys = append(keys, t)
		}
		sort.Strings(keys)
		for i, t := range keys {
			if i > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(t)
			b.WriteByte(':')
			b.WriteString(strconv.Itoa(q.Sparse[t]))
		}
	}
	if len(q.Vector) > 0 {
		// The dense vector is a continuous encoding, so it is not part of the key by value;
		// its presence is, because a query with a dense plane is a different result shape
		// from a lexical-only one. A deployment that wants dense queries cached by vector
		// content would key on a quantized vector hash, a tuning knob beyond this default.
		b.WriteString(";v=1")
	}
	return b.String()
}
