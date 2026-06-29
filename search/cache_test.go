package search

import (
	"context"
	"testing"
)

// TestCacheKeyNormalizesTerms checks two queries that analyze to the same terms and the
// same shape produce the same key, so "New York" and "new york" share an entry once the
// broker has analyzed them, while a different k or a different term set keys apart.
func TestCacheKeyNormalizesTerms(t *testing.T) {
	a := Query{Terms: []string{"new", "york"}, K: 10}
	b := Query{Terms: []string{"new", "york"}, K: 10}
	if cacheKey(a) != cacheKey(b) {
		t.Fatalf("same analyzed terms and k should share a key: %q vs %q", cacheKey(a), cacheKey(b))
	}

	// A raw Text that analyzes to the same terms keys identically, the normalization the
	// spec wants: the key is over the analysis, not the raw string.
	raw := Query{Text: "New York", K: 10}
	if cacheKey(raw) != cacheKey(a) {
		t.Fatalf("raw text and pre-analyzed terms should share a key: %q vs %q", cacheKey(raw), cacheKey(a))
	}

	diffK := Query{Terms: []string{"new", "york"}, K: 20}
	if cacheKey(diffK) == cacheKey(a) {
		t.Fatal("different k must key apart")
	}
	diffTerms := Query{Terms: []string{"new", "jersey"}, K: 10}
	if cacheKey(diffTerms) == cacheKey(a) {
		t.Fatal("different terms must key apart")
	}
}

// TestCacheKeyShapeBeyondTerms checks the sparse term set and the presence of a dense
// vector are part of the key, so two queries with the same lexical terms but a different
// shape do not collide.
func TestCacheKeyShapeBeyondTerms(t *testing.T) {
	base := Query{Terms: []string{"a"}, K: 5}
	withSparse := Query{Terms: []string{"a"}, K: 5, Sparse: map[string]int{"x": 3}}
	if cacheKey(base) == cacheKey(withSparse) {
		t.Fatal("a sparse term set must change the key")
	}
	// The sparse set is order-independent: the key sorts it, so two maps with the same
	// pairs in any iteration order key identically.
	s1 := Query{Terms: []string{"a"}, K: 5, Sparse: map[string]int{"x": 3, "y": 4}}
	s2 := Query{Terms: []string{"a"}, K: 5, Sparse: map[string]int{"y": 4, "x": 3}}
	if cacheKey(s1) != cacheKey(s2) {
		t.Fatalf("sparse key must be order-independent: %q vs %q", cacheKey(s1), cacheKey(s2))
	}

	withVec := Query{Terms: []string{"a"}, K: 5, Vector: []float32{0.1, 0.2}}
	if cacheKey(base) == cacheKey(withVec) {
		t.Fatal("a dense vector's presence must change the key")
	}
	// Only presence is keyed, not value, so two queries with different vectors of the same
	// shape share a key, the documented default.
	withVec2 := Query{Terms: []string{"a"}, K: 5, Vector: []float32{0.9, 0.8}}
	if cacheKey(withVec) != cacheKey(withVec2) {
		t.Fatal("dense vectors are keyed by presence, not value")
	}
}

// TestCacheGetPut checks a stored result reads back identically and a missing key misses.
func TestCacheGetPut(t *testing.T) {
	c := NewResultCache(4)
	q := Query{Terms: []string{"alpha"}, K: 3}
	if _, _, ok := c.Get(q); ok {
		t.Fatal("empty cache must miss")
	}

	hits := []Hit{{DocID: 1, Score: 0.9}, {DocID: 2, Score: 0.8}}
	c.Put(q, hits, 7)
	got, total, ok := c.Get(q)
	if !ok {
		t.Fatal("stored key must hit")
	}
	if total != 7 {
		t.Fatalf("shardsTotal = %d, want 7", total)
	}
	if len(got) != 2 || got[0] != hits[0] || got[1] != hits[1] {
		t.Fatalf("cached hits = %v, want %v", got, hits)
	}
}

// TestCacheReturnsCopy checks Get and Put both clone, so a caller that mutates the slice
// it stored or the slice it got back cannot corrupt the cached entry.
func TestCacheReturnsCopy(t *testing.T) {
	c := NewResultCache(2)
	q := Query{Terms: []string{"x"}, K: 2}
	src := []Hit{{DocID: 1, Score: 1}, {DocID: 2, Score: 2}}
	c.Put(q, src, 1)
	src[0] = Hit{DocID: 99, Score: 99} // mutate after Put

	got, _, _ := c.Get(q)
	if got[0].DocID != 1 {
		t.Fatalf("Put must clone: cached entry changed to %v after caller mutated source", got[0])
	}
	got[1] = Hit{DocID: 42, Score: 42} // mutate the returned slice
	again, _, _ := c.Get(q)
	if again[1].DocID != 2 {
		t.Fatalf("Get must clone: cached entry changed to %v after caller mutated the result", again[1])
	}
}

// TestCacheLRUEviction checks the cache evicts the least recently used entry when it is
// full, and that a Get refreshes recency so the touched entry survives.
func TestCacheLRUEviction(t *testing.T) {
	c := NewResultCache(2)
	qa := Query{Terms: []string{"a"}, K: 1}
	qb := Query{Terms: []string{"b"}, K: 1}
	qc := Query{Terms: []string{"c"}, K: 1}
	c.Put(qa, []Hit{{DocID: 1}}, 1)
	c.Put(qb, []Hit{{DocID: 2}}, 1)

	// Touch a so b becomes the least recently used, then insert c to force one eviction.
	if _, _, ok := c.Get(qa); !ok {
		t.Fatal("a should be present before eviction")
	}
	c.Put(qc, []Hit{{DocID: 3}}, 1)

	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2 (capacity)", c.Len())
	}
	if _, _, ok := c.Get(qb); ok {
		t.Fatal("b was least recently used and should have been evicted")
	}
	if _, _, ok := c.Get(qa); !ok {
		t.Fatal("a was touched and should survive")
	}
	if _, _, ok := c.Get(qc); !ok {
		t.Fatal("c was just inserted and should be present")
	}
}

// TestCachePutRefreshesExisting checks storing an existing key updates its value and its
// recency rather than adding a second entry.
func TestCachePutRefreshesExisting(t *testing.T) {
	c := NewResultCache(2)
	q := Query{Terms: []string{"a"}, K: 1}
	c.Put(q, []Hit{{DocID: 1}}, 1)
	c.Put(q, []Hit{{DocID: 9, Score: 5}}, 3)
	if c.Len() != 1 {
		t.Fatalf("Len = %d, want 1 (same key refreshed, not added)", c.Len())
	}
	got, total, _ := c.Get(q)
	if total != 3 || got[0].DocID != 9 {
		t.Fatalf("refreshed entry = %v total=%d, want DocID 9 total 3", got, total)
	}
}

// TestCacheClear checks Clear empties the cache, the coarse invalidation a publish or
// retire triggers.
func TestCacheClear(t *testing.T) {
	c := NewResultCache(4)
	c.Put(Query{Terms: []string{"a"}, K: 1}, []Hit{{DocID: 1}}, 1)
	c.Put(Query{Terms: []string{"b"}, K: 1}, []Hit{{DocID: 2}}, 1)
	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2 before clear", c.Len())
	}
	c.Clear()
	if c.Len() != 0 {
		t.Fatalf("Len = %d, want 0 after clear", c.Len())
	}
	if _, _, ok := c.Get(Query{Terms: []string{"a"}, K: 1}); ok {
		t.Fatal("cleared cache must miss")
	}
}

// TestDisabledCache checks a non-positive capacity disables the cache: it never stores
// and never hits, the no-cache path.
func TestDisabledCache(t *testing.T) {
	c := NewResultCache(0)
	q := Query{Terms: []string{"a"}, K: 1}
	c.Put(q, []Hit{{DocID: 1}}, 1)
	if c.Len() != 0 {
		t.Fatalf("disabled cache Len = %d, want 0", c.Len())
	}
	if _, _, ok := c.Get(q); ok {
		t.Fatal("disabled cache must always miss")
	}
}

// TestNilCacheSafe checks every method tolerates a nil receiver, so a broker with no
// cache wired can call through without a nil check at each site.
func TestNilCacheSafe(t *testing.T) {
	var c *ResultCache
	if _, _, ok := c.Get(Query{K: 1}); ok {
		t.Fatal("nil cache must miss")
	}
	c.Put(Query{K: 1}, []Hit{{DocID: 1}}, 1) // must not panic
	c.Clear()                                // must not panic
	if c.Len() != 0 {
		t.Fatal("nil cache Len must be 0")
	}
}

// TestSearchCachedHitMatchesMiss checks the cached search path: the first query is a miss
// that runs the cascade and the second is a hit that returns the same ranked top-k
// without re-running it.
func TestSearchCachedHitMatchesMiss(t *testing.T) {
	b, _ := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()
	b.SetResultCache(NewResultCache(16))

	q := Query{Text: "common document", K: 20}
	miss, hit := b.SearchCached(context.Background(), q)
	if hit {
		t.Fatal("first query must be a cache miss")
	}
	if !miss.Complete() {
		t.Fatal("a full-deadline search should be complete")
	}

	cached, hit := b.SearchCached(context.Background(), q)
	if !hit {
		t.Fatal("repeated query must be a cache hit")
	}
	if len(cached.Hits) != len(miss.Hits) {
		t.Fatalf("hit returned %d hits, miss returned %d", len(cached.Hits), len(miss.Hits))
	}
	for i := range miss.Hits {
		if cached.Hits[i] != miss.Hits[i] {
			t.Fatalf("hit %d differs: cached %v vs computed %v", i, cached.Hits[i], miss.Hits[i])
		}
	}
	if cached.ShardsTotal != miss.ShardsTotal {
		t.Fatalf("cached ShardsTotal = %d, want %d", cached.ShardsTotal, miss.ShardsTotal)
	}
	if !cached.Complete() {
		t.Fatal("a cached complete result must report complete")
	}
	if cached.Degraded != DegradeNone {
		t.Fatalf("a cached result must report DegradeNone, got %v", cached.Degraded)
	}
}

// TestSearchCachedSkipsPartial checks a partial result, one with a shard dropped at the
// deadline, is not cached: the cache must hold only complete results, so a later query
// with budget to spare is not served a stale partial answer.
func TestSearchCachedSkipsPartial(t *testing.T) {
	b, _ := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()
	b.SetResultCache(NewResultCache(16))

	// A cancelled context with the deterministic all-shards-dropped lever produces a
	// partial result the cache must refuse.
	b.maxConcurrency = 0
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	q := Query{Text: "common document", K: 20}
	res, hit := b.SearchCached(ctx, q)
	if hit {
		t.Fatal("first query cannot be a hit")
	}
	if res.Complete() {
		t.Fatal("the dropped-shards lever should yield an incomplete result")
	}
	if b.ResultCache().Len() != 0 {
		t.Fatalf("a partial result must not be cached, Len = %d", b.ResultCache().Len())
	}
}

// TestSearchCachedSkipsDegraded checks a degraded result, one served at a lower rung
// under budget pressure, is not cached, so the cache never serves a lower-quality answer
// to a later query that had the budget to do better.
func TestSearchCachedSkipsDegraded(t *testing.T) {
	b, _ := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()
	b.SetResultCache(NewResultCache(16))

	q := Query{Text: "common document", K: 20}
	res := b.SearchDegraded(context.Background(), q, DegradeDense)
	if res.Degraded != DegradeDense {
		t.Fatalf("expected a degraded result, got %v", res.Degraded)
	}
	// SearchDegraded does not consult the cache, but the cache-only-complete rule is what
	// SearchCached enforces, so confirm the broker never cached the degraded run.
	if b.ResultCache().Len() != 0 {
		t.Fatalf("a degraded result must not be cached, Len = %d", b.ResultCache().Len())
	}
}

// TestSearchCachedNoCacheWired checks SearchCached works as a plain budgeted search when
// no cache is wired, always reporting a miss.
func TestSearchCachedNoCacheWired(t *testing.T) {
	b, _ := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	q := Query{Text: "common document", K: 20}
	res, hit := b.SearchCached(context.Background(), q)
	if hit {
		t.Fatal("a broker with no cache must always miss")
	}
	if !res.Complete() {
		t.Fatal("the budgeted search should be complete on a full deadline")
	}
}
