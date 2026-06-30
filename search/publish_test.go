package search

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	"github.com/tamnd/tsumugi/rank"
)

// buildShardSet builds parts shards over the shared test corpus, each holding a
// contiguous slice of the corpus at a node base equal to its first document's global id,
// and returns the shards with the model the broker reranks against. The shards are
// independent files, the unit a publish adds and a retire removes, so a test can build a
// broker over any subset and swap the rest in and out.
func buildShardSet(t *testing.T, n, parts int) ([]*Shard, *rank.Model) {
	shards, _, model := buildShardSetFiles(t, n, parts)
	return shards, model
}

// buildShardSetFiles is buildShardSet that also returns each shard's file path, so a test
// that churns the served set can reopen a retired shard's file as a fresh mapping the way a
// real reload does, rather than republishing a *Shard pointer the broker may have already
// reclaimed.
func buildShardSetFiles(t *testing.T, n, parts int) ([]*Shard, []string, *rank.Model) {
	t.Helper()
	docs := makeCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)
	size := n / parts
	shards := make([]*Shard, parts)
	paths := make([]string, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("shard%d.tsumugi", p))
		lo := p * size
		buildShardFile(t, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
		paths[p] = path
	}
	return shards, paths, model
}

// idRange is the half-open global id span a shard owns, used to check that a published
// shard's documents appear in results and a retired shard's documents do not.
func idRange(s *Shard) (lo, hi uint32) {
	return s.NodeBase(), s.NodeBase() + s.DocCount()
}

// hitsTouchRange reports whether any hit falls in the half-open id range, the test of
// whether a shard's documents are present in a result.
func hitsTouchRange(hits []Hit, lo, hi uint32) bool {
	for _, h := range hits {
		if h.DocID >= lo && h.DocID < hi {
			return true
		}
	}
	return false
}

// TestPublishAddsShard checks a publish grows the served set: the new shard's documents
// were absent before the publish and present after, the shard count and the fleet-wide
// document count both grow, and a query in flight against the old snapshot is unaffected
// because it ran against the set it loaded.
func TestPublishAddsShard(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)

	// Serve the first three shards; hold the fourth back to publish.
	b := NewBroker(shards[:parts-1], newTestCascade(model))
	defer func() { _ = b.Close() }()

	held := shards[parts-1]
	lo, hi := idRange(held)

	q := Query{Text: "common document", K: 200}
	before := b.SearchComplete(context.Background(), q)
	if hitsTouchRange(before.Hits, lo, hi) {
		t.Fatalf("held-back shard's docs [%d,%d) appeared before it was published", lo, hi)
	}
	if got := b.NumShards(); got != parts-1 {
		t.Fatalf("NumShards before publish = %d, want %d", got, parts-1)
	}
	docsBefore := b.NumDocs()

	b.Publish(held)

	if got := b.NumShards(); got != parts {
		t.Fatalf("NumShards after publish = %d, want %d", got, parts)
	}
	if got := b.NumDocs(); got != docsBefore+uint64(held.DocCount()) {
		t.Fatalf("NumDocs after publish = %d, want %d", got, docsBefore+uint64(held.DocCount()))
	}
	after := b.SearchComplete(context.Background(), q)
	if !after.Complete() {
		t.Fatalf("query after publish was incomplete")
	}
	if !hitsTouchRange(after.Hits, lo, hi) {
		t.Fatalf("published shard's docs [%d,%d) did not appear in results", lo, hi)
	}
	if after.ShardsTotal != parts {
		t.Fatalf("ShardsTotal after publish = %d, want %d", after.ShardsTotal, parts)
	}
}

// TestRetireRemovesShard checks a retire shrinks the served set: the retired shard's
// documents stop appearing, the shard and document counts drop, and the broker keeps
// serving the survivors. The retired shard's mapping is kept open until Close, so the
// retire does not fault an in-flight reader, which the post-retire query exercises.
func TestRetireRemovesShard(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)

	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	victim := shards[1]
	lo, hi := idRange(victim)

	q := Query{Text: "common document", K: 200}
	before := b.SearchComplete(context.Background(), q)
	if !hitsTouchRange(before.Hits, lo, hi) {
		t.Fatalf("victim shard's docs [%d,%d) absent before retire", lo, hi)
	}
	docsBefore := b.NumDocs()

	got := b.Retire(func(s *Shard) bool { return s.NodeBase() == victim.NodeBase() })
	if got != 1 {
		t.Fatalf("Retire removed %d shards, want 1", got)
	}
	if n := b.NumShards(); n != parts-1 {
		t.Fatalf("NumShards after retire = %d, want %d", n, parts-1)
	}
	if n := b.NumDocs(); n != docsBefore-uint64(victim.DocCount()) {
		t.Fatalf("NumDocs after retire = %d, want %d", n, docsBefore-uint64(victim.DocCount()))
	}
	after := b.SearchComplete(context.Background(), q)
	if !after.Complete() {
		t.Fatalf("query after retire was incomplete")
	}
	if hitsTouchRange(after.Hits, lo, hi) {
		t.Fatalf("retired shard's docs [%d,%d) still appeared in results", lo, hi)
	}
	if after.ShardsTotal != parts-1 {
		t.Fatalf("ShardsTotal after retire = %d, want %d", after.ShardsTotal, parts-1)
	}
}

// TestRetireNoMatch checks a retire whose predicate selects nothing is a no-op: it
// returns zero, leaves the served set unchanged, and does not clear the cache, so a
// sweep that finds nothing to retire costs nothing.
func TestRetireNoMatch(t *testing.T) {
	const n, parts = 80, 4
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	cache := NewResultCache(8)
	b.SetResultCache(cache)
	if _, hit := b.SearchCached(context.Background(), Query{Text: "common document", K: 10}); hit {
		t.Fatalf("first cached query was a hit")
	}
	if cache.Len() == 0 {
		t.Fatalf("cache did not warm")
	}

	if got := b.Retire(func(s *Shard) bool { return false }); got != 0 {
		t.Fatalf("Retire matching nothing returned %d, want 0", got)
	}
	if got := b.NumShards(); got != parts {
		t.Fatalf("NumShards after no-op retire = %d, want %d", got, parts)
	}
	if cache.Len() == 0 {
		t.Fatalf("no-op retire cleared the cache")
	}
}

// TestPublishClearsCache checks the publish lifecycle invalidates the result cache: a
// warmed entry is gone after a publish, so the next query recomputes against the new
// shard set rather than serving a top-k that predates the published shard.
func TestPublishClearsCache(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards[:parts-1], newTestCascade(model))
	defer func() { _ = b.Close() }()

	cache := NewResultCache(16)
	b.SetResultCache(cache)

	q := Query{Text: "common document", K: 10}
	if _, hit := b.SearchCached(context.Background(), q); hit {
		t.Fatalf("cold query reported a cache hit")
	}
	if _, hit := b.SearchCached(context.Background(), q); !hit {
		t.Fatalf("warm query was not a cache hit")
	}

	b.Publish(shards[parts-1])

	if cache.Len() != 0 {
		t.Fatalf("publish left %d cache entries, want 0", cache.Len())
	}
	if _, hit := b.SearchCached(context.Background(), q); hit {
		t.Fatalf("query after publish hit a stale cache entry")
	}
}

// TestRetireClearsCache mirrors the publish case: a retire invalidates the cache too,
// because a retired shard can drop a document out of a cached query's top-k.
func TestRetireClearsCache(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	cache := NewResultCache(16)
	b.SetResultCache(cache)

	q := Query{Text: "common document", K: 10}
	b.SearchCached(context.Background(), q)
	if _, hit := b.SearchCached(context.Background(), q); !hit {
		t.Fatalf("warm query was not a cache hit")
	}

	if got := b.Retire(func(s *Shard) bool { return s.NodeBase() == shards[0].NodeBase() }); got != 1 {
		t.Fatalf("Retire removed %d shards, want 1", got)
	}
	if cache.Len() != 0 {
		t.Fatalf("retire left %d cache entries, want 0", cache.Len())
	}
}

// TestConcurrentQueriesDuringSwap is the safety pin: many queries run while publishes and
// retires swap the served set, and every query returns a consistent, complete result over
// whatever set it loaded. Run under the race detector it checks the atomic snapshot swap
// has no data race between a query reading the state and a publish or retire storing a new
// one, and that a retired shard is reclaimed only after the last query that loaded it
// finishes, never read after free. The writer reopens the churn shards from their files
// each cycle, the way a real reload does, so a republish never reuses a mapping the reclaim
// may already have unmapped.
func TestConcurrentQueriesDuringSwap(t *testing.T) {
	const n, parts = 200, 5
	shards, paths, model := buildShardSetFiles(t, n, parts)

	// Serve a stable core of the first three shards; churn the last two in and out.
	core := parts - 2
	b := NewBroker(shards[:core], newTestCascade(model))
	defer func() { _ = b.Close() }()

	coreLo := shards[0].NodeBase()
	coreHi := shards[core-1].NodeBase() + shards[core-1].DocCount()
	churnBase := shards[core].NodeBase()

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Readers: keep querying and assert the core shards' documents are always present and
	// every result is internally complete, regardless of which churn shards are mounted.
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := Query{Text: "common document", K: 200}
			for {
				select {
				case <-stop:
					return
				default:
				}
				res := b.SearchComplete(context.Background(), q)
				if !res.Complete() {
					t.Errorf("concurrent query was incomplete: %d/%d shards", res.ShardsOK, res.ShardsTotal)
					return
				}
				if !hitsTouchRange(res.Hits, coreLo, coreHi) {
					t.Errorf("core shards' docs [%d,%d) missing under churn", coreLo, coreHi)
					return
				}
			}
		}()
	}

	// Writer: repeatedly open the two churn shards fresh, publish them, and retire them again,
	// so the served set is constantly swapping under the readers and each retire leaves a
	// mapping for the reclaim to release. Reopening each cycle mirrors a real reload: the old
	// mapping is reclaimed and the new one is a distinct file mapping.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			c0, err := OpenShard(paths[core], newTestCascade(model))
			if err != nil {
				t.Errorf("reopen churn shard 0: %v", err)
				break
			}
			c1, err := OpenShard(paths[core+1], newTestCascade(model))
			if err != nil {
				t.Errorf("reopen churn shard 1: %v", err)
				break
			}
			b.Publish(c0)
			b.Publish(c1)
			b.Retire(func(s *Shard) bool { return s.NodeBase() >= churnBase })
		}
		close(stop)
	}()

	wg.Wait()

	// The churn ends with both churn shards retired, so the steady state is the core set.
	if got := b.NumShards(); got != core {
		t.Fatalf("after churn NumShards = %d, want %d", got, core)
	}
	// Every churn shard opened during the run was retired, and with no query in flight the
	// reclaim has released all of them, so the broker holds exactly the core mappings open.
	if got := b.mappedShards(); got != core {
		t.Fatalf("after churn mappedShards = %d, want %d (retired mappings not reclaimed)", got, core)
	}
}
