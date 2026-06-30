package search

import (
	"context"
	"testing"
)

// TestRetireReclaimsMappingWhenIdle is the core reclaim proof: with no query in flight, a
// retire releases the retired shard's mapping immediately rather than holding it until Close.
// The broker's open-mapping count drops by one the moment the retire swaps the shard out, and
// the survivors keep serving, so a fleet that churns shards does not accumulate retired
// mappings for the life of the process.
func TestRetireReclaimsMappingWhenIdle(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	if got := b.mappedShards(); got != parts {
		t.Fatalf("mappedShards at start = %d, want %d", got, parts)
	}

	victim := shards[1]
	lo, hi := idRange(victim)
	if got := b.Retire(func(s *Shard) bool { return s == victim }); got != 1 {
		t.Fatalf("Retire removed %d shards, want 1", got)
	}

	// No query was in flight, so the snapshot that held the victim drained the instant the
	// retire dropped its owner reference, and the reclaim unmapped the victim right away.
	if got := b.mappedShards(); got != parts-1 {
		t.Fatalf("mappedShards after idle retire = %d, want %d (retired mapping not reclaimed)", got, parts-1)
	}

	res := b.SearchComplete(context.Background(), Query{Text: "common document", K: 200})
	if !res.Complete() {
		t.Fatalf("query after reclaim was incomplete")
	}
	if hitsTouchRange(res.Hits, lo, hi) {
		t.Fatalf("retired shard's docs [%d,%d) still appeared after reclaim", lo, hi)
	}
}

// TestReclaimWaitsForInflightQuery is the safety half: a retire must not unmap a shard a
// query loaded before the retire, or that query would read freed memory. Holding a reference
// on the pre-retire snapshot (the stand-in for an in-flight query) keeps the retired shard
// mapped across the retire; only when that reference is released does the reclaim run and the
// mapping fall away. This is the deterministic version of what the concurrent churn test
// exercises under the race detector.
func TestReclaimWaitsForInflightQuery(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	// Take a reference on the current snapshot, as a query entering SearchDegraded would, and
	// hold it across the retire.
	held := b.acquire()

	victim := shards[2]
	if got := b.Retire(func(s *Shard) bool { return s == victim }); got != 1 {
		t.Fatalf("Retire removed %d shards, want 1", got)
	}

	// The retire swapped the victim out of the served set, but the held snapshot still
	// references it, so its mapping is kept open: the in-flight reader can finish safely.
	if got := b.NumShards(); got != parts-1 {
		t.Fatalf("NumShards after retire = %d, want %d", got, parts-1)
	}
	if got := b.mappedShards(); got != parts {
		t.Fatalf("mappedShards while a query holds the old snapshot = %d, want %d (reclaimed too early)", got, parts)
	}

	// The reader finishes: now nothing references the old snapshot, so the reclaim runs and
	// the victim's mapping is released.
	b.release(held)
	if got := b.mappedShards(); got != parts-1 {
		t.Fatalf("mappedShards after the in-flight query released = %d, want %d", got, parts-1)
	}
}

// TestCloseReleasesRetainedMapping checks the Close backstop: if a snapshot is still
// referenced at shutdown (a query that never finished, or a test holding a reference), Close
// still releases every mapping the broker holds open, both the current set and any retired
// shard a draining snapshot kept mapped, each exactly once.
func TestCloseReleasesRetainedMapping(t *testing.T) {
	const n, parts = 120, 3
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards, newTestCascade(model))

	// Retire one shard while a reference on the pre-retire snapshot keeps it mapped, so at
	// Close the broker holds both the two survivors and the one retired-but-still-mapped shard.
	held := b.acquire()
	if got := b.Retire(func(s *Shard) bool { return s == shards[0] }); got != 1 {
		t.Fatalf("Retire removed %d shards, want 1", got)
	}
	if got := b.mappedShards(); got != parts {
		t.Fatalf("mappedShards with retired shard held = %d, want %d", got, parts)
	}

	if err := b.Close(); err != nil {
		t.Fatalf("Close returned %v", err)
	}
	if got := b.mappedShards(); got != 0 {
		t.Fatalf("mappedShards after Close = %d, want 0", got)
	}
	// Releasing the still-held reference after Close must not double-close: the reclaim it
	// triggers finds the mappings already gone. Reaching here without a panic is the check.
	b.release(held)
}
