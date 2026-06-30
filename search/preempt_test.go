package search

import (
	"context"
	"testing"
)

// tripContext is a context whose Err trips to cancelled after a fixed number of Err calls,
// so a test can pin exactly where in a retrieval the deadline appears to pass. Its Done
// channel never fires (it embeds Background), so retrieve, which polls Err and never selects
// on Done, abandons precisely at the boundary the call count lands on and nowhere else. This
// is how the tests below place the deadline between two retrieval planes deterministically.
type tripContext struct {
	context.Context
	calls  *int
	tripAt int
}

func (c tripContext) Err() error {
	*c.calls++
	if *c.calls > c.tripAt {
		return context.Canceled
	}
	return c.Context.Err()
}

// TestRetrieveCompletesUnderLiveContext is the no-regression check: with a live context the
// retrieval runs every plane to the end and reports completed, returning the same candidate
// count it did before the deadline check existed (every doc matches the shared term).
func TestRetrieveCompletesUnderLiveContext(t *testing.T) {
	const n = 80
	docs := makeCorpus(n)
	dir := t.TempDir()
	path := dir + "/shard.tsumugi"
	buildShardFile(t, path, docs, 0, n, 0, false)
	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	lex, _, _, completed := s.retrieve(context.Background(), Query{Text: "common document", K: 5})
	if !completed {
		t.Fatalf("retrieve under a live context reported not completed")
	}
	if len(lex) != n {
		t.Fatalf("retrieve returned %d candidates, want %d (every doc matches)", len(lex), n)
	}
}

// TestRetrieveAbandonsBeforeAnyPlane checks the entry guard: a context already cancelled when
// the retrieval starts makes it abandon before it touches a single plane, so a shard the
// broker dispatched just as the budget expired does no scanning at all.
func TestRetrieveAbandonsBeforeAnyPlane(t *testing.T) {
	const n = 80
	docs := makeCorpus(n)
	dir := t.TempDir()
	path := dir + "/shard.tsumugi"
	buildShardFile(t, path, docs, 0, n, 0, false)
	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	lex, dense, _, completed := s.retrieve(ctx, Query{Text: "common document", K: 5})
	if completed {
		t.Fatalf("retrieve under a cancelled context reported completed")
	}
	if len(lex) != 0 || len(dense) != 0 {
		t.Fatalf("retrieve abandoned before any plane should return nothing, got lex=%d dense=%d", len(lex), len(dense))
	}
}

// TestRetrieveSkipsDensePlaneAfterDeadline is the headline: the dense plane is the most
// expensive, so when the deadline passes during the lexical plane the retrieval must not run
// the ANN search at all. The trip context lets the entry check and the lexical plane pass,
// then trips at the lexical-to-dense boundary, and the test confirms the lexical candidates
// are there while the dense plane produced nothing and the retrieval is reported abandoned.
func TestRetrieveSkipsDensePlaneAfterDeadline(t *testing.T) {
	const n = 80
	docs := makeCorpus(n)
	dir := t.TempDir()
	path := dir + "/shard.tsumugi"
	buildShardFile(t, path, docs, 0, n, 0, true) // with a vector region, so the dense plane exists
	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Err calls for a lexical+dense query on an n<stride shard: entry (1), lexical loop i=0
	// (2), the before-sparse boundary (3), the before-dense boundary (4). Trip on the 4th so
	// the lexical plane completes and the dense plane is abandoned at its boundary.
	calls := 0
	ctx := tripContext{Context: context.Background(), calls: &calls, tripAt: 3}
	q := Query{Text: "common document", K: 5, Vector: docs[0].vec}
	lex, dense, _, completed := s.retrieve(ctx, q)
	if completed {
		t.Fatalf("retrieve past the deadline reported completed")
	}
	if len(lex) != n {
		t.Fatalf("lexical plane ran before the deadline, want %d candidates, got %d", n, len(lex))
	}
	if len(dense) != 0 {
		t.Fatalf("dense plane should have been skipped after the deadline, got %d candidates", len(dense))
	}
}

// TestRetrievePreemptsWithinLexicalPlane checks the stride poll inside a plane, not just at
// the boundaries: on a shard with more candidates than the stride, a deadline that passes
// after the first stride poll abandons the plane mid-scan, returning exactly the rows gathered
// up to the second poll rather than scanning the whole plane.
func TestRetrievePreemptsWithinLexicalPlane(t *testing.T) {
	const n = retrievePreemptStride*2 + 50 // more than two strides of candidates
	docs := makeCorpus(n)
	dir := t.TempDir()
	path := dir + "/shard.tsumugi"
	buildShardFile(t, path, docs, 0, n, 0, false)
	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Err calls: entry (1), lexical loop i=0 (2), lexical loop i=stride+1 (3). Trip on the 3rd
	// so the scan stops at the second poll, having gathered exactly the first stride+1 rows.
	calls := 0
	ctx := tripContext{Context: context.Background(), calls: &calls, tripAt: 2}
	lex, _, _, completed := s.retrieve(ctx, Query{Text: "common document", K: n})
	if completed {
		t.Fatalf("retrieve preempted mid-plane reported completed")
	}
	if want := retrievePreemptStride + 1; len(lex) != want {
		t.Fatalf("mid-plane preemption gathered %d candidates, want %d (one stride)", len(lex), want)
	}
}

// TestSearchDropsPreemptedShards is the broker-level integration: with an already-expired
// deadline every shard the broker dispatches abandons at its entry check and sends nothing,
// so the collection returns an honest partial (no shard counted, no hits) rather than serving
// a half-built answer as complete or hanging on shards that produced nothing. It exercises
// the goroutine's completed gate and the not-responded roll-up together, deterministically:
// whichever shards the dispatch loop launches before it sees the dead deadline, none of them
// contribute, so the outcome is the same regardless of the launch race.
func TestSearchDropsPreemptedShards(t *testing.T) {
	const n, parts = 160, 4
	shards, model := buildShardSet(t, n, parts)
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res := b.SearchComplete(ctx, Query{Text: "common document", K: 10})
	if res.Complete() {
		t.Fatalf("query under an expired deadline reported complete")
	}
	if res.ShardsOK != 0 {
		t.Fatalf("preempted shards counted as responded: ShardsOK=%d, want 0", res.ShardsOK)
	}
	if res.ShardsTotal != parts {
		t.Fatalf("ShardsTotal=%d, want %d", res.ShardsTotal, parts)
	}
	if len(res.Hits) != 0 {
		t.Fatalf("preempted query returned %d hits, want 0", len(res.Hits))
	}
}
