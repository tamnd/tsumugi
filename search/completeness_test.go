package search

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
)

// TestResultsComplete checks the completeness predicate: a result is complete only
// when every contributing shard responded, the rule the serve path's flag rests on.
func TestResultsComplete(t *testing.T) {
	cases := []struct {
		total, ok int
		want      bool
	}{
		{0, 0, true},  // nothing to route is trivially complete
		{3, 3, true},  // every contributing shard responded
		{3, 2, false}, // one dropped
		{3, 0, false}, // all dropped
		{1, 1, true},
	}
	for _, c := range cases {
		got := Results{ShardsTotal: c.total, ShardsOK: c.ok}.Complete()
		if got != c.want {
			t.Fatalf("Complete(total=%d, ok=%d) = %v, want %v", c.total, c.ok, got, c.want)
		}
	}
}

// brokerForCompleteness builds a multi-shard broker over the shared test corpus and
// returns it with the count of shards a whole-corpus query routes to. Every document
// carries the words "common document", so a query on them is a contributing-shard
// query against every shard, which is what lets the completeness count be checked.
func brokerForCompleteness(t *testing.T) (*Broker, int) {
	t.Helper()
	const n, parts = 160, 4
	docs := makeCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, fmt.Sprintf("shard%d.tsumugi", p))
		lo := p * size
		buildShardFile(t, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	b := NewBroker(shards, newTestCascade(model))
	return b, parts
}

// TestSearchCompleteAllShardsRespond checks that with the full deadline every
// contributing shard responds, so the result is flagged complete and the count of
// shards reached equals the count routed, and the hits match the back-compatible
// Search that drops the indicator.
func TestSearchCompleteAllShardsRespond(t *testing.T) {
	b, parts := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	q := Query{Text: "common document", K: 20}
	res := b.SearchComplete(context.Background(), q)

	if res.ShardsTotal != parts {
		t.Fatalf("ShardsTotal = %d, want %d (every shard holds the query terms)", res.ShardsTotal, parts)
	}
	if res.ShardsOK != parts {
		t.Fatalf("ShardsOK = %d, want %d (every shard responds within the full deadline)", res.ShardsOK, parts)
	}
	if !res.Complete() {
		t.Fatalf("result flagged incomplete despite every shard responding")
	}
	if len(res.Hits) == 0 {
		t.Fatalf("complete result returned no hits")
	}

	// The completeness-carrying path and the back-compatible one return the same top-k.
	got := b.Search(context.Background(), q)
	if len(got) != len(res.Hits) {
		t.Fatalf("Search returned %d hits, SearchComplete %d", len(got), len(res.Hits))
	}
	for i := range got {
		if got[i] != res.Hits[i] {
			t.Fatalf("rank %d: Search %+v, SearchComplete %+v", i, got[i], res.Hits[i])
		}
	}
}

// TestSearchPartialWhenShardsDropped checks the honesty pin: when contributing shards
// are dropped at the deadline the result is flagged partial, never passed off as
// complete. It forces the deadline to win the dispatch deterministically by closing
// the fan-out's concurrency (so a slot is never free) and cancelling the context, the
// white-box equivalent of every shard missing the budget, and checks the count of
// shards reached reports the drop.
func TestSearchPartialWhenShardsDropped(t *testing.T) {
	b, parts := brokerForCompleteness(t)
	defer func() { _ = b.Close() }()

	// With no concurrency slot ever free, the dispatch select can only take the
	// cancelled context's branch, so every contributing shard is dropped, the
	// all-shards-miss-the-deadline tail.
	b.maxConcurrency = 0
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res := b.SearchComplete(ctx, Query{Text: "common document", K: 20})

	if res.ShardsTotal != parts {
		t.Fatalf("ShardsTotal = %d, want %d", res.ShardsTotal, parts)
	}
	if res.ShardsOK != 0 {
		t.Fatalf("ShardsOK = %d, want 0 (every shard dropped)", res.ShardsOK)
	}
	if res.Complete() {
		t.Fatalf("a result missing every contributing shard was flagged complete")
	}
	if len(res.Hits) != 0 {
		t.Fatalf("no shard responded yet %d hits came back", len(res.Hits))
	}
}
