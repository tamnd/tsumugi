package search

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/rank"
)

// openParts builds parts shards over an n-document corpus and returns them opened, along
// with the corpus and the trained model, for the aggregator tests to wire into brokers
// however they need.
func openParts(t *testing.T, n, parts int) ([]*Shard, []doc, *rank.Model) {
	t.Helper()
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
	return shards, docs, model
}

// TestAggregatorExactOverBrokers is the exactness-up-the-tree gate: an aggregator over two
// brokers, each over half the shards, returns the same top-k a single index over every
// shard produces, bit for bit on ids and scores. It builds one monolith shard over the
// whole corpus as the ground truth, partitions the corpus into four shards split across two
// brokers, wraps the brokers in an aggregator, and requires the aggregator's top-k to equal
// the monolith's. The corpus scores on features rather than idf, so a document's score does
// not depend on which broker holds it, which is what makes the cross-broker merge exact.
func TestAggregatorExactOverBrokers(t *testing.T) {
	const n, parts = 160, 4
	docs := makeCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	defer func() { _ = ms.Close() }()

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
	b0 := NewBroker(shards[:2], newTestCascade(model))
	b1 := NewBroker(shards[2:], newTestCascade(model))
	agg := NewAggregator([]Searcher{b0, b1})
	defer func() { _ = b0.Close(); _ = b1.Close() }()

	if agg.NumShards() != parts {
		t.Fatalf("aggregator NumShards = %d, want %d", agg.NumShards(), parts)
	}

	q := Query{Text: "common document", K: 20}
	want := ms.Search(q)
	got := agg.Search(context.Background(), q)

	if len(got) != len(want) {
		t.Fatalf("aggregator returned %d hits, monolith %d", len(got), len(want))
	}
	for i := range want {
		if got[i].DocID != want[i].DocID {
			t.Fatalf("rank %d: aggregator doc %d, monolith doc %d", i, got[i].DocID, want[i].DocID)
		}
		if got[i].Score != want[i].Score {
			t.Fatalf("rank %d doc %d: aggregator score %v, monolith %v", i, got[i].DocID, got[i].Score, want[i].Score)
		}
	}
}

// TestAggregatorExactTwoLevels checks the tree composes exactness at depth two: an
// aggregator over an aggregator over brokers returns the monolith's top-k. It is the proof
// that the Searcher nesting works, since a sub-aggregator is just another child to the root.
func TestAggregatorExactTwoLevels(t *testing.T) {
	const n, parts = 160, 4
	docs := makeCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	defer func() { _ = ms.Close() }()

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
	// Two brokers of one shard each, wrapped in a sub-aggregator, plus one broker of two
	// shards directly under the root: a mixed tree of brokers and a sub-aggregator.
	b0 := NewBroker(shards[:1], newTestCascade(model))
	b1 := NewBroker(shards[1:2], newTestCascade(model))
	b2 := NewBroker(shards[2:], newTestCascade(model))
	sub := NewAggregator([]Searcher{b0, b1})
	root := NewAggregator([]Searcher{sub, b2})
	defer func() { _ = b0.Close(); _ = b1.Close(); _ = b2.Close() }()

	if root.NumShards() != parts {
		t.Fatalf("root NumShards = %d, want %d", root.NumShards(), parts)
	}

	q := Query{Text: "common document", K: 20}
	want := ms.Search(q)
	got := root.Search(context.Background(), q)
	if len(got) != len(want) {
		t.Fatalf("tree returned %d hits, monolith %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rank %d: tree %+v, monolith %+v", i, got[i], want[i])
		}
	}
}

// TestAggregatorCompleteWhenChildrenRespond checks that with the full deadline every child
// responds, so the aggregator's result is complete and its shard counts are the sum over
// the children, and the hits match the back-compatible Search.
func TestAggregatorCompleteWhenChildrenRespond(t *testing.T) {
	const n, parts = 160, 4
	shards, _, model := openParts(t, n, parts)
	b0 := NewBroker(shards[:2], newTestCascade(model))
	b1 := NewBroker(shards[2:], newTestCascade(model))
	agg := NewAggregator([]Searcher{b0, b1})
	defer func() { _ = b0.Close(); _ = b1.Close() }()

	q := Query{Text: "common document", K: 20}
	res := agg.SearchComplete(context.Background(), q)

	if res.ShardsTotal != parts {
		t.Fatalf("ShardsTotal = %d, want %d (sum over the children)", res.ShardsTotal, parts)
	}
	if res.ShardsOK != parts {
		t.Fatalf("ShardsOK = %d, want %d (every child responds)", res.ShardsOK, parts)
	}
	if !res.Complete() {
		t.Fatalf("aggregator flagged incomplete despite every child responding")
	}
	if len(res.Hits) == 0 {
		t.Fatalf("complete aggregator returned no hits")
	}
	got := agg.Search(context.Background(), q)
	if len(got) != len(res.Hits) {
		t.Fatalf("Search returned %d hits, SearchComplete %d", len(got), len(res.Hits))
	}
	for i := range got {
		if got[i] != res.Hits[i] {
			t.Fatalf("rank %d: Search %+v, SearchComplete %+v", i, got[i], res.Hits[i])
		}
	}
}

// TestAggregatorPartialWhenChildrenDropped is the honesty pin one level up: when every child
// is dropped at the deadline the aggregator reports its full shard total with nothing
// reached and flags the result partial. It forces the deadline to win deterministically by
// closing the fan-out's concurrency and cancelling the context, the white-box equivalent of
// every child missing the budget, and checks the dropped children's shards are counted into
// the total from their static NumShards so a fully-dropped tree still reports its size.
func TestAggregatorPartialWhenChildrenDropped(t *testing.T) {
	const n, parts = 160, 4
	shards, _, model := openParts(t, n, parts)
	b0 := NewBroker(shards[:2], newTestCascade(model))
	b1 := NewBroker(shards[2:], newTestCascade(model))
	agg := NewAggregator([]Searcher{b0, b1})
	defer func() { _ = b0.Close(); _ = b1.Close() }()

	agg.maxConcurrency = 0
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res := agg.SearchComplete(ctx, Query{Text: "common document", K: 20})

	if res.ShardsTotal != parts {
		t.Fatalf("ShardsTotal = %d, want %d (dropped children counted from NumShards)", res.ShardsTotal, parts)
	}
	if res.ShardsOK != 0 {
		t.Fatalf("ShardsOK = %d, want 0 (every child dropped)", res.ShardsOK)
	}
	if res.Complete() {
		t.Fatalf("a result missing every child was flagged complete")
	}
	if len(res.Hits) != 0 {
		t.Fatalf("no child responded yet %d hits came back", len(res.Hits))
	}
}
