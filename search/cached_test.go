package search

import (
	"context"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

// TestCachedSearcherHitEqualsMiss is the core cache contract over a Searcher: the result a hit
// returns is bit-identical to the cold result a miss computes, and the second call to the same
// query is a hit while the first is a miss. It runs over a real broker behind the interface, so
// it proves the decorator caches the wrapped Searcher's own answer rather than recomputing a
// different one.
func TestCachedSearcherHitEqualsMiss(t *testing.T) {
	const n, parts = 160, 4
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)
	broker, shards := buildBrokerFromDocs(t, dir, "c", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	cs := NewCachedSearcher(broker, NewResultCache(8))
	ctx := context.Background()
	q := Query{Terms: []string{"common"}, K: 10}

	cold, hit := cs.SearchCached(ctx, q)
	if hit {
		t.Fatalf("first call to a fresh cache was a hit")
	}
	warm, hit := cs.SearchCached(ctx, q)
	if !hit {
		t.Fatalf("second call to the same query was a miss; the result was not cached")
	}
	if len(warm.Hits) != len(cold.Hits) {
		t.Fatalf("warm returned %d hits, cold %d", len(warm.Hits), len(cold.Hits))
	}
	if warm.ShardsTotal != cold.ShardsTotal || warm.ShardsOK != cold.ShardsOK {
		t.Fatalf("warm completeness %d/%d, cold %d/%d", warm.ShardsOK, warm.ShardsTotal, cold.ShardsOK, cold.ShardsTotal)
	}
	for i := range cold.Hits {
		if warm.Hits[i].DocID != cold.Hits[i].DocID || warm.Hits[i].Score != cold.Hits[i].Score {
			t.Fatalf("hit %d: warm {%d,%v}, cold {%d,%v}", i,
				warm.Hits[i].DocID, warm.Hits[i].Score, cold.Hits[i].DocID, cold.Hits[i].Score)
		}
	}
	if cs.ResultCache().Len() != 1 {
		t.Fatalf("cache holds %d entries after one query, want 1", cs.ResultCache().Len())
	}
}

// TestCachedSearcherIsSearcher pins that the decorator delegates the rest of the Searcher
// contract to the wrapped node, so a parent aggregator fans across a CachedSearcher exactly as
// it does across the broker it wraps: same NumShards, NumDocs, Stats, DocFreqs, and the same
// SearchComplete answer.
func TestCachedSearcherIsSearcher(t *testing.T) {
	const n, parts = 120, 3
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)
	broker, shards := buildBrokerFromDocs(t, dir, "d", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	var s Searcher = NewCachedSearcher(broker, NewResultCache(4))
	ctx := context.Background()
	if s.NumShards() != broker.NumShards() {
		t.Fatalf("decorator NumShards %d, broker %d", s.NumShards(), broker.NumShards())
	}
	if s.NumDocs() != broker.NumDocs() {
		t.Fatalf("decorator NumDocs %d, broker %d", s.NumDocs(), broker.NumDocs())
	}
	if s.Stats().DocCount != broker.Stats().DocCount {
		t.Fatalf("decorator Stats.DocCount %d, broker %d", s.Stats().DocCount, broker.Stats().DocCount)
	}
	df := s.DocFreqs(ctx, []string{"common", "alpha"})
	bdf := broker.DocFreqs(ctx, []string{"common", "alpha"})
	for term, want := range bdf {
		if df[term] != want {
			t.Fatalf("decorator df[%q] = %d, broker %d", term, df[term], want)
		}
	}
	q := Query{Terms: []string{"common"}, K: 10}
	got := s.SearchComplete(ctx, q)
	want := broker.SearchComplete(ctx, q)
	if len(got.Hits) != len(want.Hits) {
		t.Fatalf("decorator SearchComplete %d hits, broker %d", len(got.Hits), len(want.Hits))
	}
}

// TestCachedSearcherOverAggregatorMatchesMonolith is the head-cache proof: a CachedSearcher in
// front of an aggregator over two remote brokers serves the same top-k a single broker over
// every shard would, on both the cold miss and the warm hit. This is the distributed serving
// tree of TestAggregatorOverRemotesMatchesMonolith with the head cache the serve head-node mode
// runs in front of it, so it proves the cache composes over the whole tree, not just one broker.
func TestCachedSearcherOverAggregatorMatchesMonolith(t *testing.T) {
	const n = 200
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	monoBroker := NewBroker([]*Shard{ms}, newTestCascade(model))
	defer func() { _ = monoBroker.Close() }()

	const parts = 4
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, "s"+string(rune('0'+p))+".tsumugi")
		lo := p * size
		buildShardFile(t, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()
	b0 := NewBroker(shards[:2], newTestCascade(model))
	b1 := NewBroker(shards[2:], newTestCascade(model))
	agg := NewAggregator([]Searcher{serveSearcher(t, b0), serveSearcher(t, b1)})
	head := NewCachedSearcher(agg, NewResultCache(16))

	ctx := context.Background()
	nontrivial := 0
	for _, q := range []Query{
		{Terms: []string{"common"}, K: 10},
		{Terms: []string{"alpha", "common"}, K: 20},
		{Terms: []string{"beta", "common"}, K: 20},
		{Terms: []string{"alpha", "beta", "common"}, K: 25},
	} {
		want := monoBroker.SearchComplete(ctx, q)
		// First call is a cold miss through the whole tree; the second is a head-cache hit. Both
		// must reproduce the monolith document for document.
		cold, hit := head.SearchCached(ctx, q)
		if hit {
			t.Fatalf("query %v was a hit on a cold cache", q.Terms)
		}
		warm, hit := head.SearchCached(ctx, q)
		if !hit {
			t.Fatalf("query %v was not cached after a cold miss", q.Terms)
		}
		for _, got := range []Results{cold, warm} {
			if !got.Complete() {
				t.Fatalf("query %v over the tree was not complete: %d/%d", q.Terms, got.ShardsOK, got.ShardsTotal)
			}
			if len(got.Hits) != len(want.Hits) {
				t.Fatalf("query %v: tree returned %d hits, monolith %d", q.Terms, len(got.Hits), len(want.Hits))
			}
			for i := range want.Hits {
				if got.Hits[i].DocID != want.Hits[i].DocID || got.Hits[i].Score != want.Hits[i].Score {
					t.Fatalf("query %v rank %d: tree {%d,%v}, monolith {%d,%v}", q.Terms, i,
						got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].DocID, want.Hits[i].Score)
				}
			}
		}
		if len(cold.Hits) > 1 && cold.Hits[0].Score != cold.Hits[len(cold.Hits)-1].Score {
			nontrivial++
		}
	}
	if nontrivial == 0 {
		t.Fatalf("every query's top-k collapsed to one score; the reproduction is vacuous")
	}
}

// TestCachedSearcherSkipsPartial pins the load-bearing rule: a partial result, one where a child
// was dropped at the deadline so the answer is incomplete, is never cached, because a later query
// with the budget to reach every child must not be served the transient degraded answer. The
// dropped child is a remote peer closed after construction, the cross-machine drop the head sees.
func TestCachedSearcherSkipsPartial(t *testing.T) {
	const n, parts = 120, 2
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	live, liveShards := buildBrokerFromDocs(t, dir, "live", docs[:n/2], parts, model)
	defer func() {
		for _, sh := range liveShards {
			_ = sh.Close()
		}
	}()
	dead, deadShards := buildBrokerFromDocs(t, dir, "dead", docs[n/2:], parts, model)

	liveRemote := serveSearcher(t, live)
	deadRemote := dialThenKill(t, dead)
	for _, sh := range deadShards {
		_ = sh.Close()
	}

	agg := NewAggregator([]Searcher{liveRemote, deadRemote})
	head := NewCachedSearcher(agg, NewResultCache(8))

	ctx := context.Background()
	q := Query{Terms: []string{"common"}, K: 10}
	res, hit := head.SearchCached(ctx, q)
	if hit {
		t.Fatalf("partial query was a hit on a cold cache")
	}
	if res.Complete() {
		t.Fatalf("query over a dead child reported complete: %d/%d", res.ShardsOK, res.ShardsTotal)
	}
	if head.ResultCache().Len() != 0 {
		t.Fatalf("partial result was cached: %d entries", head.ResultCache().Len())
	}
	// A second call is still a miss, since nothing was cached, so the head keeps trying the full
	// tree rather than serving the stale partial.
	if _, hit := head.SearchCached(ctx, q); hit {
		t.Fatalf("partial result served from cache on the second call")
	}
}

// TestCachedSearcherNilCache pins that a nil or zero-capacity cache leaves SearchCached a plain
// SearchComplete with a false hit flag, so a caller uses the decorator uniformly with the cache
// off.
func TestCachedSearcherNilCache(t *testing.T) {
	const n, parts = 80, 2
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)
	broker, shards := buildBrokerFromDocs(t, dir, "n", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	for _, cs := range []*CachedSearcher{
		NewCachedSearcher(broker, nil),
		NewCachedSearcher(broker, NewResultCache(0)),
	} {
		ctx := context.Background()
		q := Query{Terms: []string{"common"}, K: 10}
		_, hit := cs.SearchCached(ctx, q)
		if hit {
			t.Fatalf("disabled cache reported a hit")
		}
		if _, hit := cs.SearchCached(ctx, q); hit {
			t.Fatalf("disabled cache cached an entry and hit on it")
		}
	}
}

// dialThenKill stands a Searcher behind an httptest server, dials a RemoteSearcher at it so the
// metadata snapshot is captured, then closes the server, so the remote answers NumShards from its
// snapshot but fails every query, the peer-dies-after-construction case.
func dialThenKill(t *testing.T, s Searcher) *RemoteSearcher {
	t.Helper()
	srv := httptest.NewServer(NewSearcherHandler(s))
	rs, err := NewRemoteSearcher(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("dial remote: %v", err)
	}
	srv.Close()
	return rs
}
