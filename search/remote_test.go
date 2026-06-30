package search

import (
	"context"
	"math"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/rank"
)

// buildBrokerFromDocs builds a broker over `parts` shards split evenly across the given docs,
// returning the broker and the open shards so the caller can close them. It is the shared
// setup for the remote tests: a broker stood up here is wrapped in a handler and reached over
// the wire, and the monolith is one broker over a single shard holding every document.
func buildBrokerFromDocs(t testing.TB, dir, prefix string, docs []doc, parts int, model *rank.Model) (*Broker, []*Shard) {
	t.Helper()
	n := len(docs)
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, prefix+string(rune('0'+p))+".tsumugi")
		lo := p * size
		hi := lo + size
		if p == parts-1 {
			hi = n
		}
		buildShardFile(t, path, docs, lo, hi, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	return NewBroker(shards, newTestCascade(model)), shards
}

// remoteRankCorpus is the cross-broker df corpus of dfCorpus, alpha only in the first
// quarter and beta only in the last so each of two brokers holds one rare term, but with the
// in-degree feature spread across [0,1] rather than dfCorpus's raw 0..4095. trainModel learns
// label = round(inDegree*4) over features in [0,1], so a corpus whose in-degree lands in that
// range gives the model a signal to discriminate on and the merged top-k a genuine ranked
// order; dfCorpus's out-of-range values saturate the tree to one leaf and flatten the scores,
// which proves df gathering but not the score-ordered merge this test is for.
func remoteRankCorpus(n int) []doc {
	docs := dfCorpus(n)
	for g := range docs {
		// A deterministic spread in [0,1] that is not monotonic in the id, so the ranked order
		// interleaves documents from different shards and the cross-broker merge has to order
		// candidates by score rather than by which broker returned them.
		docs[g].feats[feature.FeatInDegree] = float64((g*37)%100) / 100
	}
	return docs
}

// serveSearcher stands a Searcher up behind an httptest server speaking the RPC wire and
// returns a RemoteSearcher dialed at it, registering the teardown. It is the loopback the
// remote tests run over: a real broker on one side, a RemoteSearcher on the other, with a
// real HTTP round trip between them, so the test exercises the encode, the transport, and the
// decode rather than a mock.
func serveSearcher(t *testing.T, s Searcher, opts ...RemoteOption) *RemoteSearcher {
	t.Helper()
	srv := httptest.NewServer(NewSearcherHandler(s))
	t.Cleanup(srv.Close)
	rs, err := NewRemoteSearcher(context.Background(), srv.URL, opts...)
	if err != nil {
		t.Fatalf("dial remote: %v", err)
	}
	return rs
}

// TestRemoteSearcherMatchesLocal is the bit-exactness proof for the wire itself: a
// RemoteSearcher over an httptest server reproduces the in-process broker's own answer exactly,
// the same hits in the same order with scores equal to the float, plus the same NumShards,
// NumDocs, DocFreqs, and Stats. If the JSON encoding lost or reordered anything, this fails.
func TestRemoteSearcherMatchesLocal(t *testing.T) {
	const n, parts = 120, 3
	docs := dfCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()
	rs := serveSearcher(t, broker)

	if rs.NumShards() != broker.NumShards() {
		t.Fatalf("remote NumShards = %d, local %d", rs.NumShards(), broker.NumShards())
	}
	if rs.NumDocs() != broker.NumDocs() {
		t.Fatalf("remote NumDocs = %d, local %d", rs.NumDocs(), broker.NumDocs())
	}
	if rs.Stats() != broker.Stats() {
		t.Fatalf("remote Stats = %+v, local %+v", rs.Stats(), broker.Stats())
	}

	ctx := context.Background()
	terms := []string{"alpha", "beta", "common"}
	wantDF := broker.DocFreqs(ctx, terms)
	gotDF := rs.DocFreqs(ctx, terms)
	for _, term := range terms {
		if gotDF[term] != wantDF[term] {
			t.Fatalf("remote df[%q] = %d, local %d", term, gotDF[term], wantDF[term])
		}
	}

	for _, q := range []Query{
		{Terms: []string{"common"}, K: 10},
		{Terms: []string{"alpha", "common"}, K: 15},
		{Terms: []string{"beta", "common"}, K: 5},
	} {
		want := broker.SearchComplete(ctx, q)
		got := rs.SearchComplete(ctx, q)
		if got.ShardsTotal != want.ShardsTotal || got.ShardsOK != want.ShardsOK {
			t.Fatalf("remote completeness %d/%d, local %d/%d", got.ShardsOK, got.ShardsTotal, want.ShardsOK, want.ShardsTotal)
		}
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("remote returned %d hits, local %d", len(got.Hits), len(want.Hits))
		}
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID || got.Hits[i].Score != want.Hits[i].Score {
				t.Fatalf("hit %d: remote {%d,%v}, local {%d,%v}", i,
					got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].DocID, want.Hits[i].Score)
			}
		}
	}
}

// TestAggregatorOverRemotesMatchesMonolith is the distributed serving proof: an aggregator
// whose children are RemoteSearchers, each dialing a broker behind its own httptest server,
// reproduces a single broker over every shard, exactly. The corpus is the cross-broker df
// corpus, so the rare terms force the aggregator's idf gather across the wire, and the merge
// is over real model scores carried back as JSON. This is the in-process aggregator exactness
// test (TestAggregatorFleetDFMatchesMonolith and its merge siblings) run over the network.
func TestAggregatorOverRemotesMatchesMonolith(t *testing.T) {
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

	// Build four shards over the full corpus with disjoint global id ranges, the way a
	// deployment assigns each shard a node base, so the shard ids align with the monolith's.
	// Each shard covers a quarter, so grouping the first two under one broker and the last two
	// under another puts exactly one rare term in each broker, the cross-broker skew the idf
	// gather must cross the wire to resolve.
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

	ctx := context.Background()
	if agg.NumDocs() != monoBroker.NumDocs() {
		t.Fatalf("aggregator NumDocs = %d, monolith %d", agg.NumDocs(), monoBroker.NumDocs())
	}
	// The monolith is one shard over every document; the remote tree fans the same documents
	// across four shards under two brokers. The document counts must match, the shard counts
	// must not, so the tree is genuinely distributing what the monolith holds in one place.
	if agg.NumShards() != 4 {
		t.Fatalf("aggregator NumShards = %d, want 4", agg.NumShards())
	}

	nontrivial := 0
	for _, q := range []Query{
		{Terms: []string{"common"}, K: 10},
		{Terms: []string{"alpha", "common"}, K: 20},
		{Terms: []string{"beta", "common"}, K: 20},
		{Terms: []string{"alpha", "beta", "common"}, K: 25},
	} {
		want := monoBroker.SearchComplete(ctx, q)
		got := agg.SearchComplete(ctx, q)
		if !got.Complete() {
			t.Fatalf("query %v over remotes was not complete: %d/%d", q.Terms, got.ShardsOK, got.ShardsTotal)
		}
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("query %v: remote tree returned %d hits, monolith %d", q.Terms, len(got.Hits), len(want.Hits))
		}
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID {
				t.Fatalf("query %v hit %d: remote tree id %d, monolith %d", q.Terms, i, got.Hits[i].DocID, want.Hits[i].DocID)
			}
			if math.Abs(got.Hits[i].Score-want.Hits[i].Score) > 1e-6 {
				t.Fatalf("query %v hit %d: remote tree score %v, monolith %v", q.Terms, i, got.Hits[i].Score, want.Hits[i].Score)
			}
		}
		if len(got.Hits) > 1 && got.Hits[0].Score != got.Hits[len(got.Hits)-1].Score {
			nontrivial++
		}
	}
	if nontrivial == 0 {
		t.Fatal("every query produced a flat ranking; the corpus or model is not exercising the merge")
	}
	t.Logf("%d shards over the wire reproduced the monolith exactly, %d queries with a non-trivial ranked top-k",
		agg.NumShards(), nontrivial)
}

// TestRemoteSearcherDroppedSubtree checks that an unreachable peer is charged as a fully
// missed subtree rather than silently dropped: its SearchComplete reports its whole shard
// count as total and nothing reached, and an aggregator that holds it rolls the answer up as
// incomplete. This is the cross-machine form of the broker's deadline drop, the honesty doc 11
// requires of a partial result.
func TestRemoteSearcherDroppedSubtree(t *testing.T) {
	const n, parts = 80, 2
	docs := dfCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	// Dial a live server for the metadata snapshot, then close it so the per-query call fails,
	// the way a node that was up at construction goes down before a query.
	srv := httptest.NewServer(NewSearcherHandler(broker))
	rs, err := NewRemoteSearcher(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("dial remote: %v", err)
	}
	srv.Close()

	res := rs.SearchComplete(context.Background(), Query{Terms: []string{"common"}, K: 10})
	if res.Complete() {
		t.Fatal("a dead peer reported a complete result")
	}
	if res.ShardsTotal != parts || res.ShardsOK != 0 {
		t.Fatalf("dead peer completeness = %d/%d, want 0/%d", res.ShardsOK, res.ShardsTotal, parts)
	}
	if len(res.Hits) != 0 {
		t.Fatalf("dead peer returned %d hits, want none", len(res.Hits))
	}

	// An aggregator that holds a live child and a dead one returns the live child's hits but
	// flags the whole answer incomplete, the partial-result roll-up across hosts.
	live := serveSearcher(t, broker)
	agg := NewAggregator([]Searcher{live, rs})
	got := agg.SearchComplete(context.Background(), Query{Terms: []string{"common"}, K: 10})
	if got.Complete() {
		t.Fatal("aggregator with a dead child reported complete")
	}
	if got.ShardsOK != parts {
		t.Fatalf("aggregator reached %d shards, want the %d live ones", got.ShardsOK, parts)
	}
	if got.ShardsTotal != 2*parts {
		t.Fatalf("aggregator total = %d, want %d (both subtrees charged)", got.ShardsTotal, 2*parts)
	}
	if len(got.Hits) == 0 {
		t.Fatal("aggregator dropped the live child's hits along with the dead one's")
	}
}

// TestRemoteVocabMatchesLocal checks the /vocab stream reproduces the broker's own merged
// vocabulary exactly: every term the broker enumerates locally with ForEachTerm comes back over
// the wire with the same document frequency, and nothing extra. This is what lets a head node
// build the same corrector dictionary from a peer it would build from the peer's local shards.
func TestRemoteVocabMatchesLocal(t *testing.T) {
	const n, parts = 80, 2
	docs := dfCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	want := map[string]uint32{}
	broker.ForEachTerm(func(term string, df uint32) { want[term] = df })
	if len(want) == 0 {
		t.Fatal("broker has no vocabulary to compare")
	}

	rs := serveSearcher(t, broker)
	got := map[string]uint32{}
	if err := rs.Vocab(context.Background(), func(term string, df uint32) { got[term] = df }); err != nil {
		t.Fatalf("vocab: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("vocab streamed %d terms, broker has %d", len(got), len(want))
	}
	for term, df := range want {
		if got[term] != df {
			t.Fatalf("term %q df over wire = %d, local = %d", term, got[term], df)
		}
	}
}

// TestRemoteVectorDimMatchesLocal checks the dense dimension rides the /meta snapshot: a broker
// over shards that carry a vector region reports the same dimension over the wire that its own
// VectorDim reports, so a head node builds its dense encoder at the width every leaf can read.
// A broker with no vector region reports the dense plane off, so the head leaves its encoder off.
func TestRemoteVectorDimMatchesLocal(t *testing.T) {
	const n = 120
	docs := makeCorpus(n)
	model := trainModel(t)

	t.Run("with vector region", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "v.tsumugi")
		buildShardFile(t, path, docs, 0, n, 0, true)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard: %v", err)
		}
		defer func() { _ = sh.Close() }()
		broker := NewBroker([]*Shard{sh}, newTestCascade(model))

		wantDim, wantOK := broker.VectorDim()
		if !wantOK {
			t.Fatal("broker over a vector shard reports no dense plane")
		}
		rs := serveSearcher(t, broker)
		gotDim, gotOK := rs.VectorDim()
		if gotOK != wantOK || gotDim != wantDim {
			t.Fatalf("remote VectorDim = (%d,%t), local = (%d,%t)", gotDim, gotOK, wantDim, wantOK)
		}
	})

	t.Run("without vector region", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "p.tsumugi")
		buildShardFile(t, path, docs, 0, n, 0, false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard: %v", err)
		}
		defer func() { _ = sh.Close() }()
		broker := NewBroker([]*Shard{sh}, newTestCascade(model))

		rs := serveSearcher(t, broker)
		if dim, ok := rs.VectorDim(); ok {
			t.Fatalf("remote reported a dense plane (dim %d) for a broker with no vector region", dim)
		}
	})
}

// TestRemoteVocabAbsent checks a node that does not serve a vocabulary, a bare aggregator whose
// terms live on its remote children, answers /vocab with an error rather than an empty stream,
// so a head dialing it knows to gather vocabulary from that node's children instead of taking
// absent for empty.
func TestRemoteVocabAbsent(t *testing.T) {
	const n, parts = 40, 2
	docs := dfCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	// An aggregator over a remote child has no ForEachTerm of its own, so a RemoteSearcher dialed
	// at an aggregator's handler gets a 404 on /vocab.
	agg := NewAggregator([]Searcher{serveSearcher(t, broker)})
	rs := serveSearcher(t, agg)
	err := rs.Vocab(context.Background(), func(string, uint32) {})
	if err == nil {
		t.Fatal("vocab on a node without a vocabulary returned no error")
	}
}
