package search

import (
	"context"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

// BenchmarkRemoteSearch measures one query's round trip through the RPC seam over the
// loopback httptest server: marshal the query, POST it, run the broker, marshal the results,
// and decode them. It is the per-hop overhead an aggregator pays for a remote child versus an
// in-process one, the latency the 10ms budget has to absorb at each level of the tree.
func BenchmarkRemoteSearch(b *testing.B) {
	const n, parts = 2000, 4
	docs := remoteRankCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)
	broker, shards := buildBrokerFromDocs(b, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()
	srv := httptest.NewServer(NewSearcherHandler(broker))
	defer srv.Close()
	rs, err := NewRemoteSearcher(context.Background(), srv.URL)
	if err != nil {
		b.Fatalf("dial remote: %v", err)
	}
	q := Query{Terms: []string{"common"}, K: 10}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res := rs.SearchComplete(ctx, q); len(res.Hits) == 0 {
			b.Fatal("no hits")
		}
	}
}

// BenchmarkAggregatorOverRemotes measures a full distributed query: an aggregator gathering
// df across two remote brokers, pushing the shared idf and field averages down, fanning the
// search out over the wire, and merging. It is the wall-clock a head node spends per query
// when its children live on other machines, the number the 100k-shard target must stay under.
func BenchmarkAggregatorOverRemotes(b *testing.B) {
	const n = 4000
	docs := remoteRankCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)

	const parts = 4
	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, "s"+string(rune('0'+p))+".tsumugi")
		lo := p * size
		buildShardFile(b, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
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
	srv0 := httptest.NewServer(NewSearcherHandler(b0))
	srv1 := httptest.NewServer(NewSearcherHandler(b1))
	defer srv0.Close()
	defer srv1.Close()
	rs0, err := NewRemoteSearcher(context.Background(), srv0.URL)
	if err != nil {
		b.Fatalf("dial 0: %v", err)
	}
	rs1, err := NewRemoteSearcher(context.Background(), srv1.URL)
	if err != nil {
		b.Fatalf("dial 1: %v", err)
	}
	agg := NewAggregator([]Searcher{rs0, rs1})

	q := Query{Terms: []string{"alpha", "beta", "common"}, K: 10}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res := agg.SearchComplete(ctx, q); !res.Complete() {
			b.Fatal("incomplete result under no deadline")
		}
	}
}

// BenchmarkRemoteVocab measures streaming a broker's whole vocabulary over /vocab, the cost a
// head node pays once at startup per peer to build its fleet-wide corrector. It is off the
// serving path, so it does not count against the 10ms query budget, but it bounds how long a
// head takes to come up over a large fleet.
func BenchmarkRemoteVocab(b *testing.B) {
	const n, parts = 4000, 4
	docs := remoteRankCorpus(n)
	dir := b.TempDir()
	model := trainModel(b)
	broker, shards := buildBrokerFromDocs(b, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()
	srv := httptest.NewServer(NewSearcherHandler(broker))
	defer srv.Close()
	rs, err := NewRemoteSearcher(context.Background(), srv.URL)
	if err != nil {
		b.Fatalf("dial remote: %v", err)
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n := 0
		if err := rs.Vocab(ctx, func(string, uint32) { n++ }); err != nil {
			b.Fatalf("vocab: %v", err)
		}
		if n == 0 {
			b.Fatal("empty vocabulary")
		}
	}
}
