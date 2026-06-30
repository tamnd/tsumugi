package search

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"
)

// BenchmarkHedgedNoHedge measures the common case the hedge delay is tuned for: the primary
// replica answers before the delay, so no second replica is ever launched. It is the orchestration
// overhead, the timer and the one goroutine and the select loop, a query pays on top of a direct
// call when hedging is wired but does not fire, the cost a deployment accepts everywhere to buy
// tail tolerance only where a replica is slow.
func BenchmarkHedgedNoHedge(b *testing.B) {
	primary := &fakeSearcher{res: complete(1, 4), shards: 4}
	backup := &fakeSearcher{res: complete(2, 4), shards: 4}
	h := NewHedgedSearcher([]Searcher{primary, backup}, time.Hour)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res := h.SearchComplete(ctx, Query{}); !res.Complete() {
			b.Fatal("incomplete")
		}
	}
}

// BenchmarkHedgedOverRemotes measures one query through a HedgedSearcher over two equivalent
// remote replicas of a broker with the primary fast, so the realistic per-query cost when hedging
// is wired over the wire but the primary wins is one RPC round trip plus the orchestration, the
// number that has to stay inside the 10ms budget at each tree level that runs a replica set.
func BenchmarkHedgedOverRemotes(b *testing.B) {
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

	srv0 := httptest.NewServer(NewSearcherHandler(broker))
	srv1 := httptest.NewServer(NewSearcherHandler(broker))
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
	h := NewHedgedSearcher([]Searcher{rs0, rs1}, 5*time.Millisecond)

	q := Query{Terms: []string{"common"}, K: 10}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res := h.SearchComplete(ctx, q); len(res.Hits) == 0 {
			b.Fatal("no hits")
		}
	}
}
