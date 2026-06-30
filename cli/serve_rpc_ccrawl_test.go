package cli

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/search"
)

// TestAggregatorOverRemotesCCrawl is the distributed-serving proof on real data: an aggregator
// whose two children are RemoteSearchers, each dialing a broker behind its own httptest server,
// reproduces a single broker over every shard, document for document, on genuine crawl shards.
// It is TestAggregatorPartitionedStatsCCrawl run over the wire: the same real shards split by
// body length across two brokers, but the brokers are now reached over HTTP, so the idf gather,
// the field-average fold, the fan-out, and the merge all cross the network and the merged top-k
// is still the monolith's. This is the real-data form of the search package's
// TestAggregatorOverRemotesMatchesMonolith, the path to a corpus larger than one machine holds.
func TestAggregatorOverRemotesCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	if res.Shards < 4 {
		t.Fatalf("need at least 4 shards to split across two brokers, got %d", res.Shards)
	}

	model := lengthModelCCrawl(t)
	shards, _, owner, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = owner.Close() }()
	pl := buildPipeline(shards)

	mono := search.NewBroker(shards, wideCascade(model))

	// Split the real shards by their own body length, the cross-broker skew the field-average
	// push-down resolves, the same ordering TestAggregatorPartitionedStatsCCrawl builds.
	order := make([]int, len(shards))
	for i := range order {
		order[i] = i
	}
	shardAvg := func(s *search.Shard) float64 {
		return search.NewBroker([]*search.Shard{s}, wideCascade(model)).Stats().AvgFieldLen[1]
	}
	avgs := make([]float64, len(shards))
	for i, s := range shards {
		avgs[i] = shardAvg(s)
	}
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && avgs[order[j]] < avgs[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
	sorted := make([]*search.Shard, len(shards))
	for i, idx := range order {
		sorted[i] = shards[idx]
	}
	half := len(sorted) / 2
	b0 := search.NewBroker(sorted[:half], wideCascade(model))
	b1 := search.NewBroker(sorted[half:], wideCascade(model))

	// Stand each broker up behind its own server and fan the aggregator across the two over the
	// wire, the two-machine deployment in miniature.
	srv0 := httptest.NewServer(http.StripPrefix("/rpc", search.NewSearcherHandler(b0)))
	srv1 := httptest.NewServer(http.StripPrefix("/rpc", search.NewSearcherHandler(b1)))
	defer srv0.Close()
	defer srv1.Close()
	ctx := context.Background()
	rs0, err := search.NewRemoteSearcher(ctx, srv0.URL+"/rpc")
	if err != nil {
		t.Fatalf("dial b0: %v", err)
	}
	rs1, err := search.NewRemoteSearcher(ctx, srv1.URL+"/rpc")
	if err != nil {
		t.Fatalf("dial b1: %v", err)
	}
	agg := search.NewAggregator([]search.Searcher{rs0, rs1})

	// The fold over the remote children's stats lands on the monolith's fleet average, the
	// length-normalization denominator the aggregator pushes back down across the wire.
	if math.Abs(agg.Stats().AvgFieldLen[1]-mono.Stats().AvgFieldLen[1]) > 1e-6 {
		t.Fatalf("folded fleet body avg %v != monolith %v", agg.Stats().AvgFieldLen[1], mono.Stats().AvgFieldLen[1])
	}

	queries := []string{"data", "page", "home", "search", "news", "world", "time", "people", "free", "online"}
	compared, nonTrivial := 0, 0
	for _, qs := range queries {
		pq := pl.parse(qs)
		if pq.Empty() {
			continue
		}
		q := toQuery(pq, 20)
		if len(q.Terms) == 0 {
			continue
		}
		want := mono.SearchComplete(ctx, q)
		got := agg.SearchComplete(ctx, q)
		if !got.Complete() {
			t.Fatalf("query %q over remotes was not complete: %d/%d", qs, got.ShardsOK, got.ShardsTotal)
		}
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("query %q: remote tree returned %d hits, monolith %d", qs, len(got.Hits), len(want.Hits))
		}
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID {
				t.Fatalf("query %q rank %d: remote tree doc %d, monolith doc %d", qs, i, got.Hits[i].DocID, want.Hits[i].DocID)
			}
			if d := math.Abs(got.Hits[i].Score - want.Hits[i].Score); d > 1e-6 {
				t.Fatalf("query %q rank %d doc %d: remote tree score %v, monolith %v, diff %v", qs, i, got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].Score, d)
			}
		}
		if len(want.Hits) > 0 {
			compared++
		}
		for i := 1; i < len(want.Hits); i++ {
			if want.Hits[i].Score != want.Hits[0].Score {
				nonTrivial++
				break
			}
		}
	}
	if compared == 0 {
		t.Fatalf("no real query returned hits over %d docs; nothing was compared", res.Docs)
	}
	if nonTrivial == 0 {
		t.Fatalf("every query's top-k collapsed to one score; the reproduction is vacuous")
	}
	t.Logf("distributed serving over real data: %d shards across 2 remote brokers, %d docs, %d queries reproduced the monolith exactly, %d with a non-trivial ranked top-k",
		res.Shards, res.Docs, compared, nonTrivial)
}
