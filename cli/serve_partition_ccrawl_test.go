package cli

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// lengthModelCCrawl fits a ranking model whose learned signal is the online body-BM25 feature,
// the one column the per-field length normalization moves. The default serve model trains only
// the offline matrix columns, so the online BM25F features carry zero ranking weight and the
// partitioned field-average gap is invisible to it; this model weights OnBM25Body so a wrong body
// denominator changes the ranking it produces, which is what makes the real-data partitioned test
// a test of the field-average push-down rather than of features that ignore it. The label tracks
// the body-BM25 column over the concatenated offline-plus-online width the L2 model scores, spread
// across the range real BM25 lands in. It is left unstamped so the broker's model check skips it.
func lengthModelCCrawl(t *testing.T) *rank.Model {
	t.Helper()
	nf := len(feature.DefaultSchema())
	width := nf + int(search.NumOnline)
	bodyCol := nf + int(search.OnBM25Body)
	d := &rank.Dataset{NumFeatures: width}
	var s uint64 = 7
	rnd := func() float64 {
		s = s*6364136223846793005 + 1442695040888963407
		return float64(s>>11) / float64(1<<53)
	}
	for q := 0; q < 80; q++ {
		d.Groups = append(d.Groups, 12)
		for i := 0; i < 12; i++ {
			row := make([]float64, width)
			for f := range row {
				row[f] = rnd()
			}
			row[bodyCol] = rnd() * 20
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, row[bodyCol])
		}
	}
	p := rank.DefaultParams()
	p.Rounds = 80
	return rank.Train(d, p).Compile()
}

// wideCascade keeps the cut wide so the recall-complete candidate set reaches the reranker
// intact, the precondition the cross-broker exactness proof rests on, mirroring the search
// package's test cascade.
func wideCascade(model *rank.Model) *rank.Cascade {
	c := rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
	c.L0Max = 1 << 20
	c.L1Keep = 1 << 20
	return c
}

// TestAggregatorPartitionedStatsCCrawl is the partitioned-GlobalStats reproduction gate on real
// data: an aggregator over two brokers that split the real shards reproduces the ranked top-k a
// single broker over every shard produces, document for document, because the aggregator folds
// the brokers' field averages into one deployment-wide set and pushes it down so both brokers
// normalize BM25F against the same denominators. The model weights the online body-BM25 feature,
// so the length normalization participates in the ranking, which is what TestAggregatorCCrawl
// could not assert: with the field averages unified the partitioned tree's ranking is the
// monolith's. The load-bearing proof that the broker actually re-normalizes against a pushed
// average, that a different denominator moves the score, is the search package's
// TestBrokerHonorsAvgFieldLenOverride, where the skew and the model's feature range are both
// controlled so the change is guaranteed to cross a leaf. On this crawl sample the natural
// cross-broker length skew is small enough that it does not cross the tree model's leaves, so the
// real-data test asserts the integration over genuine collection.Build forward regions: the fold
// matches the monolith's averages, the ranking is feature-driven rather than a flat tie broken by
// id, and the partitioned tree reproduces the single index.
func TestAggregatorPartitionedStatsCCrawl(t *testing.T) {
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

	// openShards owns the file mappings through the broker it returns; that broker is the one
	// closed. The monolith and the two sub-brokers below reference the same read-only shards with
	// a wide cascade so the candidate set is recall-complete, and so must not close them again.
	shards, _, owner, err := openShards(out, model)
	if err != nil {
		t.Fatalf("openShards: %v", err)
	}
	defer func() { _ = owner.Close() }()
	pl := buildPipeline(shards)

	mono := search.NewBroker(shards, wideCascade(model))

	// Order the real shards by their own average body length and split at the median, so one
	// broker holds the short-bodied shards and the other the long-bodied ones. A random split of
	// crawl shards leaves both halves near the fleet average, too close for the length
	// normalization to move the ranking; partitioning by length reproduces on real data the
	// cross-broker skew the synthetic test builds in, the case a broker normalizing against its
	// own half's lengths scores a document differently than a single index over the whole corpus.
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
	agg := search.NewAggregator([]search.Searcher{b0, b1})

	// The brokers' own body averages really are skewed across the length split, the precondition
	// that makes the push-down do work, and the fold lands on the monolith's fleet average.
	if b0.Stats().AvgFieldLen[1] >= b1.Stats().AvgFieldLen[1] {
		t.Fatalf("expected skew: short-half body avg %v should be below long-half %v", b0.Stats().AvgFieldLen[1], b1.Stats().AvgFieldLen[1])
	}
	if math.Abs(agg.Stats().AvgFieldLen[1]-mono.Stats().AvgFieldLen[1]) > 1e-6 {
		t.Fatalf("folded fleet body avg %v != monolith %v", agg.Stats().AvgFieldLen[1], mono.Stats().AvgFieldLen[1])
	}

	ctx := context.Background()
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

		// The partitioned tree reproduces the monolith: the aggregator folds the brokers' field
		// averages into one fleet set and pushes it down, so both halves normalize BM25F against
		// the same denominators and their merged top-k is the single-index top-k. The monolith
		// computes its idf and averages over every shard at once; the aggregator gathers the same
		// idf and folds the same averages, so the two land on the same scores.
		want := mono.Search(ctx, q)
		got := agg.Search(ctx, q)
		if len(got) != len(want) {
			t.Fatalf("query %q: aggregator returned %d hits, monolith %d", qs, len(got), len(want))
		}
		for i := range want {
			if got[i].DocID != want[i].DocID {
				t.Fatalf("query %q rank %d: aggregator doc %d, monolith doc %d", qs, i, got[i].DocID, want[i].DocID)
			}
			// Scores match up to the floating-point rounding of reconstructing the field sums from
			// the brokers' averages, the documented negligible error of the fold.
			if d := math.Abs(got[i].Score - want[i].Score); d > 1e-6 {
				t.Fatalf("query %q rank %d doc %d: aggregator score %v, monolith %v, diff %v exceeds rounding", qs, i, got[i].DocID, got[i].Score, want[i].Score, d)
			}
		}
		if len(want) > 0 {
			compared++
		}
		// The reproduction is non-trivial only if the model actually ranks on the body-BM25 signal,
		// so the top-k must carry a real score spread rather than collapsing to one tie the order
		// would then break by id. A query whose hits all score the same proves nothing about the
		// length normalization, so it is not counted.
		for i := 1; i < len(want); i++ {
			if want[i].Score != want[0].Score {
				nonTrivial++
				break
			}
		}
	}
	if compared == 0 {
		t.Fatalf("no real query returned hits over %d docs; nothing was compared", res.Docs)
	}
	if nonTrivial == 0 {
		t.Fatalf("every query's top-k collapsed to one score; the model is not ranking, so the reproduction is vacuous")
	}
	t.Logf("partitioned stats over real data: %d shards, %d docs, %d queries reproduced the monolith exactly, %d with a non-trivial ranked top-k",
		res.Shards, res.Docs, compared, nonTrivial)
}
