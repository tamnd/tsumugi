package collection_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// scaleModel trains a real LambdaMART model over the canonical feature schema so the
// latency this test measures runs the production rerank path, the QuickScorer tree walk
// over the survivors, not a constant model that skips it. The training set is synthetic
// because the number we want is the model's per-query cost, which depends on the tree
// count and depth, not on what the trees learned.
func scaleModel(t testing.TB) *rank.Model {
	t.Helper()
	cols := feature.DefaultSchema()
	nf := len(cols)
	d := &rank.Dataset{NumFeatures: nf}
	s := uint64(1234567)
	rnd := func() float64 {
		s = s*6364136223846793005 + 1442695040888963407
		return float64(s>>11) / float64(1<<53)
	}
	const queries, per = 80, 12
	for q := 0; q < queries; q++ {
		d.Groups = append(d.Groups, per)
		for i := 0; i < per; i++ {
			row := make([]float64, nf)
			for f := range row {
				row[f] = rnd()
			}
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, math.Round(row[0]*4))
		}
	}
	p := rank.DefaultParams()
	return rank.Train(d, p).Compile()
}

// dirBytes sums the size of every regular file under a directory, the collection's
// total on-disk footprint.
func dirBytes(t testing.TB, dir string) int64 {
	t.Helper()
	var total int64
	err := filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.Mode().IsRegular() {
			total += info.Size()
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	return total
}

func fileBytes(path string) int64 {
	if info, err := os.Stat(path); err == nil {
		return info.Size()
	}
	return 0
}

func durPercentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func intPercentile(sorted []int, p float64) int {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// brokerOver opens a fresh broker over the given shard count, the way to measure how
// latency scales with fan-out: a query against an n-shard broker fans out to at most n
// shards, so timing the same query over growing n traces the parallel fan-out cost.
func brokerOver(t testing.TB, dir string, ix *collection.Index, model *rank.Model, n int) (*search.Broker, []*search.Shard) {
	t.Helper()
	all := openInIndexOrder(t, dir, ix)
	shards := all[:n]
	for _, s := range all[n:] {
		_ = s.Close()
	}
	routing := search.BuildRoutingIndex(shards)
	stats := search.GlobalStats{
		DocCount:    ix.Stats.DocCount,
		TokenCount:  ix.Stats.TokenCount,
		AvgDocLen:   ix.Stats.AvgDocLen,
		AvgFieldLen: ix.Stats.AvgFieldLen,
	}
	cascade := rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
	return search.NewBrokerWith(shards, cascade, routing, stats), shards
}

func timeQueries(b *search.Broker, queries []string, reps int) []time.Duration {
	ctx := context.Background()
	for w := 0; w < 2; w++ { // warm the page cache and the per-query allocations
		for _, qt := range queries {
			b.Search(ctx, search.Query{Text: qt, K: 10})
		}
	}
	var samples []time.Duration
	for r := 0; r < reps; r++ {
		for _, qt := range queries {
			start := time.Now()
			b.Search(ctx, search.Query{Text: qt, K: 10})
			samples = append(samples, time.Since(start))
		}
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	return samples
}

// TestScaleCCrawl is the latency-and-scalability measurement the headline gate rests
// on. It builds the largest real Common Crawl collection the sample parquet yields,
// cut into many small shards so the fleet has real breadth, loads it through the
// manifest the serve scale path uses, and measures the numbers that decide whether the
// engine holds at 100k shards and two billion documents:
//
//   - the routing fan-out distribution over the whole vocabulary, the proof that the
//     long-tail terms that dominate real traffic prune to a few shards while only the
//     stopword head fans out wide;
//   - per-query latency for selective queries (routing-bounded) and head queries
//     (fan-out-bounded), on the real retrieve, merge, and rerank path;
//   - how latency grows with fan-out, the cost model the projection extrapolates;
//   - the per-shard and per-document resident heap and disk the fleet multiplies.
//
// It is not a Benchmark because it logs a derived table, not a single ns/op.
func TestScaleCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data scale build in short mode")
	}

	// A small shard size turns the sample corpus into many shards, so the routing index,
	// the manifest, and the fan-out are exercised at real breadth rather than the handful
	// of shards a large shard size would leave. The serve fast path is what scales, so
	// this is the partitioning the fleet runs.
	const shardSize = 256
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")

	buildStart := time.Now()
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: shardSize})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	buildDur := time.Since(buildStart)
	if res.Shards < 8 {
		t.Fatalf("need a multi-shard fleet to measure routing, got %d shards", res.Shards)
	}

	// Load the manifest, the serve startup the fleet pays once. Timing it is the proxy
	// for the startup cost that must stay flat as shards grow, since the manifest is one
	// file read whose size is the routing vocabulary, not the corpus.
	loadStart := time.Now()
	ix, err := collection.LoadIndex(out)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	loadDur := time.Since(loadStart)
	model := scaleModel(t)

	// --- resident heap, in two stages so the per-shard cost is separated from the
	// broker-side routing index (which scales per broker, not per shard) ---
	var m0, m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m0)
	shards := openInIndexOrder(t, out, ix)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()
	runtime.GC()
	runtime.ReadMemStats(&m1)
	routing := search.NewRoutingIndex(ix.RoutingMap(), ix.AlwaysRouted(), len(shards))
	stats := search.GlobalStats{
		DocCount:    ix.Stats.DocCount,
		TokenCount:  ix.Stats.TokenCount,
		AvgDocLen:   ix.Stats.AvgDocLen,
		AvgFieldLen: ix.Stats.AvgFieldLen,
	}
	broker := search.NewBrokerWith(shards, rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model), routing, stats)
	runtime.GC()
	runtime.ReadMemStats(&m2)
	shardHeap := int64(m1.HeapInuse) - int64(m0.HeapInuse)
	routingHeap := int64(m2.HeapInuse) - int64(m1.HeapInuse)
	if shardHeap < 0 {
		shardHeap = 0
	}
	if routingHeap < 0 {
		routingHeap = 0
	}

	// --- fan-out distribution over the whole routing vocabulary ---
	// This is the selectivity proof. Each term's fan-out is how many shards hold it; the
	// distribution over every term shows that the long tail routes to a few shards and
	// only the stopword head fans out wide. Routing prunes for the traffic that matters.
	rmap := ix.RoutingMap()
	fan := make([]int, 0, len(rmap))
	var singleShard, leqFive, fullFan int
	for _, lst := range rmap {
		f := len(lst)
		fan = append(fan, f)
		switch {
		case f == 1:
			singleShard++
		case f <= 5:
		}
		if f <= 5 {
			leqFive++
		}
		if f >= len(shards) {
			fullFan++
		}
	}
	sort.Ints(fan)
	vocab := len(fan)

	// A pool of selective terms, ones that route to a single shard, drawn from the real
	// vocabulary. These stand in for the long-tail queries that are most of real traffic.
	var selective []string
	for term, lst := range rmap {
		if len(lst) == 1 && len(term) >= 4 {
			selective = append(selective, term)
			if len(selective) >= 32 {
				break
			}
		}
	}
	sort.Strings(selective) // stable order, no Math.random in tests

	head := []string{
		"the", "time", "data", "page", "information", "new york", "world news",
		"how to", "best price", "contact us", "privacy policy", "search results",
	}

	// --- latency, head vs selective, full fleet ---
	const reps = 200
	headLat := timeQueries(broker, head, reps)
	selLat := timeQueries(broker, selective, reps)

	// Fan-out the head queries actually drive, so the head latency is read against its
	// real fan-out rather than assumed to be full.
	var headFan []int
	for _, qt := range head {
		headFan = append(headFan, len(routing.RouteTerms(lexical.Analyze(qt))))
	}
	sort.Ints(headFan)

	// --- latency as a function of fan-out: rebuild brokers over growing shard counts and
	// time a head query, tracing the parallel fan-out cost curve the projection rests on ---
	type point struct {
		shards int
		p50    time.Duration
	}
	var curve []point
	for _, n := range []int{1, 5, 10, 20, 40, len(shards)} {
		if n > len(shards) {
			continue
		}
		bn, sn := brokerOver(t, out, ix, model, n)
		lat := timeQueries(bn, []string{"the time data"}, 120)
		curve = append(curve, point{shards: len(sn), p50: durPercentile(lat, 50)})
		_ = bn.Close()
	}

	totalBytes := dirBytes(t, out)
	manifestBytes := fileBytes(filepath.Join(out, "index.tsm"))
	bytesPerDoc := float64(totalBytes) / float64(res.Docs)

	t.Logf("=== tsumugi scale measurement, real CC-MAIN-2026-25 ===")
	t.Logf("corpus:   %d docs, %d shards (%d docs/shard), %d hosts, built in %s",
		res.Docs, res.Shards, shardSize, res.Hosts, buildDur.Round(time.Millisecond))
	t.Logf("disk:     %.1f MB total, %.0f bytes/doc, manifest %.1f KB",
		float64(totalBytes)/1e6, bytesPerDoc, float64(manifestBytes)/1e3)
	t.Logf("startup:  manifest load %s; heap: shards %.1f MB (%.0f KB/shard), routing index %.1f MB",
		loadDur.Round(time.Microsecond), float64(shardHeap)/1e6,
		float64(shardHeap)/float64(len(shards))/1e3, float64(routingHeap)/1e6)
	t.Logf("routing selectivity over %d terms: %.1f%% touch 1 shard, %.1f%% touch <=5, %.1f%% touch all %d",
		vocab, 100*float64(singleShard)/float64(vocab), 100*float64(leqFive)/float64(vocab),
		100*float64(fullFan)/float64(vocab), len(shards))
	t.Logf("  fan-out per term: p50 %d, p90 %d, p99 %d, max %d shards",
		intPercentile(fan, 50), intPercentile(fan, 90), intPercentile(fan, 99), fan[len(fan)-1])
	t.Logf("head-query fan-out: p50 %d, max %d shards (of %d)", intPercentile(headFan, 50), headFan[len(headFan)-1], len(shards))
	t.Logf("latency selective (long-tail, routing-bounded): p50 %s, p99 %s over %d terms",
		durPercentile(selLat, 50).Round(time.Microsecond), durPercentile(selLat, 99).Round(time.Microsecond), len(selective))
	t.Logf("latency head (full fan-out %d shards): p50 %s, p90 %s, p99 %s",
		len(shards), durPercentile(headLat, 50).Round(time.Microsecond),
		durPercentile(headLat, 90).Round(time.Microsecond), durPercentile(headLat, 99).Round(time.Microsecond))
	t.Logf("latency vs fan-out (head query 'the time data'):")
	for _, p := range curve {
		t.Logf("  %3d shards -> p50 %s", p.shards, p.p50.Round(time.Microsecond))
	}

	// --- projection to the fleet target from the measured constants ---
	const targetDocs = 2_000_000_000
	const targetShards = 100_000
	projDisk := bytesPerDoc * float64(targetDocs)
	heapPerShard := float64(shardHeap) / float64(len(shards))
	// Per-shard marginal latency from the curve: the slope from 1 shard to full fan-out,
	// divided by the parallelism the broker runs the fan-out at. This is what a head query
	// pays per extra shard once the cores are saturated.
	var perShardNs float64
	if len(curve) >= 2 {
		lo, hi := curve[0], curve[len(curve)-1]
		perShardNs = float64(hi.p50-lo.p50) / float64(hi.shards-lo.shards)
	}
	t.Logf("projection 2B docs / 100k shards (%d docs/shard):", targetDocs/targetShards)
	t.Logf("  disk:  %.1f TB at %.0f bytes/doc", projDisk/1e12, bytesPerDoc)
	t.Logf("  heap:  %.0f KB/shard resident off the mmap; a box holding 1000 shards pays ~%.1f GB Go heap, the rest stays OS page cache",
		heapPerShard/1e3, heapPerShard*1000/1e9)
	t.Logf("  selective queries: fan-out is set by term selectivity, not shard count, so the long-tail p50 %s holds on a flat broker at 100k shards",
		durPercentile(selLat, 50).Round(time.Microsecond))
	t.Logf("  head queries: pay ~%.0f us per added shard at saturation, so a flat broker over 100k shards is not viable for stopword queries; they need a broker tree (fan-out to ~100 mid-tier brokers, each over ~1000 shards, in parallel), keeping each tier's fan-out near the measured full-fleet %s",
		perShardNs/1e3, durPercentile(headLat, 50).Round(time.Microsecond))

	// Guard the invariant the scale story rests on: routing must prune for the long tail.
	// The bulk of the vocabulary has to touch only a few shards, or the bounded-fan-out
	// argument fails. The stopword head touching all shards is expected and allowed.
	if frac := float64(leqFive) / float64(vocab); frac < 0.5 {
		t.Errorf("routing did not prune the long tail: only %.1f%% of terms touch <=5 shards", 100*frac)
	}
}
