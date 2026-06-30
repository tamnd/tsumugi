package search

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
)

// TestParallelL2MatchesSerial is the parity gate for the broker's parallel L2 feature
// extraction: the rerank fans the per-survivor online feature decode across workers
// between the L1 cut and the model scoring, and this proves that fan-out changes
// nothing the caller observes. It builds one broker over text-bearing shards, then
// runs the same queries twice, once with the extraction forced serial (one worker)
// and once at the broker's real worker count, and asserts the top-k is byte-for-byte
// identical: same documents, same order, same score bits. The online extractor holds
// per-query scratch it reuses across the survivors one worker scores, so the proof
// that a per-worker extractor yields the same row a single shared one would is what
// this test pins, over a real multi-shard fan-out rather than the unit extractor.
func TestParallelL2MatchesSerial(t *testing.T) {
	b, cleanup := buildTextBroker(t)
	defer cleanup()

	queries := []Query{
		{Text: "common document", K: 20},
		{Text: "common", K: 50},
		{Text: "document text body", K: 30},
		{Text: "common document", K: 5},
	}
	for _, q := range queries {
		t.Run(q.Text, func(t *testing.T) {
			b.maxConcurrency = 1
			serial := b.Search(context.Background(), q)
			b.maxConcurrency = 8
			parallel := b.Search(context.Background(), q)
			assertHitsIdentical(t, serial, parallel)
		})
	}
}

// TestParallelL2WorkerCountInvariant pins that the result is invariant to the worker
// count, not just serial-vs-some-parallel: the same query at one, two, three, and many
// workers returns the identical top-k. A partition bug that split survivors unevenly or
// raced on a shared extractor would show as a worker-count-dependent result, which this
// sweep would catch where a single serial-vs-parallel comparison might not.
func TestParallelL2WorkerCountInvariant(t *testing.T) {
	b, cleanup := buildTextBroker(t)
	defer cleanup()

	q := Query{Text: "common document text", K: 40}
	b.maxConcurrency = 1
	ref := b.Search(context.Background(), q)
	for _, workers := range []int{2, 3, 4, 7, 16, 64} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			b.maxConcurrency = workers
			got := b.Search(context.Background(), q)
			assertHitsIdentical(t, ref, got)
		})
	}
}

// TestParallelL2CCrawl is the real-data parity gate: the serial and parallel L2
// extraction agree bit for bit over the real, messy, multi-language crawl text, where
// document lengths and term distributions vary far more than the synthetic corpus.
func TestParallelL2CCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl sample not present: %v", err)
	}
	docs := readRealDocs(t, 1200)
	if len(docs) < 100 {
		t.Skipf("too few real docs: %d", len(docs))
	}

	dir := t.TempDir()
	model := trainExactMatchModel(t)
	const parts = 5
	size := (len(docs) + parts - 1) / parts
	shards := make([]*Shard, 0, parts)
	for p := 0; p < parts; p++ {
		lo := p * size
		if lo >= len(docs) {
			break
		}
		hi := lo + size
		if hi > len(docs) {
			hi = len(docs)
		}
		path := filepath.Join(dir, fmt.Sprintf("real%d.tsumugi", p))
		buildTextShardBase(t, path, docs[lo:hi], uint32(lo))
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards = append(shards, sh)
	}
	b := NewBroker(shards, newTestCascade(model))
	defer func() { _ = b.Close() }()

	// Drive queries from real document titles so the survivors carry real text into the
	// L2 decode, the path the parallel extraction speeds up.
	var queries []Query
	for i := 0; i < len(docs) && len(queries) < 16; i++ {
		toks := uniqueTokens(docs[i].title)
		if len(toks) >= 2 && len(toks) <= 6 {
			queries = append(queries, Query{Text: docs[i].title, K: 25})
		}
	}
	if len(queries) == 0 {
		t.Skip("no usable real-title queries in the sample")
	}
	for i, q := range queries {
		t.Run(fmt.Sprintf("q%d", i), func(t *testing.T) {
			b.maxConcurrency = 1
			serial := b.Search(context.Background(), q)
			b.maxConcurrency = 8
			parallel := b.Search(context.Background(), q)
			assertHitsIdentical(t, serial, parallel)
		})
	}
	t.Logf("checked %d real-title queries over %d shards", len(queries), len(shards))
}

// BenchmarkBrokerL2Serial and BenchmarkBrokerL2Parallel bracket the slice's win: the
// same fleet query reranked with the L2 feature extraction forced serial versus fanned
// across the box's cores. The gap is the wall time the parallel extraction reclaims
// from the stage the per-stage breakdown found dominates the cascade.
func BenchmarkBrokerL2Serial(b *testing.B) {
	benchBrokerL2(b, 1)
}

func BenchmarkBrokerL2Parallel(b *testing.B) {
	benchBrokerL2(b, 0) // 0 keeps the broker's real GOMAXPROCS worker count
}

func benchBrokerL2(b *testing.B, workers int) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl sample not present: %v", err)
	}
	docs := readRealDocs(b, 2000)
	if len(docs) < 200 {
		b.Skipf("too few real docs: %d", len(docs))
	}
	dir := b.TempDir()
	model := trainExactMatchModel(b)
	const parts = 5
	size := (len(docs) + parts - 1) / parts
	shards := make([]*Shard, 0, parts)
	for p := 0; p < parts; p++ {
		lo := p * size
		if lo >= len(docs) {
			break
		}
		hi := lo + size
		if hi > len(docs) {
			hi = len(docs)
		}
		path := filepath.Join(dir, fmt.Sprintf("bench%d.tsumugi", p))
		buildTextShardBase(b, path, docs[lo:hi], uint32(lo))
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			b.Fatalf("open shard %d: %v", p, err)
		}
		shards = append(shards, sh)
	}
	br := NewBroker(shards, newTestCascade(model))
	defer func() { _ = br.Close() }()
	if workers > 0 {
		br.maxConcurrency = workers
	}
	q := Query{Text: docs[0].title, K: 20}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = br.Search(context.Background(), q)
	}
}

// buildTextBroker builds a small multi-shard broker over the synthetic text corpus,
// the fixture the parity tests share. Every shard carries the body text the online L2
// features decode, so the parallel extraction the broker fans is actually exercised.
func buildTextBroker(t testing.TB) (*Broker, func()) {
	t.Helper()
	const n, parts = 240, 4
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
	return b, func() { _ = b.Close() }
}

// buildTextShardBase writes a text shard at a given node base, the multi-shard variant
// of buildTextShard with a global docID offset so a broker over several of them sees a
// contiguous docID space. It mirrors buildTextShard's regions (a lexical index over the
// three fields, an empty feature matrix, and the url/title/body forward store) and only
// adds the node base the broker stitches the shards into one docID space with.
func buildTextShardBase(t testing.TB, path string, docs []textDoc, nodeBase uint32) {
	t.Helper()
	lb := lexical.NewBuilder(lexical.DefaultParams())
	fb := feature.NewBuilder(feature.DefaultSchema(), feature.SchemaVersion)
	fwdb := forward.NewBuilder([]forward.Column{
		{Name: "url", Type: forward.ColString},
		{Name: "title", Type: forward.ColString},
		{Name: "body", Type: forward.ColString, Flags: forward.FlagBlob},
	})
	var tokens float64
	for i, d := range docs {
		id := uint32(i)
		lb.AddDoc(id, map[lexical.Field]string{
			lexical.FieldTitle: d.title,
			lexical.FieldBody:  d.body,
			lexical.FieldURL:   d.url,
		})
		tokens += float64(len(lexical.Analyze(d.body)))
		fwdb.Set(id, "url", []byte(d.url))
		fwdb.Set(id, "title", []byte(d.title))
		fwdb.Set(id, "body", []byte(d.body))
	}
	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	w.SetDocCount(uint32(len(docs)))
	w.SetNodeBase(uint64(nodeBase))
	w.SetStat(tsumugi.StatTokenCount, tokens)
	if err := w.AddRegion(tsumugi.RegionLexical, tsumugi.CodecZstd, 0, 0, lb.Build()); err != nil {
		t.Fatalf("add lexical: %v", err)
	}
	if err := w.AddRegion(tsumugi.RegionFeature, tsumugi.CodecZstd, 0, 0, fb.Build()); err != nil {
		t.Fatalf("add feature: %v", err)
	}
	if err := w.AddRegion(tsumugi.RegionForward, tsumugi.CodecZstd, 0, 0, fwdb.Build()); err != nil {
		t.Fatalf("add forward: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// assertHitsIdentical fails unless the two hit lists are byte-for-byte equal: same
// length, same docIDs in the same order, same score bits.
func assertHitsIdentical(t *testing.T, want, got []Hit) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("hit count: serial %d, parallel %d", len(want), len(got))
	}
	for i := range want {
		if got[i].DocID != want[i].DocID {
			t.Fatalf("rank %d: serial doc %d, parallel doc %d", i, want[i].DocID, got[i].DocID)
		}
		if math.Float64bits(got[i].Score) != math.Float64bits(want[i].Score) {
			t.Fatalf("rank %d doc %d: serial score %v, parallel %v", i, want[i].DocID, want[i].Score, got[i].Score)
		}
	}
}
