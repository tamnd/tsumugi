package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	tsumugi "github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/eval"
	"github.com/tamnd/tsumugi/forward"
)

// readTitles enumerates a built collection's forward stores and returns each
// document's title keyed by its global id, the shard's node base plus the row. It is
// the readback the known-item evaluation builds its queries and judgments from: the
// title a document carries is the query a searcher who remembers the page would type,
// and that document's global id is the one relevant answer.
func readTitles(t *testing.T, dir string) map[uint32]string {
	t.Helper()
	infos, err := collection.List(dir)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	titles := map[uint32]string{}
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			t.Fatalf("open %s: %v", filepath.Base(info.Path), err)
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			t.Fatalf("forward region %s: %v", filepath.Base(info.Path), err)
		}
		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			t.Fatalf("open forward %s: %v", filepath.Base(info.Path), err)
		}
		for id := uint32(0); id < fwd.DocCount(); id++ {
			raw, _ := fwd.Column("title", id)
			titles[info.NodeBase+id] = string(raw)
		}
		fwd.Close()
		_ = r.Close()
	}
	return titles
}

// TestEvalKnownItemCCrawl is the quality harness exercised end to end on real data: it
// builds a real multi-shard collection, recovers each document's title, and runs a
// known-item evaluation. For a sample of documents it types the document's own title as
// the query, takes the engine's ranked top-k as a TREC run, and judges the single
// document the title came from as the one relevant answer. The harness then reports
// NDCG, MRR, and Recall over those queries the same way pytrec_eval would. A known item
// retrieved against its own title is the cleanest signal that the ranking is doing real
// work: with thousands of documents a random ranker's reciprocal rank is vanishingly
// small, so a mean reciprocal rank well above zero is the engine putting the remembered
// page near the top. It also round-trips the run through the TREC file format and
// re-scores it, the proof the committed-run reproducibility path reports the same
// numbers the in-memory run does over real text.
func TestEvalKnownItemCCrawl(t *testing.T) {
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
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards to exercise the fan-out, got %d", res.Shards)
	}

	modelPath := filepath.Join(tmp, "model.bin")
	writeModel(t, modelPath)
	broker, pl, err := openCollection(out, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	titles := readTitles(t, out)

	// Select known-item queries from distinctive titles. A title is usable when it
	// parses to a few content terms (a one-word or all-stopword title is not a query a
	// searcher would type to find a specific page), and a title string is taken at most
	// once so a boilerplate title shared by many pages does not flood the sample with
	// queries that have no single right answer. A bounded stride across the global id
	// space keeps the sample spread over every shard while holding the test fast.
	const wantQueries = 60
	const minTerms = 3
	run := eval.Run{}
	qrels := eval.Qrels{}
	seen := map[string]struct{}{}
	ctx := context.Background()
	for id := uint32(0); id < uint32(res.Docs) && len(run) < wantQueries; id++ {
		title := titles[id]
		if title == "" {
			continue
		}
		if _, dup := seen[title]; dup {
			continue
		}
		pq := pl.parse(title)
		if pq.Empty() {
			continue
		}
		q := toQuery(pq, 100)
		if len(q.Terms) < minTerms {
			continue
		}
		seen[title] = struct{}{}
		sc := broker.SearchComplete(ctx, q)
		qid := strconv.FormatUint(uint64(id), 10)
		docs := make([]eval.RankedDoc, len(sc.Hits))
		for i, h := range sc.Hits {
			docs[i] = eval.RankedDoc{Doc: strconv.FormatUint(uint64(h.DocID), 10), Score: h.Score}
		}
		run[qid] = docs
		// The one relevant answer is the document the title came from, judged grade one.
		qrels[qid] = map[string]float64{qid: 1}
	}
	if len(run) < 20 {
		t.Fatalf("only %d known-item queries built from %d docs; need a richer corpus to measure", len(run), res.Docs)
	}

	rep := eval.Evaluate(run, qrels, []int{10, 20}, []int{100})
	if rep.NumQueries != len(run) {
		t.Fatalf("scored %d of %d queries, want all (every known-item query has one relevant doc)", rep.NumQueries, len(run))
	}
	t.Logf("known-item over %d real queries, %d shards, %d docs: NDCG@10 %.4f NDCG@20 %.4f MRR %.4f Recall@100 %.4f",
		rep.NumQueries, res.Shards, res.Docs, rep.MeanNDCG[10], rep.MeanNDCG[20], rep.MeanMRR, rep.MeanRecall[100])

	// The bar is set against an explicit random-ranker baseline rather than a tuned
	// constant, so the gate stays meaningful whatever model the test loads. A ranker that
	// returned k documents uniformly at random over a corpus of res.Docs would put the one
	// relevant document in the top 100 with probability 100/res.Docs and, when it did,
	// at an expected rank near the middle, so its mean reciprocal rank is on the order of
	// (100/res.Docs) * (1/50). The engine must clear that chance level by a wide margin:
	// finding the remembered page is the whole point of retrieval.
	chanceRecall := 100.0 / float64(res.Docs)
	chanceMRR := chanceRecall / 50.0
	if rep.MeanRecall[100] < 10*chanceRecall {
		t.Fatalf("mean recall@100 %.4f is within 10x of the random baseline %.4f; the known item is not being retrieved", rep.MeanRecall[100], chanceRecall)
	}
	if rep.MeanMRR < 20*chanceMRR {
		t.Fatalf("mean reciprocal rank %.4f is within 20x of the random baseline %.5f; the ranker is not ordering the known item", rep.MeanMRR, chanceMRR)
	}

	// Round-trip the run through the TREC six-column format and re-score it: a committed
	// run file read back must report the same means, the reproducibility property doc 14
	// rests the quality gate on. WriteRun orders by score and ParseRun reads the score
	// back, so the re-scored report is identical to the in-memory one over real text.
	var buf bytes.Buffer
	if err := eval.WriteRun(&buf, run, "tsumugi"); err != nil {
		t.Fatalf("write run: %v", err)
	}
	back, err := eval.ParseRun(&buf)
	if err != nil {
		t.Fatalf("parse run: %v", err)
	}
	rep2 := eval.Evaluate(back, qrels, []int{10, 20}, []int{100})
	if rep2.NumQueries != rep.NumQueries {
		t.Fatalf("round-trip scored %d queries, in-memory %d", rep2.NumQueries, rep.NumQueries)
	}
	approxClose := func(name string, a, b float64) {
		if d := a - b; d > 1e-9 || d < -1e-9 {
			t.Fatalf("round-trip %s = %.9f, in-memory %.9f", name, b, a)
		}
	}
	approxClose("mean ndcg@10", rep.MeanNDCG[10], rep2.MeanNDCG[10])
	approxClose("mean ndcg@20", rep.MeanNDCG[20], rep2.MeanNDCG[20])
	approxClose("mean mrr", rep.MeanMRR, rep2.MeanMRR)
	approxClose("mean recall@100", rep.MeanRecall[100], rep2.MeanRecall[100])
}
