package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/bootstrap"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/eval"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// TestBootstrapTrainsAndGatesCCrawl runs the M9 training bootstrap end to end on real
// crawl data: it builds a multi-shard collection, retrieves a candidate pool per known-
// item query through the same feature path serving scores on, grades every candidate with
// the deterministic UMBRELA judge over the documents' real text, fits a LambdaMART model
// on the graded labels, and checks it beats the untrained cold-start baseline on a held-
// out query split. The known item gives the gate an honest signal: a document retrieved
// against its own title is the relevant one, and a model that learns the query-dependent
// features ranks it above a static-prior baseline that cannot see the query, so the
// trained model's NDCG@10 gain is the proof the bootstrap produced a ranker worth its
// training, the milestone's gate measured on a real corpus.
func TestBootstrapTrainsAndGatesCCrawl(t *testing.T) {
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
		t.Fatalf("need at least 2 shards, got %d", res.Shards)
	}

	coldPath := filepath.Join(tmp, "cold.bin")
	writeModel(t, coldPath)
	broker, pl, err := openCollection(out, coldPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	// Known-item queries: a document's title retrieved against the engine, kept when it
	// parses to a few content terms and is unique, the same selection the known-item harness
	// uses. The relevant document is the one the title came from; the judge grades it high
	// against the query while burying the lexically-similar near-misses, which is the hard-
	// negative gradient the model trains on.
	passages := readPassages(t, out)
	const wantQueries = 60
	const minTerms = 3
	queries := map[string]string{}
	seen := map[string]struct{}{}
	for id := uint32(0); id < uint32(res.Docs) && len(queries) < wantQueries; id++ {
		title := passages[id].Title
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
		if len(toQuery(pq, 100).Terms) < minTerms {
			continue
		}
		seen[title] = struct{}{}
		queries[strconv.FormatUint(uint64(id), 10)] = title
	}
	if len(queries) < 20 {
		t.Fatalf("only %d known-item queries built; need a richer corpus", len(queries))
	}

	passage, closeP, err := forwardPassages(out)
	if err != nil {
		t.Fatalf("forwardPassages: %v", err)
	}
	defer closeP()

	ctx := context.Background()
	qf := bootstrap.QueryFunc(func(raw string) search.Query { return toQuery(pl.parse(raw), 200) })
	pool, err := bootstrap.Build(ctx, broker, eval.LexicalJudge{}, qf, queries, passage)
	if err != nil {
		t.Fatalf("bootstrap.Build: %v", err)
	}
	if len(pool.Queries) < 20 {
		t.Fatalf("only %d queries produced a judged pool; need more", len(pool.Queries))
	}
	// The pool must carry the full L2 row, the matrix features plus the online query-
	// dependent features the model needs to learn from, not just the static matrix columns.
	if pool.NumFeatures <= len(feature.DefaultSchema()) {
		t.Fatalf("pool row width %d does not exceed the matrix schema width %d, online features are missing",
			pool.NumFeatures, len(feature.DefaultSchema()))
	}

	train, evalSet := bootstrap.Split(pool.Queries, 0.3)
	if len(train) == 0 || len(evalSet) == 0 {
		t.Fatalf("degenerate split: train=%d eval=%d", len(train), len(evalSet))
	}
	untrained, err := loadModel(coldPath)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}
	params := rank.DefaultParams()
	params.Rounds = 150
	outcome, ens, err := bootstrap.Gate(train, evalSet, pool.NumFeatures, params, untrained, []int{10})
	if err != nil {
		t.Fatalf("bootstrap.Gate: %v", err)
	}
	judged := 0
	for _, q := range pool.Queries {
		judged += len(q.Docs)
	}
	t.Logf("bootstrap over %d queries (%d judged candidates, %d features/row), %d shards: trained NDCG@10 %.4f over untrained %.4f (gain %.4f), %d trees",
		len(pool.Queries), judged, pool.NumFeatures, res.Shards,
		outcome.Trained.MeanNDCG[10], outcome.Untrained.MeanNDCG[10], outcome.GainNDCG10, ens.NumTrees())
	if !outcome.Improved {
		t.Fatalf("trained NDCG@10 %.4f did not beat the untrained baseline %.4f (gain %.4f)",
			outcome.Trained.MeanNDCG[10], outcome.Untrained.MeanNDCG[10], outcome.GainNDCG10)
	}

	// The trained model the gate produced must drop into the serving path: stamped with the
	// matrix schema, it loads and the broker accepts it, which is what makes the bootstrap's
	// output a model the serve command can rank with.
	trainedPath := filepath.Join(tmp, "trained.bin")
	f, err := os.Create(trainedPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := ens.Save(f); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	tb, tpl, err := openCollection(out, trainedPath)
	if err != nil {
		t.Fatalf("the trained model is not serve-loadable: %v", err)
	}
	_ = tpl
	_ = tb.Close()
}

// TestRunBootstrapCCrawl drives the full train command bootstrap path: it writes a query
// file, runs runBootstrap with the deterministic judge, and checks it reports the gate and
// writes a model the serve path loads. It is the CLI smoke over the same real data the
// gate test exercises directly, so the command wiring is proven, not just the package.
func TestRunBootstrapCCrawl(t *testing.T) {
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

	coldPath := filepath.Join(tmp, "cold.bin")
	writeModel(t, coldPath)

	// Build a query file from a handful of known-item titles.
	_, pl, err := openCollection(out, coldPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	passages := readPassages(t, out)
	var lines []string
	seen := map[string]struct{}{}
	for id := uint32(0); id < uint32(res.Docs) && len(lines) < 50; id++ {
		title := strings.TrimSpace(passages[id].Title)
		if title == "" {
			continue
		}
		if _, dup := seen[title]; dup {
			continue
		}
		pq := pl.parse(title)
		if pq.Empty() || len(toQuery(pq, 100).Terms) < 3 {
			continue
		}
		seen[title] = struct{}{}
		// One query per line, title only (auto-id by line), tabs flattened to spaces.
		lines = append(lines, strings.ReplaceAll(title, "\t", " "))
	}
	if len(lines) < 20 {
		t.Fatalf("only %d queries built for the CLI smoke", len(lines))
	}
	qfile := filepath.Join(tmp, "queries.txt")
	if err := os.WriteFile(qfile, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	trainedPath := filepath.Join(tmp, "trained.bin")
	var buf bytes.Buffer
	if err := runBootstrap(&buf, out, coldPath, qfile, trainedPath, 0.3, 150, 200); err != nil {
		t.Fatalf("runBootstrap: %v\noutput:\n%s", err, buf.String())
	}
	t.Logf("runBootstrap output:\n%s", buf.String())
	if !strings.Contains(buf.String(), "over untrained baseline") {
		t.Fatalf("output did not report the gate:\n%s", buf.String())
	}
	if _, err := os.Stat(trainedPath); err != nil {
		t.Fatalf("trained model not written: %v", err)
	}
	tb, _, err := openCollection(out, trainedPath)
	if err != nil {
		t.Fatalf("the written model is not serve-loadable: %v", err)
	}
	_ = tb.Close()
}
