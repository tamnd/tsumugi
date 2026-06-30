package cli

import (
	"bufio"
	"context"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/bootstrap"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/eval"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// runBootstrap runs the doc 09 training bootstrap end to end: it opens the collection,
// retrieves a candidate pool per training query through the broker's feature path, grades
// every candidate with the UMBRELA judge, fits a LambdaMART model on the graded labels,
// and gates the trained model's NDCG@10 against the untrained cold-start baseline on a
// held-out query split, writing the model only when it clears the gate. It is the M9
// deliverable: the no-clicks bootstrap that turns BM25 candidates and an LLM judge into a
// trained ranker, with the gold-set cross-check the caller runs through eval.
func runBootstrap(out io.Writer, dir, modelPath, queriesPath, outPath string, evalFrac float64, rounds, poolK int) error {
	queries, err := readQueries(queriesPath)
	if err != nil {
		return err
	}
	if len(queries) == 0 {
		return fmt.Errorf("query file %s has no queries", queriesPath)
	}

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		return err
	}
	defer func() { _ = broker.Close() }()

	untrained, err := loadModel(modelPath)
	if err != nil {
		return err
	}

	passage, closePassages, err := forwardPassages(dir)
	if err != nil {
		return err
	}
	defer closePassages()

	judge, judgeName := selectJudge(out)
	qf := bootstrap.QueryFunc(func(raw string) search.Query { return toQuery(pl.parse(raw), poolK) })

	_, _ = fmt.Fprintf(out, "bootstrapping over %d queries with the %s judge\n", len(queries), judgeName)
	pool, err := bootstrap.Build(context.Background(), broker, judge, qf, queries, passage)
	if err != nil {
		return err
	}
	if len(pool.Queries) == 0 {
		return fmt.Errorf("no candidates judged: the queries returned no documents with text to grade")
	}
	judged := 0
	for _, q := range pool.Queries {
		judged += len(q.Docs)
	}
	_, _ = fmt.Fprintf(out, "judged %d candidates across %d queries, %d features per row\n",
		judged, len(pool.Queries), pool.NumFeatures)

	train, evalSet := bootstrap.Split(pool.Queries, evalFrac)
	_, _ = fmt.Fprintf(out, "split into %d training and %d held-out evaluation queries\n", len(train), len(evalSet))

	params := rank.DefaultParams()
	params.Rounds = rounds
	outcome, ens, err := bootstrap.Gate(train, evalSet, pool.NumFeatures, params, untrained, []int{10})
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "trained %d trees: NDCG@10 %.4f over untrained baseline %.4f (gain %.4f)\n",
		ens.NumTrees(), outcome.Trained.MeanNDCG[10], outcome.Untrained.MeanNDCG[10], outcome.GainNDCG10)
	if !outcome.Improved {
		return fmt.Errorf("trained model did not improve NDCG@10 over the untrained baseline (gain %.4f), not writing %s",
			outcome.GainNDCG10, outPath)
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	if err := ens.Save(f); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "wrote %s\n", outPath)
	return nil
}

// selectJudge picks the UMBRELA judge: the env-configured LLM judge when TSUMUGI_JUDGE_URL
// is set, otherwise the deterministic lexical judge. The lexical judge is a real term-
// coverage grader, not a stub, so a run with no LLM configured still produces graded
// labels and a trained model; the LLM judge is the production path the spec specifies.
func selectJudge(out io.Writer) (eval.Judge, string) {
	if j, ok := eval.NewLLMJudgeFromEnv(); ok {
		return j, "LLM"
	}
	_, _ = fmt.Fprintln(out, "no TSUMUGI_JUDGE_URL set, grading with the deterministic lexical judge")
	return eval.LexicalJudge{}, "lexical"
}

// readQueries reads the training query file: one query per line, an optional id and a tab
// before the text, otherwise the line is the text and the id is its line position. Blank
// lines and lines beginning with a hash are skipped, so a query file can carry comments.
func readQueries(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	queries := map[string]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		id := fmt.Sprintf("q%04d", line)
		if tab := strings.IndexByte(text, '\t'); tab >= 0 {
			id = strings.TrimSpace(text[:tab])
			text = strings.TrimSpace(text[tab+1:])
		}
		if text != "" {
			queries[id] = text
		}
	}
	return queries, sc.Err()
}

// forwardPassages opens every shard's forward store and returns a lookup from a global
// document id to its passage, the query, title, and body text the judge reads, along with
// a closer the caller defers. The passage's stable key is the document's durable doc_id
// (the SHA256 of its canonical URL, hex-encoded) so the run and qrels are keyed by the
// cross-build identity the spec pins, falling back to the global id only when a shard
// carries no doc_id column. The lookup binary-searches the shards by their node base, so
// resolving a candidate is a log-shard-count step over the open regions, not a rescan.
func forwardPassages(dir string) (bootstrap.PassageFunc, func(), error) {
	infos, err := collection.List(dir)
	if err != nil {
		return nil, nil, err
	}
	type shardFwd struct {
		base  uint32
		count uint32
		fwd   *forward.Region
		rdr   *tsumugi.Reader
	}
	var shards []shardFwd
	closeAll := func() {
		for _, s := range shards {
			s.fwd.Close()
			_ = s.rdr.Close()
		}
	}
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			closeAll()
			return nil, nil, err
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			closeAll()
			return nil, nil, err
		}
		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			closeAll()
			return nil, nil, err
		}
		shards = append(shards, shardFwd{base: info.NodeBase, count: fwd.DocCount(), fwd: fwd, rdr: r})
	}
	sort.Slice(shards, func(i, j int) bool { return shards[i].base < shards[j].base })

	lookup := func(id uint32) (eval.Passage, bool) {
		i := sort.Search(len(shards), func(i int) bool { return shards[i].base+shards[i].count > id })
		if i >= len(shards) || id < shards[i].base {
			return eval.Passage{}, false
		}
		local := id - shards[i].base
		fwd := shards[i].fwd
		title, _ := fwd.Column("title", local)
		body, _ := fwd.Column("body", local)
		key := strconv.FormatUint(uint64(id), 10)
		if raw, ok := fwd.Column("doc_id", local); ok && len(raw) > 0 {
			key = hex.EncodeToString(raw)
		}
		return eval.Passage{Doc: key, Title: string(title), Body: string(body)}, true
	}
	return lookup, closeAll, nil
}
