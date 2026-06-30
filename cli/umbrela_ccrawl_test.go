package cli

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unicode"

	tsumugi "github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/eval"
	"github.com/tamnd/tsumugi/forward"
)

// readPassages enumerates a built collection's forward stores and returns each document's
// passage, its title and body, keyed by global id. It is the text the UMBRELA judge reads
// to grade a pooled document: the same title and body a reader would see, so a label is a
// judgment of the page and not of an identifier.
func readPassages(t *testing.T, dir string) map[uint32]eval.Passage {
	t.Helper()
	infos, err := collection.List(dir)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	out := map[uint32]eval.Passage{}
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			t.Fatalf("open %s: %v", filepath.Base(info.Path), err)
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			t.Fatalf("forward region: %v", err)
		}
		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			t.Fatalf("open forward: %v", err)
		}
		for id := uint32(0); id < fwd.DocCount(); id++ {
			title, _ := fwd.Column("title", id)
			body, _ := fwd.Column("body", id)
			gid := info.NodeBase + id
			out[gid] = eval.Passage{
				Doc:   strconv.FormatUint(uint64(gid), 10),
				Title: string(title),
				Body:  string(body),
			}
		}
		fwd.Close()
		_ = r.Close()
	}
	return out
}

// TestUmbrelaLadderCCrawl exercises the full UMBRELA quality pipeline end to end on real
// data: it builds a real multi-shard collection, runs known-item queries through the
// engine, pools the engine's ranking against a deliberately weakened one, judges the pool
// over the documents' real titles and bodies with the deterministic graded judge, runs the
// NDCG ladder over the two configurations against those shared judged labels, and checks
// the judge against a human-style gold subset by Kendall tau and the confusion matrix. The
// known item is the gold signal a human trusts: a document retrieved against its own title
// is perfectly relevant, and an unrelated document is irrelevant, so the graded judge
// reproducing those two extremes from the documents' text is the agreement the spec gates
// the LLM labels on, measured here on a real corpus with the offline judge standing in for
// the model the live path would call.
func TestUmbrelaLadderCCrawl(t *testing.T) {
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

	modelPath := filepath.Join(tmp, "model.bin")
	writeModel(t, modelPath)
	broker, pl, err := openCollection(out, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	passages := readPassages(t, out)
	ctx := context.Background()

	// Build the known-item query set and two systems to pool. The engine system is the
	// broker's ranking; the weak system is the same retrieved documents in reversed order, a
	// clearly worse ranking that gives the ladder a step to measure and the pool a second
	// contributor. A title is usable as a known-item query when it parses to a few content
	// terms and is taken at most once, the same selection the known-item harness uses.
	const wantQueries = 60
	const minTerms = 3
	engineRun := eval.Run{}
	weakRun := eval.Run{}
	queries := map[string]string{}
	relDoc := map[string]string{}
	seen := map[string]struct{}{}
	for id := uint32(0); id < uint32(res.Docs) && len(engineRun) < wantQueries; id++ {
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
		q := toQuery(pq, 100)
		if len(q.Terms) < minTerms {
			continue
		}
		sc := broker.SearchComplete(ctx, q)
		if len(sc.Hits) == 0 {
			continue
		}
		seen[title] = struct{}{}
		qid := strconv.FormatUint(uint64(id), 10)
		docs := make([]eval.RankedDoc, len(sc.Hits))
		weak := make([]eval.RankedDoc, len(sc.Hits))
		n := len(sc.Hits)
		for i, h := range sc.Hits {
			doc := strconv.FormatUint(uint64(h.DocID), 10)
			docs[i] = eval.RankedDoc{Doc: doc, Score: h.Score}
			// Reverse the order by handing the weak system a score that grows toward the tail,
			// so a document the engine ranked first the weak system ranks last.
			weak[i] = eval.RankedDoc{Doc: doc, Score: float64(i - n)}
		}
		engineRun[qid] = docs
		weakRun[qid] = weak
		queries[qid] = title
		relDoc[qid] = qid // the document the title came from is the known item
	}
	if len(engineRun) < 20 {
		t.Fatalf("only %d known-item queries built; need a richer corpus", len(engineRun))
	}

	// Pool both systems to a depth that covers the page cutoffs with margin, then judge the
	// pool once over the documents' real text. The qrels this produces are reused for every
	// configuration, which is what makes the ladder's row-to-row comparison valid.
	pool := eval.Pool([]eval.Run{engineRun, weakRun}, 20)
	passage := func(doc string) (eval.Passage, bool) {
		gid, err := strconv.ParseUint(doc, 10, 32)
		if err != nil {
			return eval.Passage{}, false
		}
		p, ok := passages[uint32(gid)]
		return p, ok
	}
	bulk, err := eval.JudgePool(ctx, eval.LexicalJudge{}, queries, pool, passage)
	if err != nil {
		t.Fatalf("JudgePool: %v", err)
	}
	if len(bulk) == 0 {
		t.Fatal("the judged pool is empty; nothing to score the ladder against")
	}

	// The ladder: the weak system below, the engine above, scored against the shared judged
	// labels. The engine ranks the perfectly-relevant known item near the top while the weak
	// system buries it, so the engine must show a positive NDCG@10 gain, the signature of a
	// step that earns its place.
	rungs := eval.Ladder([]eval.Config{
		{Name: "reversed", Run: weakRun},
		{Name: "engine", Run: engineRun},
	}, bulk, nil, []int{10, 20}, []int{100})
	t.Logf("ladder over %d real queries, %d shards: reversed NDCG@10 %.4f, engine NDCG@10 %.4f (gain %.4f, p %.4f)",
		rungs[0].Bulk.NumQueries, res.Shards, rungs[0].Bulk.MeanNDCG[10], rungs[1].Bulk.MeanNDCG[10],
		rungs[1].GainNDCG10, rungs[1].PValue)
	if rungs[1].GainNDCG10 <= 0 {
		t.Fatalf("engine NDCG@10 gain over the reversed baseline = %.4f, want positive", rungs[1].GainNDCG10)
	}
	if rungs[1].Bulk.MeanNDCG[10] <= rungs[0].Bulk.MeanNDCG[10] {
		t.Fatalf("engine NDCG@10 %.4f did not beat reversed %.4f over the judged pool",
			rungs[1].Bulk.MeanNDCG[10], rungs[0].Bulk.MeanNDCG[10])
	}

	// The gold-set agreement check on real passages. The human-style gold pairs each known
	// item against its own title as perfectly relevant and an unrelated document as
	// irrelevant, the two trusted extremes. The judge is run over those exact pairs and
	// compared to the human labels by Kendall tau and the confusion matrix. Because the
	// distractor is chosen to share no term with the query, the judge has a clean signal and
	// should reproduce the human's two grades, clearing the UMBRELA trust gate on real text.
	gold := eval.Qrels{}
	judged := eval.Qrels{}
	ids := sortedQids(engineRun)
	for i, qid := range ids {
		own := relDoc[qid]
		distractor := pickDisjointDistractor(queries[qid], ids, relDoc, passages, i)
		if distractor == "" {
			continue
		}
		gold[qid] = map[string]float64{own: 3, distractor: 0}
		judged[qid] = map[string]float64{}
		for doc := range gold[qid] {
			p, _ := passage(doc)
			g, gerr := eval.LexicalJudge{}.Grade(ctx, queries[qid], p)
			if gerr != nil {
				t.Fatalf("judge gold pair: %v", gerr)
			}
			judged[qid][doc] = float64(g)
		}
	}
	ag := eval.CompareToGold(judged, gold)
	t.Logf("gold-set agreement over %d real pairs: Kendall tau %.4f, confusion %v", ag.N, ag.KendallTau, ag.Confusion)
	if ag.N < 20 {
		t.Fatalf("only %d gold pairs compared; need more to measure agreement", ag.N)
	}
	if !ag.Passes() {
		t.Fatalf("the judge failed the UMBRELA trust gate on real passages: tau %.4f < %.2f", ag.KendallTau, eval.UmbrelaTauGate)
	}
}

// sortedQids returns a run's query ids in deterministic order, so the gold construction
// walks the queries the same way every run.
func sortedQids(run eval.Run) []string {
	ids := make([]string, 0, len(run))
	for q := range run {
		ids = append(ids, q)
	}
	// Numeric ids: sort by integer value for a stable, readable order.
	sortStringsAsInts(ids)
	return ids
}

// pickDisjointDistractor finds another known item whose passage shares no term with the
// query, the clean irrelevant document the gold set needs so the judge has an unambiguous
// grade-zero pair. It scans from just after the current index and wraps, returning the
// empty string when the corpus is too uniform to find one, in which case the caller skips
// that gold pair rather than judging an ambiguous distractor.
func pickDisjointDistractor(query string, ids []string, relDoc map[string]string, passages map[uint32]eval.Passage, self int) string {
	qterms := termsOf(query)
	if len(qterms) == 0 {
		return ""
	}
	for off := 1; off < len(ids); off++ {
		j := (self + off) % len(ids)
		doc := relDoc[ids[j]]
		gid, err := strconv.ParseUint(doc, 10, 32)
		if err != nil {
			continue
		}
		p := passages[uint32(gid)]
		pt := termsOf(p.Title + " " + p.Body)
		disjoint := true
		for term := range qterms {
			if _, ok := pt[term]; ok {
				disjoint = false
				break
			}
		}
		if disjoint {
			return doc
		}
	}
	return ""
}

// termsOf tokenizes text into the set of its distinct lowercased alphanumeric terms, the
// same bag the lexical judge measures coverage over, so the distractor selection agrees
// with the grader on what a term is.
func termsOf(text string) map[string]struct{} {
	set := map[string]struct{}{}
	for _, f := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		set[f] = struct{}{}
	}
	return set
}

// sortStringsAsInts sorts a slice of numeric id strings by their integer value in place, a
// stable readable order for the known-item query ids.
func sortStringsAsInts(s []string) {
	sort.Slice(s, func(i, j int) bool {
		a, _ := strconv.ParseUint(s[i], 10, 64)
		b, _ := strconv.ParseUint(s[j], 10, 64)
		return a < b
	})
}
