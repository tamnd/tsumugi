package search

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/vector"
)

// ccrawlParquet is the real crawl export the online features are exercised against,
// the same sample the graph and signal milestones gate on. The real-data tests skip
// when it is absent so the suite still runs on a machine without the corpus.
const ccrawlParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// textDoc is a candidate's stored text, the three forward-region columns the online
// features decode.
type textDoc struct {
	url, title, body string
}

// buildForward builds a forward region over the text docs with the url, title, and
// body columns the extractor reads, the same schema the build writes.
func buildForward(t testing.TB, docs []textDoc) *forward.Region {
	t.Helper()
	cols := []forward.Column{
		{Name: "url", Type: forward.ColString},
		{Name: "title", Type: forward.ColString},
		{Name: "body", Type: forward.ColString, Flags: forward.FlagBlob},
	}
	fb := forward.NewBuilder(cols)
	for i, d := range docs {
		id := uint32(i)
		fb.Set(id, "url", []byte(d.url))
		fb.Set(id, "title", []byte(d.title))
		fb.Set(id, "body", []byte(d.body))
	}
	r, err := forward.Open(fb.Build())
	if err != nil {
		t.Fatalf("open forward: %v", err)
	}
	return r
}

// extract builds an extractor over the docs and returns the online feature vector of
// one document, the shape every mechanism test reads.
func extract(t testing.TB, q Query, docs []textDoc, idf map[string]float64, avgBody float64, id uint32) []float64 {
	t.Helper()
	fwd := buildForward(t, docs)
	e := newOnlineExtractor(q, fwd, nil, idf, [3]float64{fBody: avgBody})
	return e.features(id)
}

func TestOnlineBM25RewardsFrequency(t *testing.T) {
	docs := []textDoc{
		{body: "apple apple apple banana cherry"},
		{body: "apple banana cherry date elderberry"},
	}
	q := Query{Text: "apple"}
	idf := map[string]float64{"apple": 1.0}
	f0 := extract(t, q, docs, idf, 5, 0)
	f1 := extract(t, q, docs, idf, 5, 1)
	if f0[OnBM25Body] <= f1[OnBM25Body] {
		t.Fatalf("more frequent term should score higher: doc0 %.4f, doc1 %.4f", f0[OnBM25Body], f1[OnBM25Body])
	}
	if f0[OnBM25Body] <= 0 {
		t.Fatalf("bm25 of a present term should be positive, got %.4f", f0[OnBM25Body])
	}
}

func TestOnlineBM25RewardsIDF(t *testing.T) {
	docs := []textDoc{{body: "apple banana"}}
	rare := extract(t, Query{Text: "apple"}, docs, map[string]float64{"apple": 5.0}, 2, 0)
	common := extract(t, Query{Text: "apple"}, docs, map[string]float64{"apple": 0.5}, 2, 0)
	if rare[OnBM25Body] <= common[OnBM25Body] {
		t.Fatalf("rarer term should score higher: rare %.4f, common %.4f", rare[OnBM25Body], common[OnBM25Body])
	}
}

func TestOnlineTermCoverage(t *testing.T) {
	docs := []textDoc{
		{body: "apple banana cherry"},
		{body: "apple date elderberry"},
	}
	q := Query{Text: "apple banana cherry"}
	all := extract(t, q, docs, nil, 3, 0)
	if math.Abs(all[OnTermCoverage]-1.0) > 1e-9 {
		t.Fatalf("full coverage want 1.0, got %.4f", all[OnTermCoverage])
	}
	one := extract(t, q, docs, nil, 3, 1)
	if math.Abs(one[OnTermCoverage]-1.0/3.0) > 1e-9 {
		t.Fatalf("one-of-three coverage want 0.333, got %.4f", one[OnTermCoverage])
	}
}

func TestOnlineExactMatch(t *testing.T) {
	docs := []textDoc{
		{title: "fresh red apple pie", body: "i ate a red apple today"},
		{title: "apple is red", body: "the apple was very red"},
	}
	q := Query{Text: "red apple"}
	hit := extract(t, q, docs, nil, 6, 0)
	if hit[OnExactMatch] != 1 {
		t.Fatalf("contiguous phrase should match, got %.0f", hit[OnExactMatch])
	}
	if hit[OnExactMatchTitle] != 1 {
		t.Fatalf("title phrase should match, got %.0f", hit[OnExactMatchTitle])
	}
	miss := extract(t, q, docs, nil, 6, 1)
	if miss[OnExactMatch] != 0 {
		t.Fatalf("scattered terms should not match as a phrase, got %.0f", miss[OnExactMatch])
	}
}

func TestOnlineProximityCloserIsLarger(t *testing.T) {
	docs := []textDoc{
		{body: "alpha beta gamma delta epsilon"}, // query terms adjacent at 0,1
		{body: "alpha gamma delta epsilon beta"}, // query terms at 0 and 4
	}
	q := Query{Text: "alpha beta"}
	near := extract(t, q, docs, nil, 5, 0)
	far := extract(t, q, docs, nil, 5, 1)
	// doc0: window over {alpha@0, beta@1} = 2 tokens, feature 1/3.
	if math.Abs(near[OnUnorderedProximity]-1.0/3.0) > 1e-9 {
		t.Fatalf("adjacent window want 0.333, got %.4f", near[OnUnorderedProximity])
	}
	// doc1: window over {alpha@0, beta@4} = 5 tokens, feature 1/6.
	if math.Abs(far[OnUnorderedProximity]-1.0/6.0) > 1e-9 {
		t.Fatalf("spread window want 0.167, got %.4f", far[OnUnorderedProximity])
	}
	if near[OnUnorderedProximity] <= far[OnUnorderedProximity] {
		t.Fatalf("closer terms should score higher: near %.4f, far %.4f", near[OnUnorderedProximity], far[OnUnorderedProximity])
	}
	if near[OnMinPairDistance] <= far[OnMinPairDistance] {
		t.Fatalf("closer pair should score higher: near %.4f, far %.4f", near[OnMinPairDistance], far[OnMinPairDistance])
	}
}

func TestOnlineOrderedProximity(t *testing.T) {
	docs := []textDoc{
		{body: "alpha x beta"}, // in order, span 3
		{body: "beta x alpha"}, // reverse order, no in-order match
	}
	q := Query{Text: "alpha beta"}
	inOrder := extract(t, q, docs, nil, 3, 0)
	reverse := extract(t, q, docs, nil, 3, 1)
	if math.Abs(inOrder[OnOrderedProximity]-1.0/4.0) > 1e-9 {
		t.Fatalf("in-order span 3 want 0.25, got %.4f", inOrder[OnOrderedProximity])
	}
	if reverse[OnOrderedProximity] != 0 {
		t.Fatalf("reverse order should have no ordered span, got %.4f", reverse[OnOrderedProximity])
	}
	// The unordered window still covers both terms in either order.
	if reverse[OnUnorderedProximity] <= 0 {
		t.Fatalf("reverse order should still have an unordered window, got %.4f", reverse[OnUnorderedProximity])
	}
}

func TestOnlineFieldHitsAndURL(t *testing.T) {
	docs := []textDoc{{
		url:   "https://apple.com/fruit/banana",
		title: "apple recipes",
		body:  "an apple a day",
	}}
	q := Query{Text: "apple banana"}
	f := extract(t, q, docs, nil, 4, 0)
	if f[OnFieldHitTitle] != 1 {
		t.Fatalf("title holds only apple, want 1 hit, got %.0f", f[OnFieldHitTitle])
	}
	if f[OnFieldHitBody] != 1 {
		t.Fatalf("body holds only apple, want 1 hit, got %.0f", f[OnFieldHitBody])
	}
	if f[OnURLTermMatch] != 2 {
		t.Fatalf("url holds apple and banana, want 2, got %.0f", f[OnURLTermMatch])
	}
	if f[OnURLExactHost] != 1 {
		t.Fatalf("host apple.com should match query term apple, got %.0f", f[OnURLExactHost])
	}
}

func TestOnlineBroadcast(t *testing.T) {
	docs := []textDoc{{body: "apple banana"}}
	idf := map[string]float64{"apple": 2.0, "banana": 5.0, "cherry": 9.0}
	f := extract(t, Query{Text: "apple banana cherry"}, docs, idf, 2, 0)
	if f[OnQueryLength] != 3 {
		t.Fatalf("three distinct terms, want query length 3, got %.0f", f[OnQueryLength])
	}
	if math.Abs(f[OnIdfSum]-16.0) > 1e-9 {
		t.Fatalf("idf sum want 16, got %.4f", f[OnIdfSum])
	}
	if math.Abs(f[OnIdfMax]-9.0) > 1e-9 {
		t.Fatalf("idf max want 9, got %.4f", f[OnIdfMax])
	}
}

func TestOnlineDenseCosine(t *testing.T) {
	const dim = 16
	vb := vector.NewBuilder(dim).WithSeed(1).WithRerank(true)
	base := make([]float32, dim)
	for i := range base {
		base[i] = float32(math.Sin(float64(i) * 0.5))
	}
	orth := make([]float32, dim)
	for i := range orth {
		orth[i] = float32(math.Cos(float64(i) * 0.5))
	}
	vb.Add(base)
	vb.Add(orth)
	vr, err := vector.Open(vb.Build())
	if err != nil {
		t.Fatalf("open vector: %v", err)
	}
	docs := []textDoc{{body: "a"}, {body: "b"}}
	fwd := buildForward(t, docs)
	e := newOnlineExtractor(Query{Vector: base}, fwd, vr, nil, [3]float64{fBody: 1})
	same := e.features(0)[OnDenseCosine]
	diff := e.features(1)[OnDenseCosine]
	if same < 0.99 {
		t.Fatalf("cosine of a vector with itself should be ~1, got %.4f", same)
	}
	if same <= diff {
		t.Fatalf("self cosine should beat the other vector: self %.4f, other %.4f", same, diff)
	}
}

// TestOnlineFeaturesReachModel is the wiring gate: it proves the online features
// are concatenated onto the matrix row and fed to the L2 model, not dropped. It
// builds a shard with stored text, trains a model whose label is the exact-match
// online feature, and checks the cascade ranks the document that contains the query
// as a contiguous phrase above one that holds the same terms scattered. If the
// online features never reached the model both documents would tie on the matrix
// alone and the phrase document would not be singled out.
func TestOnlineFeaturesReachModel(t *testing.T) {
	docs := []textDoc{
		{url: "https://a.example/x", title: "a", body: "red apple is common here"},      // doc 0: contiguous "red apple"
		{url: "https://b.example/y", title: "b", body: "apple and red are common here"}, // doc 1: terms scattered
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "text.tsumugi")
	buildTextShard(t, path, docs)

	model := trainExactMatchModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	hits := s.Search(Query{Text: "red apple", K: 2})
	if len(hits) != 2 {
		t.Fatalf("want 2 hits, got %d", len(hits))
	}
	if hits[0].DocID != 0 {
		t.Fatalf("phrase document should rank first, got doc %d (scores %.4f, %.4f)", hits[0].DocID, hits[0].Score, hits[1].Score)
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("phrase document should score strictly higher: %.4f vs %.4f", hits[0].Score, hits[1].Score)
	}
}

// buildTextShard writes a shard with a lexical index, a feature matrix, and a
// forward store holding the url, title, and body, the regions the online extractor
// reads. Every body shares the term "here" so the lexical plane recalls both
// documents for the test query, the recall-completeness the rerank needs.
func buildTextShard(t testing.TB, path string, docs []textDoc) {
	t.Helper()
	lb := lexical.NewBuilder(lexical.DefaultParams())
	fb := feature.NewBuilder(feature.DefaultSchema(), 1)
	fwdCols := []forward.Column{
		{Name: "url", Type: forward.ColString},
		{Name: "title", Type: forward.ColString},
		{Name: "body", Type: forward.ColString, Flags: forward.FlagBlob},
	}
	fwdb := forward.NewBuilder(fwdCols)
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
	w.SetNodeBase(0)
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

// trainExactMatchModel fits a model over the full L2 width, matrix columns followed
// by the online features, whose label is the exact-match online column. The model
// learns to reward a contiguous phrase match, the online signal the wiring test
// checks the cascade surfaces.
func trainExactMatchModel(t testing.TB) *rank.Model {
	t.Helper()
	nf := len(feature.DefaultSchema()) + int(NumOnline)
	exactCol := len(feature.DefaultSchema()) + int(OnExactMatch)
	d := &rank.Dataset{NumFeatures: nf}
	r := lcgSeed(7)
	const queries, per = 80, 8
	for q := 0; q < queries; q++ {
		d.Groups = append(d.Groups, per)
		for i := 0; i < per; i++ {
			row := make([]float64, nf)
			for f := range row {
				row[f] = r()
			}
			// Make the exact-match column a clean 0/1 and let it drive the label, so
			// the trees fit a strong, learnable dependence on the online feature.
			ex := 0.0
			if r() < 0.5 {
				ex = 1.0
			}
			row[exactCol] = ex
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, math.Round(ex*4))
		}
	}
	p := rank.DefaultParams()
	p.Rounds = 80
	return rank.Train(d, p).Compile()
}

func TestOnlineMissingValues(t *testing.T) {
	// No forward region: text features are zero, not invented.
	e := newOnlineExtractor(Query{Text: "apple"}, nil, nil, map[string]float64{"apple": 1}, [3]float64{fBody: 5})
	f := e.features(0)
	if f[OnBM25Body] != 0 || f[OnTermCoverage] != 0 {
		t.Fatalf("absent text should leave text features zero, got bm25 %.4f cov %.4f", f[OnBM25Body], f[OnTermCoverage])
	}
	// No vector query: the dense cosine is the missing sentinel, distinct from zero.
	if f[OnDenseCosine] != missingFeature {
		t.Fatalf("absent dense cosine should be the sentinel %.1f, got %.4f", missingFeature, f[OnDenseCosine])
	}
}

// TestOnlineFeaturesOnCCrawl is the real-data gate: it builds a forward region over a
// slice of the real crawl, picks a query from a document's own title, and checks the
// extractor produces well-formed features over the real text distribution: coverage
// and the proximity transforms stay in their bounds, BM25 is non-negative, and the
// source document, which contains every query term by construction, comes back fully
// covered. The synthetic gates prove the exact mechanism; this proves the extractor
// survives real, messy, multi-language pages without producing a value outside its
// declared range.
func TestOnlineFeaturesOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl sample not present: %v", err)
	}
	docs := readRealDocs(t, 600)
	if len(docs) < 50 {
		t.Skipf("too few real docs with text: %d", len(docs))
	}

	// Pick a source document with a few title tokens, and use its title as the query.
	srcID := -1
	var terms []string
	for i, d := range docs {
		toks := uniqueTokens(d.title)
		if len(toks) >= 2 && len(toks) <= 6 {
			srcID = i
			terms = toks
			break
		}
	}
	if srcID < 0 {
		t.Skip("no document with a usable title in the sample")
	}
	q := Query{Text: docs[srcID].title}
	fwd := buildForward(t, docs)
	idf := map[string]float64{}
	for _, term := range terms {
		idf[term] = 1.0
	}
	e := newOnlineExtractor(q, fwd, nil, idf, [3]float64{fBody: 200})

	for id := range docs {
		f := e.features(uint32(id))
		check01 := func(name string, v float64) {
			if v < 0 || v > 1.0001 {
				t.Fatalf("%s out of [0,1] for doc %d: %.4f", name, id, v)
			}
		}
		check01("term_coverage", f[OnTermCoverage])
		check01("term_coverage_title", f[OnTermCoverageTitle])
		check01("ordered_proximity", f[OnOrderedProximity])
		check01("unordered_proximity", f[OnUnorderedProximity])
		check01("min_pair_distance", f[OnMinPairDistance])
		check01("first_occurrence", f[OnFirstOccurrence])
		if f[OnExactMatch] != 0 && f[OnExactMatch] != 1 {
			t.Fatalf("exact_match should be a flag, got %.4f", f[OnExactMatch])
		}
		if f[OnBM25Body] < 0 || f[OnBM25Title] < 0 || f[OnBM25FTotal] < 0 {
			t.Fatalf("bm25 should be non-negative for doc %d", id)
		}
		// An exact phrase match implies every query term is present, so coverage is full.
		if f[OnExactMatch] == 1 && f[OnTermCoverage] < 0.999 {
			t.Fatalf("exact match without full coverage at doc %d: cov %.4f", id, f[OnTermCoverage])
		}
	}

	// The source document's own title is the query, so it must be fully covered in the
	// title and score a positive title BM25.
	src := e.features(uint32(srcID))
	if src[OnTermCoverageTitle] < 0.999 {
		t.Fatalf("source doc should be fully covered by its own title, got %.4f", src[OnTermCoverageTitle])
	}
	if src[OnBM25Title] <= 0 {
		t.Fatalf("source doc should have a positive title BM25, got %.4f", src[OnBM25Title])
	}
	t.Logf("checked %d real docs against query %q (%d terms)", len(docs), q.Text, len(terms))
}

// BenchmarkOnlineExtract measures the per-candidate online feature cost over real
// crawl text, the figure the L2 budget of one to two milliseconds over two hundred
// survivors is checked against: at N ns per candidate the stage costs 200*N, which
// must stay inside the slice. It builds the extractor once and times one features
// call per iteration over real documents.
func BenchmarkOnlineExtract(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl sample not present: %v", err)
	}
	docs := readRealDocs(b, 600)
	if len(docs) < 50 {
		b.Skipf("too few real docs: %d", len(docs))
	}
	fwd := buildForward(b, docs)
	q := Query{Text: docs[0].title}
	idf := map[string]float64{}
	for _, t := range lexical.Analyze(q.Text) {
		idf[t] = 1.0
	}
	e := newOnlineExtractor(q, fwd, nil, idf, [3]float64{fBody: 200})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = e.features(uint32(i % len(docs)))
	}
}

// readRealDocs reads up to limit documents with non-empty bodies from the crawl
// sample and reduces each to its url, a derived title, and its body.
func readRealDocs(t testing.TB, limit int) []textDoc {
	t.Helper()
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Skipf("open ccrawl: %v", err)
	}
	defer func() { _ = src.Close() }()
	var out []textDoc
	for len(out) < limit {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read ccrawl: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		out = append(out, textDoc{url: d.URL, title: firstLine(d.Body), body: d.Body})
	}
	return out
}

// firstLine returns the first non-empty line of the body, stripped of leading
// markdown heading marks, a cheap stand-in for the build's title derivation good
// enough to drive a query in the test.
func firstLine(body string) string {
	start := 0
	for start < len(body) {
		end := start
		for end < len(body) && body[end] != '\n' {
			end++
		}
		line := body[start:end]
		for len(line) > 0 && (line[0] == '#' || line[0] == ' ') {
			line = line[1:]
		}
		if len(line) > 0 {
			if len(line) > 120 {
				line = line[:120]
			}
			return line
		}
		start = end + 1
	}
	return ""
}

// uniqueTokens returns the distinct analyzed tokens of a string, the unit the query
// terms are counted in.
func uniqueTokens(s string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range lexical.Analyze(s) {
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}
