package search

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/vector"
)

// doc is a synthetic corpus document: the lexical text, the feature values keyed by
// id, and an optional dense vector.
type doc struct {
	text  string
	feats map[feature.FeatureID]float64
	vec   []float32
}

const vecDim = 16

// makeCorpus builds a deterministic corpus where every document shares a common term
// so the lexical plane retrieves all of them, the in-degree feature spreads across
// documents so the model has a signal to rank on, and the vector is a smooth function
// of the document id. Sharing a term is what makes the candidate set recall-complete,
// which is the precondition the broker exactness proof rests on.
func makeCorpus(n int) []doc {
	docs := make([]doc, n)
	for g := 0; g < n; g++ {
		v := make([]float32, vecDim)
		for d := range v {
			v[d] = float32(math.Sin(float64(g*7+d) * 0.3))
		}
		docs[g] = doc{
			text: fmt.Sprintf("common document number %d term%d", g, g%11),
			feats: map[feature.FeatureID]float64{
				feature.FeatInDegree:   float64((g * 37) % 4096),
				feature.FeatPageRank:   float64(g%64) / 64,
				feature.FeatDocLen:     float64(50 + g%200),
				feature.FeatStaticRank: float64((g * 13) % 100),
			},
			vec: v,
		}
	}
	return docs
}

// buildShardFile writes docs[lo:hi] into one shard file with the given node base, so
// a document's global id is nodeBase plus its in-shard id. withVec adds the dense
// region. It mirrors the build pipeline: a lexical region, a feature matrix, and a
// vector region, all under the container.
func buildShardFile(t testing.TB, path string, docs []doc, lo, hi int, nodeBase uint32, withVec bool) {
	t.Helper()
	size := hi - lo
	lb := lexical.NewBuilder(lexical.DefaultParams())
	fb := feature.NewBuilder(feature.DefaultSchema(), feature.SchemaVersion)
	vb := vector.NewBuilder(vecDim).WithSeed(1).WithRerank(true)
	// The forward region carries the body text the online L2 features decode. Title and url
	// are left empty, so the online body-BM25 is the only length-normalized online signal a
	// length-sensitive model can key on; an offline-only model never reads the online columns,
	// so writing them changes nothing for the existing exactness tests.
	fwdb := forward.NewBuilder([]forward.Column{
		{Name: "url", Type: forward.ColString},
		{Name: "title", Type: forward.ColString},
		{Name: "body", Type: forward.ColString, Flags: forward.FlagBlob},
	})
	var tokens float64
	for i := 0; i < size; i++ {
		d := docs[lo+i]
		local := uint32(i)
		lb.AddDoc(local, map[lexical.Field]string{lexical.FieldBody: d.text})
		tokens += float64(len(lexical.Analyze(d.text)))
		for id, v := range d.feats {
			fb.Set(local, id, v)
		}
		fwdb.Set(local, "body", []byte(d.text))
		if withVec {
			vb.Add(d.vec)
		}
	}

	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	w.SetDocCount(uint32(size))
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
	if withVec {
		vregion, err := vb.Build()
		if err != nil {
			t.Fatalf("build vector: %v", err)
		}
		if err := w.AddRegion(tsumugi.RegionVector, tsumugi.CodecZstd, 0, 0, vregion); err != nil {
			t.Fatalf("add vector: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// trainModel fits a small ranking model over the default feature schema where the
// label tracks the in-degree feature, so the served model has a real learned signal
// and exercises the same train-to-serve path M9 built.
func trainModel(t testing.TB) *rank.Model {
	t.Helper()
	cols := feature.DefaultSchema()
	nf := len(cols)
	inDegCol := -1
	for i, c := range cols {
		if c.ID == feature.FeatInDegree {
			inDegCol = i
		}
	}
	d := &rank.Dataset{NumFeatures: nf}
	r := lcgSeed(99)
	const queries, per = 60, 10
	for q := 0; q < queries; q++ {
		d.Groups = append(d.Groups, per)
		for i := 0; i < per; i++ {
			row := make([]float64, nf)
			for f := range row {
				row[f] = r()
			}
			label := math.Round(row[inDegCol] * 4)
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, label)
		}
	}
	p := rank.DefaultParams()
	p.Rounds = 60
	return rank.Train(d, p).Compile()
}

// lcgSeed is a tiny deterministic generator local to this test.
func lcgSeed(seed uint64) func() float64 {
	s := seed
	return func() float64 {
		s = s*6364136223846793005 + 1442695040888963407
		return float64(s>>11) / float64(1<<53)
	}
}

func newTestCascade(model *rank.Model) *rank.Cascade {
	c := rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
	// Keep the cut wide so the recall-complete candidate set reaches the reranker
	// intact, which is what lets the broker reproduce the monolith exactly.
	c.L0Max = 100000
	c.L1Keep = 100000
	return c
}

// TestSingleShardSearch checks the standalone cascade over one shard returns a
// model-ranked top-k with global ids and descending scores across all planes.
func TestSingleShardSearch(t *testing.T) {
	docs := makeCorpus(120)
	dir := t.TempDir()
	path := filepath.Join(dir, "one.tsumugi")
	buildShardFile(t, path, docs, 0, 120, 0, true)

	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	hits := s.Search(Query{Text: "common document", Vector: docs[3].vec, K: 10})
	if len(hits) != 10 {
		t.Fatalf("got %d hits, want 10", len(hits))
	}
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Fatalf("hits not sorted: %v before %v", hits[i-1], hits[i])
		}
	}
}

// TestBrokerExactCrossShard is the M10 exactness gate: the broker's merged top-k over
// a partitioned collection equals the top-k a single index over every shard produces,
// bit for bit on both ids and scores. It builds one monolith shard over the whole
// corpus and four shards over its partitions, queries both with a term every document
// carries so the candidate set is recall-complete, and requires the two top-ks to be
// identical.
func TestBrokerExactCrossShard(t *testing.T) {
	const n, parts = 160, 4
	docs := makeCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	defer func() { _ = ms.Close() }()

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
	defer func() { _ = b.Close() }()

	if b.Stats().DocCount != n {
		t.Fatalf("global doc count = %d, want %d", b.Stats().DocCount, n)
	}

	q := Query{Text: "common document", K: 20}
	want := ms.Search(q)
	got := b.Search(context.Background(), q)

	if len(got) != len(want) {
		t.Fatalf("broker returned %d hits, monolith %d", len(got), len(want))
	}
	for i := range want {
		if got[i].DocID != want[i].DocID {
			t.Fatalf("rank %d: broker doc %d, monolith doc %d", i, got[i].DocID, want[i].DocID)
		}
		if math.Float64bits(got[i].Score) != math.Float64bits(want[i].Score) {
			t.Fatalf("rank %d doc %d: broker score %v, monolith %v", i, got[i].DocID, got[i].Score, want[i].Score)
		}
	}
}

// TestRoutingPrunesShards checks the routing index sends a query only to shards that
// hold one of its terms, and a term unique to one shard routes to exactly that shard.
func TestRoutingPrunesShards(t *testing.T) {
	dir := t.TempDir()
	docsA := []doc{{text: "alpha shared", feats: map[feature.FeatureID]float64{}}}
	docsB := []doc{{text: "beta shared", feats: map[feature.FeatureID]float64{}}}
	pa := filepath.Join(dir, "a.tsumugi")
	pb := filepath.Join(dir, "b.tsumugi")
	buildShardFile(t, pa, docsA, 0, 1, 0, false)
	buildShardFile(t, pb, docsB, 0, 1, 100, false)
	model := trainModel(t)
	sa, _ := OpenShard(pa, newTestCascade(model))
	sb, _ := OpenShard(pb, newTestCascade(model))
	defer func() { _ = sa.Close(); _ = sb.Close() }()

	ri := BuildRoutingIndex([]*Shard{sa, sb})
	if got := ri.Route(Query{Text: "alpha"}); len(got) != 1 || got[0] != 0 {
		t.Fatalf("route alpha = %v, want [0]", got)
	}
	if got := ri.Route(Query{Text: "beta"}); len(got) != 1 || got[0] != 1 {
		t.Fatalf("route beta = %v, want [1]", got)
	}
	if got := ri.Route(Query{Text: "shared"}); len(got) != 2 {
		t.Fatalf("route shared = %v, want both shards", got)
	}
	if got := ri.Route(Query{Text: "missing"}); len(got) != 0 {
		t.Fatalf("route missing = %v, want none", got)
	}
}

// TestAnalyzeOnceEquivalence is the M16b analyze-once gate: a query whose terms the
// broker pre-analyzed and shipped in Query.Terms produces the same fleet-wide top-k,
// bit for bit, as the same query carried as raw Text the shards each analyze. The
// pre-analyzed path is the one the broker takes in production so the analysis chain runs
// one time per query rather than once per shard; this proves taking it changes nothing
// the caller can observe. The Terms set is exactly what lexical.Analyze produces from
// the text, the same equality the broker relies on when it fills Terms itself.
func TestAnalyzeOnceEquivalence(t *testing.T) {
	const n, parts = 160, 4
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
	defer func() { _ = b.Close() }()

	const text = "common document"
	fromText := b.Search(context.Background(), Query{Text: text, K: 20})
	fromTerms := b.Search(context.Background(), Query{Text: text, Terms: lexical.Analyze(text), K: 20})

	if len(fromText) != len(fromTerms) {
		t.Fatalf("text path returned %d hits, pre-analyzed %d", len(fromText), len(fromTerms))
	}
	for i := range fromText {
		if fromText[i].DocID != fromTerms[i].DocID {
			t.Fatalf("rank %d: text doc %d, pre-analyzed doc %d", i, fromText[i].DocID, fromTerms[i].DocID)
		}
		if math.Float64bits(fromText[i].Score) != math.Float64bits(fromTerms[i].Score) {
			t.Fatalf("rank %d doc %d: text score %v, pre-analyzed %v", i, fromText[i].DocID, fromText[i].Score, fromTerms[i].Score)
		}
	}
}

// TestAnalyzeOnceSingleShard pins the same equivalence on the standalone single-shard
// path, where the pre-analyzed Terms set must produce the same ranking as the raw text
// the shard analyzes itself.
func TestAnalyzeOnceSingleShard(t *testing.T) {
	docs := makeCorpus(120)
	dir := t.TempDir()
	path := filepath.Join(dir, "one.tsumugi")
	buildShardFile(t, path, docs, 0, 120, 0, true)

	model := trainModel(t)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = s.Close() }()

	const text = "common document"
	fromText := s.Search(Query{Text: text, Vector: docs[3].vec, K: 10})
	fromTerms := s.Search(Query{Text: text, Terms: lexical.Analyze(text), Vector: docs[3].vec, K: 10})

	if len(fromText) != len(fromTerms) {
		t.Fatalf("text path %d hits, pre-analyzed %d", len(fromText), len(fromTerms))
	}
	for i := range fromText {
		if fromText[i].DocID != fromTerms[i].DocID || math.Float64bits(fromText[i].Score) != math.Float64bits(fromTerms[i].Score) {
			t.Fatalf("rank %d differs: text %v, pre-analyzed %v", i, fromText[i], fromTerms[i])
		}
	}
}
