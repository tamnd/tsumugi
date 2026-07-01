package search

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
)

// TestAggregatorStatsFoldsChildren checks the deployment-wide field averages an aggregator
// reports are the token-weighted fold of its children's, not a flat mean: a child holding far
// more documents pulls the average toward its lengths in proportion to its document count, the
// same average a single index over the whole subtree would compute.
func TestAggregatorStatsFoldsChildren(t *testing.T) {
	// Child A: 100 docs averaging 10 body tokens. Child B: 300 docs averaging 50. The fold is
	// (100*10 + 300*50) / 400 = 40, not the flat mean (10+50)/2 = 30.
	a := &recordingSearcher{shards: 1, docs: 100, stats: GlobalStats{
		DocCount: 100, TokenCount: 1000, AvgDocLen: 10, AvgFieldLen: [4]float64{0, 10, 0},
	}}
	b := &recordingSearcher{shards: 1, docs: 300, stats: GlobalStats{
		DocCount: 300, TokenCount: 15000, AvgDocLen: 50, AvgFieldLen: [4]float64{0, 50, 0},
	}}
	agg := NewAggregator([]Searcher{a, b})

	gs := agg.Stats()
	if gs.DocCount != 400 {
		t.Fatalf("folded DocCount = %d, want 400", gs.DocCount)
	}
	if math.Abs(gs.AvgFieldLen[fBody]-40) > 1e-9 {
		t.Fatalf("folded body avg = %v, want 40 (token-weighted, not the flat mean 30)", gs.AvgFieldLen[fBody])
	}
	if math.Abs(gs.AvgDocLen-40) > 1e-9 {
		t.Fatalf("folded AvgDocLen = %v, want 40", gs.AvgDocLen)
	}

	// The fold composes through a sub-aggregator: a root over the aggregator and a third child
	// reports the average over all three children's documents.
	c := &recordingSearcher{shards: 1, docs: 100, stats: GlobalStats{
		DocCount: 100, TokenCount: 100, AvgDocLen: 1, AvgFieldLen: [4]float64{0, 1, 0},
	}}
	root := NewAggregator([]Searcher{agg, c})
	rgs := root.Stats()
	if rgs.DocCount != 500 {
		t.Fatalf("nested DocCount = %d, want 500", rgs.DocCount)
	}
	// (100*10 + 300*50 + 100*1) / 500 = 16100/500 = 32.2
	if math.Abs(rgs.AvgFieldLen[fBody]-32.2) > 1e-9 {
		t.Fatalf("nested body avg = %v, want 32.2", rgs.AvgFieldLen[fBody])
	}
}

// TestAggregatorPushesUnifiedAvgFieldLen is the length-normalization counterpart to
// TestAggregatorPushesSharedIDF: the aggregator folds the children's field averages into one
// deployment-wide set and pushes it down into every child's query, so a broker normalizes BM25F
// against the unified denominators rather than its own slice's. The children carry skewed
// averages, the case a broker's own averages would be wrong, and the test checks every child
// received the folded fleet averages, that a pinned override is left alone, and that a query
// with no lexical terms triggers no push.
func TestAggregatorPushesUnifiedAvgFieldLen(t *testing.T) {
	a := &recordingSearcher{shards: 1, docs: 100, stats: GlobalStats{
		DocCount: 100, TokenCount: 1000, AvgDocLen: 10, AvgFieldLen: [4]float64{0, 10, 0},
	}}
	b := &recordingSearcher{shards: 1, docs: 300, stats: GlobalStats{
		DocCount: 300, TokenCount: 15000, AvgDocLen: 50, AvgFieldLen: [4]float64{0, 50, 0},
	}}
	agg := NewAggregator([]Searcher{a, b})

	agg.SearchComplete(context.Background(), Query{Terms: []string{"common"}, K: 10})
	want := agg.Stats().AvgFieldLen
	for _, child := range []*recordingSearcher{a, b} {
		if child.gotAvg == nil {
			t.Fatal("child received no pushed AvgFieldLen")
		}
		if *child.gotAvg != want {
			t.Fatalf("child AvgFieldLen = %v, want folded fleet %v", *child.gotAvg, want)
		}
		// The pushed average must differ from the child's own slice average, the skew the
		// push-down removes: child a alone would normalize body against 10, not the fleet 40.
		if child.gotAvg[fBody] == child.stats.AvgFieldLen[fBody] {
			t.Fatalf("pushed body avg %v equals child's own slice avg; the fold changed nothing", child.gotAvg[fBody])
		}
	}

	// A query that already carries an AvgFieldLen override is left alone, so a caller can pin it.
	a.gotAvg, b.gotAvg = nil, nil
	pinned := [4]float64{1, 2, 3}
	agg.SearchComplete(context.Background(), Query{Terms: []string{"common"}, AvgFieldLen: &pinned, K: 10})
	if a.gotAvg == nil || *a.gotAvg != pinned {
		t.Fatalf("a pinned AvgFieldLen was overwritten: got %v", a.gotAvg)
	}

	// A query with no lexical terms has no BM25F to normalize, so the push is skipped.
	a.gotAvg, b.gotAvg = nil, nil
	agg.SearchComplete(context.Background(), Query{K: 10})
	if a.gotAvg != nil {
		t.Fatalf("AvgFieldLen pushed for a term-less query: got %v", *a.gotAvg)
	}
}

// lenSkewCorpus builds a corpus whose body length is skewed across the two halves: the first
// half holds short documents that mention the common term a few times, the second half holds
// long documents that mention it dozens of times. When the halves split across two brokers each
// broker's own average body length is far from the fleet's, which is exactly the partitioned
// case where a broker normalizing BM25F against its own slice would score a document differently
// than a single index over the whole corpus would. Every document carries the common term so the
// candidate set is recall-complete, the precondition the exactness proof rests on.
func lenSkewCorpus(n int) []doc {
	docs := make([]doc, n)
	half := n / 2
	for g := 0; g < n; g++ {
		reps := 2 + g%5 // short: the common term appears 2..6 times
		if g >= half {
			reps = 40 + g%20 // long: 40..59 times
		}
		body := strings.TrimSpace(strings.Repeat("common ", reps)) + fmt.Sprintf(" doc%d", g)
		docs[g] = doc{
			text: body,
			feats: map[feature.FeatureID]float64{
				feature.FeatInDegree:   float64((g * 37) % 4096),
				feature.FeatStaticRank: float64((g * 13) % 100),
				feature.FeatDocLen:     float64(reps),
			},
		}
	}
	return docs
}

// trainLengthModel fits a ranking model whose learned signal is the online body-BM25 feature, so
// the served ranking is sensitive to the per-field length normalization the partitioned push-down
// unifies. Every existing test model trains only the offline matrix columns, so the online
// BM25F features carry zero ranking weight and the partitioned-stats gap is invisible to them;
// this model weights OnBM25Body, the one column the body-length denominator moves, so a wrong
// denominator changes the ranking it produces. The label tracks the body-BM25 column over the
// concatenated offline-plus-online width the L2 model scores.
func trainLengthModel(t testing.TB) *rank.Model {
	t.Helper()
	nf := len(feature.DefaultSchema())
	width := nf + int(NumOnline)
	bodyCol := nf + int(OnBM25Body)
	d := &rank.Dataset{NumFeatures: width}
	r := lcgSeed(7)
	const queries, per = 80, 12
	for q := 0; q < queries; q++ {
		d.Groups = append(d.Groups, per)
		for i := 0; i < per; i++ {
			row := make([]float64, width)
			for f := range row {
				row[f] = r()
			}
			// Spread the body-BM25 column across the range production BM25 lands in and make the
			// label track it, so the boosted trees split on it and nothing else carries the signal.
			row[bodyCol] = r() * 8
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, row[bodyCol])
		}
	}
	p := rank.DefaultParams()
	p.Rounds = 80
	return rank.Train(d, p).Compile()
}

// TestAggregatorExactPartitionedStats is the partitioned-GlobalStats exactness gate: an
// aggregator over two brokers that each hold a length-skewed half of the corpus reproduces the
// top-k a single index over the whole corpus produces, because the aggregator folds the brokers'
// field averages into one deployment-wide set and pushes it down so both brokers normalize BM25F
// against the same denominators. The model weights the online body-BM25 feature, so the
// length normalization is load-bearing on the ranking, which is what makes this a real test of
// the push-down rather than of offline features that ignore it. The test pins idf so the only
// fleet statistic in play is the field averages, asserts the aggregator's ranked ids match the
// monolith's exactly and its scores match within reconstruction rounding, and asserts that a
// naive merge of the brokers' own-slice scores diverges from the monolith by far more than
// rounding, proving the push-down is what recovers exactness.
func TestAggregatorExactPartitionedStats(t *testing.T) {
	const n = 160
	docs := lenSkewCorpus(n)
	dir := t.TempDir()
	model := trainLengthModel(t)

	// The monolith is one broker over a single shard holding every document, the single index
	// the partitioned deployment must reproduce. Its field averages are the fleet averages.
	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	monoBroker := NewBroker([]*Shard{ms}, newTestCascade(model))
	defer func() { _ = monoBroker.Close() }()

	// Two brokers split the corpus at the length boundary, so broker 0 holds only short documents
	// and broker 1 only long ones: each broker's own average body length is far from the fleet's.
	half := n / 2
	s0path := filepath.Join(dir, "s0.tsumugi")
	s1path := filepath.Join(dir, "s1.tsumugi")
	buildShardFile(t, s0path, docs, 0, half, 0, false)
	buildShardFile(t, s1path, docs, half, n, uint32(half), false)
	s0, err := OpenShard(s0path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open s0: %v", err)
	}
	s1, err := OpenShard(s1path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open s1: %v", err)
	}
	b0 := NewBroker([]*Shard{s0}, newTestCascade(model))
	b1 := NewBroker([]*Shard{s1}, newTestCascade(model))
	agg := NewAggregator([]Searcher{b0, b1})
	defer func() { _ = b0.Close(); _ = b1.Close() }()

	// The brokers' own averages really are skewed, the precondition that makes the push-down do
	// work: broker 0's short body average is well below broker 1's long one.
	if b0.Stats().AvgFieldLen[fBody] >= b1.Stats().AvgFieldLen[fBody] {
		t.Fatalf("expected skew: b0 body avg %v should be below b1 %v", b0.Stats().AvgFieldLen[fBody], b1.Stats().AvgFieldLen[fBody])
	}
	// The folded fleet average sits between the brokers' and matches the monolith's, the single
	// denominator the push-down puts both brokers on.
	fleet := agg.Stats().AvgFieldLen
	if math.Abs(fleet[fBody]-monoBroker.Stats().AvgFieldLen[fBody]) > 1e-9 {
		t.Fatalf("folded fleet body avg %v != monolith %v", fleet[fBody], monoBroker.Stats().AvgFieldLen[fBody])
	}

	// Pin idf so the field averages are the only fleet statistic that differs between the paths,
	// and the same idf flows to the monolith, the aggregator, and the naive merge alike.
	idf := map[string]float64{"common": 3.0}
	ctx := context.Background()
	q := Query{Text: "common", Terms: lexical.Analyze("common"), TermIDF: idf, K: 20}

	want := monoBroker.Search(ctx, q)
	got := agg.Search(ctx, q)
	if len(got) != len(want) {
		t.Fatalf("aggregator returned %d hits, monolith %d", len(got), len(want))
	}
	for i := range want {
		if got[i].DocID != want[i].DocID {
			t.Fatalf("rank %d: aggregator doc %d, monolith doc %d", i, got[i].DocID, want[i].DocID)
		}
		// Scores match the monolith up to the floating-point rounding of reconstructing the field
		// sums from the brokers' averages, the documented negligible error of the fold.
		if d := math.Abs(got[i].Score - want[i].Score); d > 1e-9 {
			t.Fatalf("rank %d doc %d: aggregator score %v, monolith %v, diff %v exceeds rounding", i, got[i].DocID, got[i].Score, want[i].Score, d)
		}
	}

	// The push-down is load-bearing: a naive merge that lets each broker score against its own
	// slice average, the behavior before the fold, diverges from the monolith by far more than
	// reconstruction rounding. Each broker is searched with no pushed average so it falls back to
	// its own, then the lists are merged the way the aggregator merges comparable scores.
	naive := mergeNaive(b0.Search(ctx, q), b1.Search(ctx, q), q.K)
	if len(naive) != len(want) {
		t.Fatalf("naive merge returned %d hits, monolith %d", len(naive), len(want))
	}
	var maxDiff float64
	for i := range want {
		// Compare each monolith hit against the naive merge's score for the same document.
		ns, ok := naive[want[i].DocID]
		if !ok {
			// A document the monolith ranks that the naive merge dropped from the top-k is itself
			// a divergence large enough to prove the point.
			maxDiff = math.Inf(1)
			break
		}
		if d := math.Abs(ns - want[i].Score); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff <= 1e-6 {
		t.Fatalf("naive own-slice merge matched the monolith within %v; the length normalization is not load-bearing in this model", maxDiff)
	}
}

// mergeNaive merges two brokers' top-k lists into a docID-to-score map of the global top-k by
// comparable score, the merge an aggregator runs, used to show what the merge produces when each
// broker scored against its own slice statistics instead of the pushed deployment-wide ones.
func mergeNaive(a, b []Hit, k int) map[uint32]float64 {
	all := append(append([]Hit{}, a...), b...)
	sortHits(all)
	if k > 0 && len(all) > k {
		all = all[:k]
	}
	out := make(map[uint32]float64, len(all))
	for _, h := range all {
		out[h.DocID] = h.Score
	}
	return out
}

// TestBrokerHonorsAvgFieldLenOverride pins the broker half of the push-down: a broker scores
// its L2 against the pushed-down field averages when the query carries them and against its own
// when it does not, and the two produce different rankings when the averages differ under a
// length-sensitive model. Passing the broker's own averages as the override reproduces the
// no-override result exactly, so the override is a clean substitution of the denominator.
func TestBrokerHonorsAvgFieldLenOverride(t *testing.T) {
	const n = 80
	docs := lenSkewCorpus(n)
	dir := t.TempDir()
	model := trainLengthModel(t)

	path := filepath.Join(dir, "s.tsumugi")
	buildShardFile(t, path, docs, 0, n, 0, false)
	s, err := OpenShard(path, newTestCascade(model))
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	b := NewBroker([]*Shard{s}, newTestCascade(model))
	defer func() { _ = b.Close() }()

	idf := map[string]float64{"common": 3.0}
	ctx := context.Background()
	base := Query{Text: "common", Terms: lexical.Analyze("common"), TermIDF: idf, K: 20}

	// No override uses the broker's own field averages.
	own := b.Search(ctx, base)

	// Passing the broker's own averages as the override changes nothing, the no-op equivalence.
	ownAvg := b.Stats().AvgFieldLen
	qSame := base
	qSame.AvgFieldLen = &ownAvg
	same := b.Search(ctx, qSame)
	if len(same) != len(own) {
		t.Fatalf("override-with-own returned %d hits, no-override %d", len(same), len(own))
	}
	for i := range own {
		if same[i].DocID != own[i].DocID || math.Float64bits(same[i].Score) != math.Float64bits(own[i].Score) {
			t.Fatalf("rank %d: override-with-own %v differs from no-override %v", i, same[i], own[i])
		}
	}

	// A different override changes the scores, proving the broker really normalizes against the
	// pushed averages rather than ignoring them.
	other := [4]float64{0, ownAvg[fBody] * 4, 0}
	qOther := base
	qOther.AvgFieldLen = &other
	got := b.Search(ctx, qOther)
	differs := false
	for i := range own {
		if math.Float64bits(got[i].Score) != math.Float64bits(own[i].Score) {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("a different AvgFieldLen override produced identical scores; the broker is ignoring it")
	}
}
