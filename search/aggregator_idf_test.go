package search

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
)

// recordingSearcher is a stub Searcher that reports a fixed document frequency and document
// count and records the query it was searched with, so a test can see exactly what idf the
// aggregator pushed down to each child without depending on a model's idf sensitivity.
type recordingSearcher struct {
	shards  int
	docs    uint64
	df      map[string]uint32
	stats   GlobalStats
	gotIDF  map[string]float64
	gotAvg  *[3]float64
	gotTerm []string
}

func (r *recordingSearcher) NumShards() int     { return r.shards }
func (r *recordingSearcher) NumDocs() uint64    { return r.docs }
func (r *recordingSearcher) Stats() GlobalStats { return r.stats }
func (r *recordingSearcher) DocFreqs(_ context.Context, terms []string) map[string]uint32 {
	out := make(map[string]uint32)
	for _, t := range terms {
		if f, ok := r.df[t]; ok {
			out[t] = f
		}
	}
	return out
}
func (r *recordingSearcher) SearchComplete(_ context.Context, q Query) Results {
	r.gotIDF = q.TermIDF
	r.gotAvg = q.AvgFieldLen
	r.gotTerm = q.Terms
	return Results{ShardsTotal: r.shards, ShardsOK: r.shards}
}

// TestAggregatorNumDocsSumsChildren checks the fleet document count an aggregator reports is
// the sum over its children, the N the shared idf divides by.
func TestAggregatorNumDocsSumsChildren(t *testing.T) {
	a := NewAggregator([]Searcher{
		&recordingSearcher{shards: 2, docs: 80},
		&recordingSearcher{shards: 2, docs: 120},
	})
	if got := a.NumDocs(); got != 200 {
		t.Fatalf("NumDocs = %d, want 200", got)
	}
	// A nested aggregator reports its whole subtree's count, so the count composes up a tree.
	root := NewAggregator([]Searcher{a, &recordingSearcher{shards: 1, docs: 50}})
	if got := root.NumDocs(); got != 250 {
		t.Fatalf("nested NumDocs = %d, want 250", got)
	}
}

// TestAggregatorDocFreqsSumsChildren checks the aggregator's df is the per-child df summed,
// including a term only one child holds, and that it composes through a sub-aggregator.
func TestAggregatorDocFreqsSumsChildren(t *testing.T) {
	c0 := &recordingSearcher{shards: 1, docs: 80, df: map[string]uint32{"common": 80, "alpha": 30}}
	c1 := &recordingSearcher{shards: 1, docs: 120, df: map[string]uint32{"common": 120, "beta": 40}}
	a := NewAggregator([]Searcher{c0, c1})

	df := a.DocFreqs(context.Background(), []string{"common", "alpha", "beta"})
	want := map[string]uint32{"common": 200, "alpha": 30, "beta": 40}
	for term, w := range want {
		if df[term] != w {
			t.Fatalf("df[%q] = %d, want %d", term, df[term], w)
		}
	}

	// Through a sub-aggregator the gather recurses, so the root sees the same fleet totals.
	root := NewAggregator([]Searcher{a})
	rdf := root.DocFreqs(context.Background(), []string{"common", "alpha", "beta"})
	for term, w := range want {
		if rdf[term] != w {
			t.Fatalf("nested df[%q] = %d, want %d", term, rdf[term], w)
		}
	}

	// An empty term set gathers nothing.
	if got := a.DocFreqs(context.Background(), nil); len(got) != 0 {
		t.Fatalf("DocFreqs(nil) = %v, want empty", got)
	}
}

// TestAggregatorPushesSharedIDF is the cross-broker exact-idf proof: the aggregator gathers
// the fleet-wide df across its children, computes one idf against the fleet N, and pushes
// that same idf down into every child's query, so a term scores against one collection-wide
// idf no matter which child holds it. The children report diverging df, the case where a
// broker's own local idf would be wrong, and the test checks every child received the fleet
// idf rather than the idf it would have computed from its own slice.
func TestAggregatorPushesSharedIDF(t *testing.T) {
	c0 := &recordingSearcher{shards: 1, docs: 80, df: map[string]uint32{"common": 80, "alpha": 30}}
	c1 := &recordingSearcher{shards: 1, docs: 120, df: map[string]uint32{"common": 120, "beta": 40}}
	a := NewAggregator([]Searcher{c0, c1})

	terms := []string{"common", "alpha", "beta"}
	a.SearchComplete(context.Background(), Query{Terms: terms, K: 10})

	// The fleet idf is computed from the summed df over the fleet document count.
	wantIDF := map[string]float64{
		"common": lexical.IDF(200, 200),
		"alpha":  lexical.IDF(200, 30),
		"beta":   lexical.IDF(200, 40),
	}
	for _, child := range []*recordingSearcher{c0, c1} {
		if child.gotIDF == nil {
			t.Fatal("child received no pushed idf")
		}
		for term, w := range wantIDF {
			if child.gotIDF[term] != w {
				t.Fatalf("child idf[%q] = %v, want fleet %v", term, child.gotIDF[term], w)
			}
		}
	}

	// The pushed idf must differ from the idf a child would compute from its own slice, which
	// is the divergence the gather removes: c0 alone would score alpha against IDF(80,30).
	if local := lexical.IDF(80, 30); c0.gotIDF["alpha"] == local {
		t.Fatalf("pushed alpha idf equals c0's local idf %v; the gather changed nothing", local)
	}

	// A query that already carries an idf override is left alone, so a caller can pin idf.
	c0.gotIDF, c1.gotIDF = nil, nil
	pinned := map[string]float64{"common": 1.5}
	a.SearchComplete(context.Background(), Query{Terms: terms, TermIDF: pinned, K: 10})
	if c0.gotIDF["common"] != 1.5 {
		t.Fatalf("a pinned TermIDF was overwritten: got %v", c0.gotIDF["common"])
	}
}

// dfCorpus builds a corpus where alpha appears only in the first quarter and beta only in the
// last quarter, every document sharing common, so when the four shards split across two
// brokers each broker holds exactly one of the rare terms. That is the case a single broker
// cannot see the fleet df for, the one the aggregator gather exists to handle.
func dfCorpus(n int) []doc {
	docs := make([]doc, n)
	q := n / 4
	for g := 0; g < n; g++ {
		text := "common document"
		if g < q {
			text = "alpha common document"
		} else if g >= 3*q {
			text = "beta common document"
		}
		docs[g] = doc{
			text: text,
			feats: map[feature.FeatureID]float64{
				feature.FeatInDegree:   float64((g * 37) % 4096),
				feature.FeatStaticRank: float64((g * 13) % 100),
				feature.FeatDocLen:     float64(20 + g%50),
			},
		}
	}
	return docs
}

// TestAggregatorFleetDFMatchesMonolith is the real-broker proof: an aggregator over two
// brokers, each holding shards with a rare term the other broker never sees, reports the same
// fleet df and document count a single broker over every shard does, and that fleet idf
// differs from the idf either broker would compute from its own shards alone. This is the
// statistic the cross-broker exact rerank rests on, gathered over real shards.
func TestAggregatorFleetDFMatchesMonolith(t *testing.T) {
	const n, parts = 160, 4
	docs := dfCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	mono := filepath.Join(dir, "mono.tsumugi")
	buildShardFile(t, mono, docs, 0, n, 0, false)
	ms, err := OpenShard(mono, newTestCascade(model))
	if err != nil {
		t.Fatalf("open mono: %v", err)
	}
	monoBroker := NewBroker([]*Shard{ms}, newTestCascade(model))
	defer func() { _ = monoBroker.Close() }()

	size := n / parts
	shards := make([]*Shard, parts)
	for p := 0; p < parts; p++ {
		path := filepath.Join(dir, "s"+string(rune('0'+p))+".tsumugi")
		lo := p * size
		buildShardFile(t, path, docs, lo, lo+size, uint32(lo), false)
		sh, err := OpenShard(path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %d: %v", p, err)
		}
		shards[p] = sh
	}
	b0 := NewBroker(shards[:2], newTestCascade(model))
	b1 := NewBroker(shards[2:], newTestCascade(model))
	agg := NewAggregator([]Searcher{b0, b1})
	defer func() { _ = b0.Close(); _ = b1.Close() }()

	ctx := context.Background()
	terms := []string{"alpha", "beta", "common"}

	if agg.NumDocs() != monoBroker.NumDocs() {
		t.Fatalf("aggregator NumDocs = %d, monolith %d", agg.NumDocs(), monoBroker.NumDocs())
	}

	want := monoBroker.DocFreqs(ctx, terms)
	got := agg.DocFreqs(ctx, terms)
	for _, term := range terms {
		if got[term] != want[term] {
			t.Fatalf("fleet df[%q] = %d, monolith %d", term, got[term], want[term])
		}
		if want[term] == 0 {
			t.Fatalf("term %q has zero df; the corpus or analysis is not what the test assumes", term)
		}
	}

	// Each broker holds only one of the rare terms, so neither sees the fleet df alone: that
	// is why the aggregator gather is necessary rather than reusing a broker's own idf.
	df0 := b0.DocFreqs(ctx, terms)
	df1 := b1.DocFreqs(ctx, terms)
	if _, ok := df0["beta"]; ok {
		t.Fatal("broker 0 should not hold beta")
	}
	if _, ok := df1["alpha"]; ok {
		t.Fatal("broker 1 should not hold alpha")
	}

	// The fleet idf for alpha differs from the idf broker 0 would compute over its own docs,
	// the bias the push-down removes.
	fleetAlpha := lexical.IDF(agg.NumDocs(), uint64(got["alpha"]))
	localAlpha := lexical.IDF(b0.NumDocs(), uint64(df0["alpha"]))
	if fleetAlpha == localAlpha {
		t.Fatalf("fleet and broker-local alpha idf both %v; the corpus has no cross-broker skew", fleetAlpha)
	}
}
