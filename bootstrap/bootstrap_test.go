package bootstrap

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/tamnd/tsumugi/eval"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// fakeSource returns canned feature hits per query, standing in for the broker so the
// bootstrap is exercised without opening a collection.
type fakeSource struct {
	hits map[string][]search.FeatureHit
}

func (f *fakeSource) SearchFeatures(_ context.Context, q search.Query) []search.FeatureHit {
	return f.hits[q.Text]
}

// gradeJudge grades a passage by a number it reads back from the body, so a test can fix
// the grade of each candidate exactly and check the pool, dataset, and qrels carry it.
type gradeJudge struct{}

func (gradeJudge) Grade(_ context.Context, _ string, p eval.Passage) (eval.Grade, error) {
	var g int
	_, err := fmt.Sscanf(p.Body, "grade:%d", &g)
	if err != nil {
		return 0, err
	}
	return eval.Grade(g), nil
}

func TestBuildJudgesPoolAndDropsEmpty(t *testing.T) {
	src := &fakeSource{hits: map[string][]search.FeatureHit{
		"alpha": {
			{DocID: 1, Row: []float64{0.1, 0.9}},
			{DocID: 2, Row: []float64{0.2, 0.1}},
		},
		"beta": { // every candidate's passage is missing, so the query drops out
			{DocID: 9, Row: []float64{0.5, 0.5}},
		},
	}}
	passage := func(id uint32) (eval.Passage, bool) {
		switch id {
		case 1:
			return eval.Passage{Doc: "d1", Body: "grade:3"}, true
		case 2:
			return eval.Passage{Doc: "d2", Body: "grade:0"}, true
		}
		return eval.Passage{}, false
	}
	qf := func(raw string) search.Query { return search.Query{Text: raw, K: 10} }
	pool, err := Build(context.Background(), src, gradeJudge{}, qf,
		map[string]string{"a": "alpha", "b": "beta"}, passage)
	if err != nil {
		t.Fatal(err)
	}
	if len(pool.Queries) != 1 {
		t.Fatalf("got %d queries, want 1 (empty query b dropped)", len(pool.Queries))
	}
	q := pool.Queries[0]
	if q.ID != "a" || len(q.Docs) != 2 {
		t.Fatalf("got query %q with %d docs, want a with 2", q.ID, len(q.Docs))
	}
	if pool.NumFeatures != 2 {
		t.Fatalf("got NumFeatures %d, want 2", pool.NumFeatures)
	}
	if q.Docs[0].Grade != 3 || q.Docs[1].Grade != 0 {
		t.Fatalf("grades %v/%v, want 3/0", q.Docs[0].Grade, q.Docs[1].Grade)
	}

	qr := pool.Qrels()
	if qr["a"]["d1"] != 3 || qr["a"]["d2"] != 0 {
		t.Fatalf("qrels %v, want d1=3 d2=0", qr["a"])
	}
	d := pool.Dataset()
	if len(d.Features) != 2 || len(d.Labels) != 2 || len(d.Groups) != 1 || d.Groups[0] != 2 {
		t.Fatalf("dataset features=%d labels=%d groups=%v", len(d.Features), len(d.Labels), d.Groups)
	}
}

func TestBuildRejectsRaggedRows(t *testing.T) {
	src := &fakeSource{hits: map[string][]search.FeatureHit{
		"alpha": {
			{DocID: 1, Row: []float64{0.1, 0.9}},
			{DocID: 2, Row: []float64{0.2}}, // narrower than the first row
		},
	}}
	passage := func(id uint32) (eval.Passage, bool) {
		return eval.Passage{Doc: fmt.Sprintf("d%d", id), Body: "grade:1"}, true
	}
	qf := func(raw string) search.Query { return search.Query{Text: raw} }
	_, err := Build(context.Background(), src, gradeJudge{}, qf, map[string]string{"a": "alpha"}, passage)
	if err == nil {
		t.Fatal("want an error on ragged feature rows, got nil")
	}
}

func TestBuildPropagatesJudgeError(t *testing.T) {
	src := &fakeSource{hits: map[string][]search.FeatureHit{
		"alpha": {{DocID: 1, Row: []float64{0.1}}},
	}}
	passage := func(id uint32) (eval.Passage, bool) {
		return eval.Passage{Doc: "d1", Body: "not a grade"}, true
	}
	qf := func(raw string) search.Query { return search.Query{Text: raw} }
	_, err := Build(context.Background(), src, gradeJudge{}, qf, map[string]string{"a": "alpha"}, passage)
	if err == nil {
		t.Fatal("want the judge parse error to propagate, got nil")
	}
}

func TestSplitIsDisjointAndStable(t *testing.T) {
	var qs []Query
	for i := 0; i < 40; i++ {
		qs = append(qs, Query{ID: fmt.Sprintf("q%02d", i)})
	}
	train, evalSet := Split(qs, 0.5)
	if len(train)+len(evalSet) != len(qs) {
		t.Fatalf("split lost queries: %d + %d != %d", len(train), len(evalSet), len(qs))
	}
	if len(train) == 0 || len(evalSet) == 0 {
		t.Fatalf("a side is empty: train=%d eval=%d", len(train), len(evalSet))
	}
	// Sorting by hash makes the held-out fraction exact, not approximate.
	if len(evalSet) != 20 {
		t.Fatalf("0.5 of 40 should hold out exactly 20, got %d", len(evalSet))
	}
	if _, e := Split(qs, 0.3); len(e) != 12 {
		t.Fatalf("0.3 of 40 should hold out 12, got %d", len(e))
	}
	seen := map[string]bool{}
	for _, q := range append(append([]Query{}, train...), evalSet...) {
		if seen[q.ID] {
			t.Fatalf("query %q on both sides", q.ID)
		}
		seen[q.ID] = true
	}
	// Stable: the same input splits the same way.
	train2, eval2 := Split(qs, 0.5)
	if len(train2) != len(train) || len(eval2) != len(evalSet) {
		t.Fatal("split is not reproducible")
	}
	for i := range train {
		if train2[i].ID != train[i].ID {
			t.Fatal("split order changed between runs")
		}
	}
}

// TestGateImprovesOverUntrained builds synthetic queries where relevance is carried by
// one feature and the untrained baseline ranks by an unrelated one, then checks the model
// the gate trains on the graded labels recovers the relevant feature and beats the
// baseline on the held-out split, the M9 invariant.
func TestGateImprovesOverUntrained(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	var queries []Query
	for qi := 0; qi < 24; qi++ {
		var docs []Doc
		for di := 0; di < 8; di++ {
			grade := float64(di % 4) // a spread of 0..3 grades per query
			// feature 0 is an unrelated prior, feature 1 carries the grade with light noise.
			prior := rng.Float64()
			signal := grade + rng.NormFloat64()*0.15
			docs = append(docs, Doc{
				Key:   fmt.Sprintf("q%d-d%d", qi, di),
				Row:   []float64{prior, signal},
				Grade: grade,
			})
		}
		queries = append(queries, Query{ID: fmt.Sprintf("q%02d", qi), Docs: docs})
	}
	train, evalSet := Split(queries, 0.4)

	// The untrained baseline ranks by the unrelated prior feature, so it has no idea which
	// documents are relevant: a model trained to rank by feature 0.
	untrained := priorModel(t, train, 0)

	params := rank.DefaultParams()
	params.Rounds = 80
	out, ens, err := Gate(train, evalSet, 2, params, untrained, []int{10})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Improved {
		t.Fatalf("trained NDCG@10 %.4f did not beat untrained %.4f",
			out.Trained.MeanNDCG[10], out.Untrained.MeanNDCG[10])
	}
	if ens.NumTrees() == 0 {
		t.Fatal("gate returned an empty ensemble")
	}
	t.Logf("trained NDCG@10 %.4f over untrained %.4f (gain %.4f)",
		out.Trained.MeanNDCG[10], out.Untrained.MeanNDCG[10], out.GainNDCG10)
}

// priorModel trains a tiny model to rank by a single feature index, the stand-in for a
// cold-start prior ranker that ignores query relevance.
func priorModel(t *testing.T, queries []Query, featIdx int) *rank.Model {
	t.Helper()
	d := &rank.Dataset{NumFeatures: len(queries[0].Docs[0].Row)}
	for _, q := range queries {
		for _, doc := range q.Docs {
			d.Features = append(d.Features, doc.Row)
			// Label by the chosen feature so the model learns to order by it.
			d.Labels = append(d.Labels, doc.Row[featIdx])
		}
		d.Groups = append(d.Groups, len(q.Docs))
	}
	p := rank.DefaultParams()
	p.Rounds = 80
	return rank.Train(d, p).Compile()
}
