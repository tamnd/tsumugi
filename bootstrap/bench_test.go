package bootstrap

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/tamnd/tsumugi/rank"
)

// benchQueries builds a synthetic judged pool of the given shape, the offline training
// input the bootstrap fits a model over, so the benchmarks measure the train-and-evaluate
// cost without standing up a collection.
func benchQueries(nQueries, perQuery, nFeat int) []Query {
	rng := rand.New(rand.NewSource(11))
	qs := make([]Query, nQueries)
	for qi := range qs {
		docs := make([]Doc, perQuery)
		for di := range docs {
			grade := float64(di % 4)
			row := make([]float64, nFeat)
			for f := range row {
				row[f] = rng.Float64()
			}
			row[1] = grade + rng.NormFloat64()*0.2 // one feature carries the grade
			docs[di] = Doc{Key: fmt.Sprintf("q%d-d%d", qi, di), Row: row, Grade: grade}
		}
		qs[qi] = Query{ID: fmt.Sprintf("q%04d", qi), Docs: docs}
	}
	return qs
}

// BenchmarkGate measures stage three plus the evaluation: train a LambdaMART model on the
// pool and score the trained and untrained models over the held-out split. It is the
// offline cost of one bootstrap round, the work the train command does per run.
func BenchmarkGate(b *testing.B) {
	qs := benchQueries(200, 50, 47)
	train, evalSet := Split(qs, 0.3)
	untrained := benchPriorModel(train, 47)
	params := rank.DefaultParams()
	params.Rounds = 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := Gate(train, evalSet, 47, params, untrained, []int{10}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRunModel measures scoring a model over a judged pool into a run, the per-model
// step the gate runs twice and the harness consumes.
func BenchmarkRunModel(b *testing.B) {
	qs := benchQueries(200, 50, 47)
	m := benchPriorModel(qs, 47)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RunModel(qs, m)
	}
}

func benchPriorModel(queries []Query, nFeat int) *rank.Model {
	d := &rank.Dataset{NumFeatures: nFeat}
	for _, q := range queries {
		for _, doc := range q.Docs {
			d.Features = append(d.Features, doc.Row)
			d.Labels = append(d.Labels, doc.Row[0])
		}
		d.Groups = append(d.Groups, len(q.Docs))
	}
	p := rank.DefaultParams()
	p.Rounds = 40
	return rank.Train(d, p).Compile()
}
