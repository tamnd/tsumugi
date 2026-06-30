// Package bootstrap is the offline training bootstrap doc 09 specifies: it turns a
// collection and a set of training queries into a trained LambdaMART model with no
// clicks at all. The three stages are the spec's. Stage one retrieves a candidate pool
// per query through the same cascade front the serving path runs, so each candidate
// carries the exact L2 feature row the model scores at serve time, and the pool's L1
// survivors are the relevant documents alongside the hard negatives, the high-retrieval
// low-relevance near-misses the ranker most needs to learn to push down. Stage two grades
// every pooled candidate with an UMBRELA-style judge into a 0-to-3 graded label. Stage
// three groups the judged candidates by query and fits a LambdaMART model over their
// feature rows, optimizing NDCG against the graded labels.
//
// The package is the orchestration seam, not the pieces: the candidate source is the
// search broker through its SearchFeatures method, the judge is an eval.Judge (the real
// LLM judge or the deterministic lexical one), the passage text comes from a caller-
// supplied lookup over the forward store, and the model and metrics are the rank and eval
// packages. Holding it here keeps the search, rank, and eval packages unaware of each
// other while the bootstrap composes all three.
package bootstrap

import (
	"context"
	"fmt"
	"sort"

	"github.com/tamnd/tsumugi/eval"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// CandidateSource yields, for one query, the retrieval candidates and the exact L2
// feature row each one would be scored on. The search broker satisfies it through
// SearchFeatures, so the bootstrap labels and trains over the rows the serving path
// builds. It is an interface so the bootstrap is testable without standing up a broker.
type CandidateSource interface {
	SearchFeatures(ctx context.Context, q search.Query) []search.FeatureHit
}

// QueryFunc turns a raw query string into the search.Query the candidate source runs,
// the seam that lets the caller plug in the query-understanding pipeline without the
// bootstrap depending on the query package. K is the candidate pool width the caller
// sets on the returned query.
type QueryFunc func(raw string) search.Query

// PassageFunc maps a candidate's global document id to the passage the judge reads and
// the stable key the run and qrels record it under, returning false when the id has no
// text to judge. The passage's Doc field is that stable key, the cross-shard document id
// the harness compares runs and labels against, not the in-memory global id which is not
// stable across builds.
type PassageFunc func(id uint32) (eval.Passage, bool)

// Doc is one judged candidate: its global id, the stable key it is recorded under, the
// feature row the model scores, and the graded relevance label the judge assigned.
type Doc struct {
	DocID uint32
	Key   string
	Row   []float64
	Grade float64
}

// Query is one training query with its judged candidate pool, the unit LambdaMART groups
// its lambdas within.
type Query struct {
	ID   string
	Text string
	Docs []Doc
}

// Pool is the judged training data: the per-query candidate pools and the feature width
// every row carries, the shape stages one and two produce and stage three consumes.
type Pool struct {
	Queries     []Query
	NumFeatures int
}

// Build runs stages one and two: for each query it retrieves the candidate pool and its
// feature rows through the source, then grades every candidate whose passage the lookup
// resolves, returning the judged pool. Queries are processed in sorted id order so the
// pool is deterministic given a deterministic judge. A query whose pool is empty after
// the passage and judge filters is dropped rather than carried as an empty group, because
// LambdaMART has nothing to learn from a query with no candidates. A judge error aborts
// the build rather than silently dropping the pair, so a broken judge is caught, not
// papered over with a missing label.
func Build(ctx context.Context, src CandidateSource, j eval.Judge, qf QueryFunc, queries map[string]string, passage PassageFunc) (*Pool, error) {
	p := &Pool{}
	for _, qid := range sortedKeys(queries) {
		text := queries[qid]
		sq := qf(text)
		hits := src.SearchFeatures(ctx, sq)
		var docs []Doc
		for _, h := range hits {
			pass, ok := passage(h.DocID)
			if !ok {
				continue
			}
			g, err := j.Grade(ctx, text, pass)
			if err != nil {
				return nil, fmt.Errorf("judge query %q doc %d: %w", qid, h.DocID, err)
			}
			if p.NumFeatures == 0 {
				p.NumFeatures = len(h.Row)
			}
			if len(h.Row) != p.NumFeatures {
				return nil, fmt.Errorf("query %q doc %d: feature row width %d, want %d",
					qid, h.DocID, len(h.Row), p.NumFeatures)
			}
			docs = append(docs, Doc{DocID: h.DocID, Key: pass.Doc, Row: h.Row, Grade: float64(g)})
		}
		if len(docs) == 0 {
			continue
		}
		p.Queries = append(p.Queries, Query{ID: qid, Text: text, Docs: docs})
	}
	return p, nil
}

// Dataset flattens the pool into the rank.Dataset stage three trains over: every
// candidate's feature row becomes a training row, its grade the label, and each query a
// group so the lambdas are computed within the query the way NDCG is. The row order
// follows the query order so a group's rows are contiguous, which is the layout the
// trainer's grouping assumes.
func (p *Pool) Dataset() *rank.Dataset {
	d := &rank.Dataset{NumFeatures: p.NumFeatures}
	for _, q := range p.Queries {
		for _, doc := range q.Docs {
			d.Features = append(d.Features, doc.Row)
			d.Labels = append(d.Labels, doc.Grade)
		}
		d.Groups = append(d.Groups, len(q.Docs))
	}
	return d
}

// Qrels turns the judged grades into the graded relevance labels the harness scores
// against, the same labels stage three trained on so the evaluation measures the model
// against the judgments it learned from, with the gold-set cross-check keeping those
// judgments honest. A grade of zero is still recorded, because an explicit irrelevant
// label is a judgment the harness uses, distinct from an unjudged pair it assumes zero.
func (p *Pool) Qrels() eval.Qrels {
	qr := make(eval.Qrels, len(p.Queries))
	for _, q := range p.Queries {
		m := make(map[string]float64, len(q.Docs))
		for _, doc := range q.Docs {
			m[doc.Key] = doc.Grade
		}
		qr[q.ID] = m
	}
	return qr
}

// RunModel scores every candidate's feature row with a model and records the per-query
// ranked list, the run the harness scores. It is how a model's order over the judged
// pool is turned into a run: the same candidate set every model ranks, so a model-to-
// model comparison is over identical candidates and identical labels and the only thing
// that varies is the model, which is what makes the trained-versus-untrained gate a fair
// test of the model rather than of the candidate set.
func RunModel(queries []Query, m *rank.Model) eval.Run {
	run := make(eval.Run, len(queries))
	for _, q := range queries {
		docs := make([]eval.RankedDoc, len(q.Docs))
		for i, doc := range q.Docs {
			docs[i] = eval.RankedDoc{Doc: doc.Key, Score: m.Score(doc.Row)}
		}
		run[q.ID] = docs
	}
	return run
}

// Split partitions the queries into a training set and an evaluation set, holding out the
// evalFrac fraction with the smallest stable hash of their query id. Ordering by the hash
// rather than thresholding it makes the held-out fraction exact rather than at the mercy
// of how the hashes happen to cluster, which matters for small query sets where a per-id
// threshold can route almost everything to one side. The split is reproducible (the hash
// is a pure function of the id) and disjoint (a query lands on exactly one side), and the
// evaluation set is held out from training so the gate measures generalization to unseen
// queries, not memorization of the training pairs, the spec's disjoint-held-out-query-set
// requirement. With at least two queries each side gets at least one.
func Split(queries []Query, evalFrac float64) (train, evalSet []Query) {
	if evalFrac < 0 {
		evalFrac = 0
	}
	if evalFrac > 1 {
		evalFrac = 1
	}
	order := make([]int, len(queries))
	for i := range order {
		order[i] = i
	}
	// Sort by hash, breaking ties by id so the order is total and reproducible.
	sort.Slice(order, func(a, b int) bool {
		ha, hb := splitHash(queries[order[a]].ID), splitHash(queries[order[b]].ID)
		if ha != hb {
			return ha < hb
		}
		return queries[order[a]].ID < queries[order[b]].ID
	})
	nEval := int(evalFrac*float64(len(queries)) + 0.5)
	if len(queries) >= 2 {
		if nEval < 1 {
			nEval = 1
		}
		if nEval > len(queries)-1 {
			nEval = len(queries) - 1
		}
	}
	evalPick := make(map[int]bool, nEval)
	for i := 0; i < nEval; i++ {
		evalPick[order[i]] = true
	}
	// Walk the queries in their original order so each side keeps the input order, which
	// keeps a group's rows contiguous for the trainer and the output readable.
	for i, q := range queries {
		if evalPick[i] {
			evalSet = append(evalSet, q)
		} else {
			train = append(train, q)
		}
	}
	return train, evalSet
}

// Outcome is the M9 gate result: the trained model's and the untrained baseline's reports
// over the held-out evaluation set, the NDCG@10 gain of trained over untrained, and
// whether the model improved, the number the milestone gates on.
type Outcome struct {
	Trained    eval.Report
	Untrained  eval.Report
	GainNDCG10 float64
	Improved   bool
}

// Gate is stage three plus the M9 evaluation: it trains a LambdaMART model on the training
// pool, scores both the trained model and the untrained baseline over the held-out
// evaluation pool against the judged labels, and reports whether the trained model
// improved NDCG@10, the milestone's gate. The trained ensemble is stamped with the feature
// schema so the serving path accepts it, and it is returned alongside the outcome so the
// caller can persist the model it just gated. The untrained baseline is supplied by the
// caller, the cold-start prior model the serve path ranks with before judgments exist, so
// the gate measures what training added over that prior.
func Gate(train, evalSet []Query, numFeatures int, params rank.Params, untrained *rank.Model, cutoffs []int) (*Outcome, *rank.Ensemble, error) {
	if len(train) == 0 {
		return nil, nil, fmt.Errorf("no training queries")
	}
	if len(evalSet) == 0 {
		return nil, nil, fmt.Errorf("no evaluation queries")
	}
	d := (&Pool{Queries: train, NumFeatures: numFeatures}).Dataset()
	if len(d.Features) == 0 {
		return nil, nil, fmt.Errorf("no training rows")
	}
	ens := rank.Train(d, params)
	ens.SetSchema(feature.SchemaVersion, feature.DefaultSchemaHash())
	trained := ens.Compile()

	qr := (&Pool{Queries: evalSet, NumFeatures: numFeatures}).Qrels()
	trainedRep := eval.Evaluate(RunModel(evalSet, trained), qr, cutoffs, nil)
	untrainedRep := eval.Evaluate(RunModel(evalSet, untrained), qr, cutoffs, nil)
	gain := trainedRep.MeanNDCG[10] - untrainedRep.MeanNDCG[10]
	return &Outcome{
		Trained:    trainedRep,
		Untrained:  untrainedRep,
		GainNDCG10: gain,
		Improved:   gain > 0,
	}, ens, nil
}

// splitHash is a stable 64-bit FNV-1a over a query id, the hash Split routes on. It is
// inlined rather than pulled from hash/fnv so the split needs no allocation per query and
// stays a pure function of the id, which is what makes the partition reproducible.
func splitHash(s string) uint64 {
	const (
		offset = 1469598103934665603
		prime  = 1099511628211
	)
	h := uint64(offset)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime
	}
	return h
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
