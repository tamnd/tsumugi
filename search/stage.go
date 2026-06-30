package search

import (
	"context"
	"sort"
	"time"

	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/vector"
)

// StageTimings is the per-stage decomposition of one query's wall time through the
// cascade, the breakdown doc 14 reports per-stage p99 against alongside the
// end-to-end p99. Each field is the time that one stage spent, so a query's total is
// the sum of the fields plus the small fixed cost of moving between them, and the
// stage that dominates the budget is the one to spend an optimization on. The lines
// match doc 14's per-stage budget table: lexical retrieval, dense recall, the linear
// cut, the L2 feature extraction over the survivors, and the L2 model scoring.
type StageTimings struct {
	// L0Lexical is the lexical and learned-sparse retrieval, the term-driven planes
	// that walk postings to gather the first candidate pool.
	L0Lexical time.Duration
	// L0Dense is the dense recall over the vector region, the optional plane doc 14
	// budgets separately because its HNSW walk is the second-most-expensive stage.
	L0Dense time.Duration
	// Fuse is the reciprocal-rank fusion of the plane lists and the cap to L0Max, the
	// cheap glue between retrieval and the linear cut.
	Fuse time.Duration
	// L1Cut is the linear scoring of the fused pool and the cut to L1Keep, the stage
	// that shrinks the expensive L2 work to a few hundred survivors.
	L1Cut time.Duration
	// L2Feature is the online feature extraction over the L1 survivors, the
	// query-dependent BM25F and proximity features assembled onto each survivor's
	// matrix row. It is timed apart from scoring because the extraction reads the
	// forward store and the scoring reads only the assembled row.
	L2Feature time.Duration
	// L2Score is the L2 model inference over the survivors' feature rows, the
	// LambdaMART QuickScorer pass that produces the final ordinal score.
	L2Score time.Duration
}

// Total is the sum of the stage times, the wall time the decomposition accounts for.
func (t StageTimings) Total() time.Duration {
	return t.L0Lexical + t.L0Dense + t.Fuse + t.L1Cut + t.L2Feature + t.L2Score
}

// SearchStaged runs the same cascade as Search and returns the same hits, but
// decomposed into per-stage timings, the measurement path behind the per-stage
// latency breakdown benchmark. It mirrors retrieve and Cascade.Rank stage for stage
// rather than calling them, so that each stage boundary can be timed: the lexical
// and sparse planes, the dense plane, the fusion, the linear cut, the online feature
// extraction over the survivors, and the model scoring. The result is identical to
// Search on the same query (the benchmark asserts it), so the breakdown is of the
// production pipeline, not of a parallel one. It runs against a background context
// with no deadline, so the retrieval's preemption checks never fire: this is an
// offline measurement of full-quality work, not a budgeted serve.
func (s *Shard) SearchStaged(q Query) ([]Hit, StageTimings) {
	var t StageTimings
	ctx := context.Background()

	l0 := s.l0
	if q.L0 > 0 {
		l0 = q.L0
	}
	k := q.K
	if k < l0 {
		k = l0
	}

	feats := make(map[uint32][]float64)
	var lex, dense []scored

	// L0 lexical and learned-sparse: both term-driven planes feed the lexical pool,
	// the way retrieve folds the sparse plane in alongside the classic lexical one.
	start := time.Now()
	if s.lex != nil && len(q.lexTerms()) > 0 {
		if cands, err := s.lexSearch(ctx, q, k); err == nil {
			for _, c := range cands {
				lex = append(lex, scored{docID: c.DocID, score: float64(c.Score)})
				if _, ok := feats[c.DocID]; !ok {
					feats[c.DocID] = s.featureRow(c.DocID)
				}
			}
		}
	}
	if s.sp != nil && len(q.Sparse) > 0 {
		if cands, completed := s.sp.SearchCtx(ctx, q.Sparse, k); completed {
			for _, c := range cands {
				lex = append(lex, scored{docID: c.DocID, score: float64(c.Score)})
				if _, ok := feats[c.DocID]; !ok {
					feats[c.DocID] = s.featureRow(c.DocID)
				}
			}
		}
	}
	t.L0Lexical = time.Since(start)

	// L0 dense: the vector plane's HNSW walk, the optional second retrieval source.
	start = time.Now()
	if s.vec != nil && len(q.Vector) > 0 {
		if cands, completed := s.vec.SearchCtx(ctx, q.Vector, k, vector.DefaultEfSearch, vector.DefaultRerankDepth); completed {
			for _, c := range cands {
				dense = append(dense, scored{docID: c.DocID, score: c.Score})
				if _, ok := feats[c.DocID]; !ok {
					feats[c.DocID] = s.featureRow(c.DocID)
				}
			}
		}
	}
	t.L0Dense = time.Since(start)

	lexIDs := localIDs(lex)
	denseIDs := localIDs(dense)
	c := s.cascade

	// Fuse: reciprocal-rank fusion of the plane lists, capped to L0Max.
	start = time.Now()
	fused := rank.RRF(c.K, lexIDs, denseIDs)
	if len(fused) > c.L0Max {
		fused = fused[:c.L0Max]
	}
	t.Fuse = time.Since(start)

	// L1 cut: the linear score over the fused pool, narrowed to L1Keep survivors.
	l1feat := func(id uint32) []float64 { return feats[id] }
	start = time.Now()
	kept := c.L1.Cut(fused, l1feat, c.L1Keep)
	t.L1Cut = time.Since(start)

	// L2 feature extraction: assemble each survivor's full feature row, the matrix
	// columns followed by the online query-dependent features. Timed apart from
	// scoring so the forward-store reads are attributed to extraction, not inference.
	ext := s.newOnline(q, q.TermIDF, s.localAvgFieldLen())
	start = time.Now()
	rows := make([][]float64, len(kept))
	for i, cd := range kept {
		rows[i] = s.l2Row(feats[cd.DocID], ext, cd.DocID)
	}
	t.L2Feature = time.Since(start)

	// L2 scoring: the model's ordinal score over each assembled row, then the top-k
	// sort. This is the LambdaMART QuickScorer pass.
	start = time.Now()
	scoredCands := make([]rank.Candidate, len(kept))
	for i, cd := range kept {
		scoredCands[i] = rank.Candidate{DocID: cd.DocID, Score: c.L2.Score(rows[i])}
	}
	// The cascade's rerank orders by ordinal score descending, ties to the smaller
	// docID; match it exactly so the staged hits equal Search's.
	sort.Slice(scoredCands, func(i, j int) bool {
		if scoredCands[i].Score != scoredCands[j].Score {
			return scoredCands[i].Score > scoredCands[j].Score
		}
		return scoredCands[i].DocID < scoredCands[j].DocID
	})
	if q.K > 0 && len(scoredCands) > q.K {
		scoredCands = scoredCands[:q.K]
	}
	t.L2Score = time.Since(start)

	hits := make([]Hit, len(scoredCands))
	for i, cd := range scoredCands {
		hits[i] = Hit{DocID: s.nodeBase + cd.DocID, Score: cd.Score}
	}
	return hits, t
}
