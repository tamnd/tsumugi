package search

import (
	"context"
	"errors"
	"fmt"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/sparse"
	"github.com/tamnd/tsumugi/vector"
)

// ErrSchemaMismatch is returned when a shard's feature region, or a model the broker
// ranks with, was built against a feature schema that does not match the one this
// build scores against. Refusing loudly at load turns a silent wrong-column misread,
// the failure that grows likely as a fleet of 100,000 shards is built over months
// against an evolving schema, into a startup error.
var ErrSchemaMismatch = errors.New("search: feature schema mismatch")

// DefaultL0 is the number of candidates each retrieval plane returns from a shard
// before fusion, the L0 width of the cascade.
const DefaultL0 = 1000

// Shard is one opened .tsumugi file ready to serve queries. It holds the reader and
// the parsed regions it found, the feature schema it scores against, and the cascade
// that turns retrieved candidates into a ranked top-k. A shard is read-only and safe
// for concurrent queries: every region reader is immutable and the cascade allocates
// its per-query state inside Score.
type Shard struct {
	r        *tsumugi.Reader
	nodeBase uint32
	docCount uint32

	lex  *lexical.Region
	vec  *vector.Region
	sp   *sparse.Region
	feat *feature.Region
	fwd  *forward.Region

	cols    []feature.Column
	cascade *rank.Cascade
	l0      int
}

// OpenShard opens a shard file and parses every region it carries, wiring them to
// the given cascade. The cascade holds the L1 linear cut and the L2 model the shard
// ranks with; pass the model trained offline. A shard missing a region simply skips
// that plane at query time.
func OpenShard(path string, cascade *rank.Cascade) (*Shard, error) {
	r, err := tsumugi.Open(path)
	if err != nil {
		return nil, err
	}
	s, err := newShard(r, cascade)
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	return s, nil
}

func newShard(r *tsumugi.Reader, cascade *rank.Cascade) (*Shard, error) {
	s := &Shard{
		r:        r,
		nodeBase: uint32(r.Header.NodeBase),
		docCount: r.DocCount(),
		cols:     feature.DefaultSchema(),
		cascade:  cascade,
		l0:       DefaultL0,
	}
	// The lexical region kind holds either a classic BM25 index or a learned-sparse
	// impact index, the two distinguished by the impact-postings flag, so the shard
	// opens it as whichever the flag says it is.
	if r.HasRegion(tsumugi.RegionLexical) {
		b, err := r.Region(tsumugi.RegionLexical)
		if err != nil {
			return nil, err
		}
		if r.Header.Has(tsumugi.FlagImpactPostings) {
			if s.sp, err = sparse.Open(b); err != nil {
				return nil, err
			}
		} else {
			if s.lex, err = lexical.Open(b); err != nil {
				return nil, err
			}
		}
	}
	if r.HasRegion(tsumugi.RegionVector) {
		b, err := r.Region(tsumugi.RegionVector)
		if err != nil {
			return nil, err
		}
		if s.vec, err = vector.Open(b); err != nil {
			return nil, err
		}
	}
	if r.HasRegion(tsumugi.RegionFeature) {
		b, err := r.Region(tsumugi.RegionFeature)
		if err != nil {
			return nil, err
		}
		if s.feat, err = feature.Open(b); err != nil {
			return nil, err
		}
		// Refuse a shard whose feature matrix was built against a different schema than
		// this build scores against. The model is trained on a fixed column order; a
		// shard that reorders, retypes, or drops a column would feed the model a row it
		// reads as different signals, a silent scoring corruption, so it is rejected here.
		if v, h := s.feat.SchemaVersion(), s.feat.SchemaHash(); v != feature.SchemaVersion || h != feature.DefaultSchemaHash() {
			return nil, fmt.Errorf("%w: shard feature region is schema v%d hash %016x, this build expects v%d hash %016x",
				ErrSchemaMismatch, v, h, feature.SchemaVersion, feature.DefaultSchemaHash())
		}
	}
	// The forward region holds the candidate text the online L2 features decode:
	// the title, body, and url a BM25F or proximity feature scores against. A shard
	// without it serves the matrix features but extracts no online text features.
	if r.HasRegion(tsumugi.RegionForward) {
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			return nil, err
		}
		if s.fwd, err = forward.Open(b); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// DocCount is the number of documents in the shard.
func (s *Shard) DocCount() uint32 { return s.docCount }

// MaxStaticRank is the highest composite static rank among the shard's documents, the
// per-shard static-rank summary the broker's degradation uses to order shards by how
// much dropping them costs. The composite static rank is the query-independent prior
// (doc 07), so the shard's maximum is the best a top-k winner it could hold would score
// before any query signal, which is why a low maximum marks a shard unlikely to hold a
// winner. A shard with no feature region carries no static-rank signal and reports zero.
func (s *Shard) MaxStaticRank() float64 {
	if s.feat == nil {
		return 0
	}
	var max float64
	first := true
	for id := uint32(0); id < s.docCount; id++ {
		v, ok := s.feat.Value(id, feature.FeatStaticRank)
		if !ok {
			continue
		}
		if first || v > max {
			max, first = v, false
		}
	}
	return max
}

// NodeBase is the global id of the shard's first document.
func (s *Shard) NodeBase() uint32 { return s.nodeBase }

// AnalyzerHash returns the analyzer_hash the shard was built with and whether it was
// recorded. A broker compares it against its query-side analyzer to refuse a shard built
// with an incompatible analysis chain, the consistency guard a query and a document must
// share an analyzer to match.
func (s *Shard) AnalyzerHash() (uint64, bool) { return s.r.AnalyzerHash() }

// VectorDim returns the dense-plane input dimension this shard's vector region expects
// and whether the shard carries one. A broker building a dense query encoder reads it to
// produce a query vector of the width the region rotates and quantizes from; a shard with
// no vector region reports false and the broker leaves the dense plane off.
func (s *Shard) VectorDim() (int, bool) {
	if s.vec == nil {
		return 0, false
	}
	return s.vec.Dim(), true
}

// Close releases the shard's forward decoder and file mapping.
func (s *Shard) Close() error {
	if s.fwd != nil {
		s.fwd.Close()
	}
	return s.r.Close()
}

// featureRow builds the L2 feature vector for a local document id by reading the
// shard's feature matrix in schema order. A shard without a feature region scores
// every document on an all-zero row, which a model trained on the same schema reads
// as the absence of every signal.
func (s *Shard) featureRow(localID uint32) []float64 {
	row := make([]float64, len(s.cols))
	if s.feat == nil {
		return row
	}
	for i, c := range s.cols {
		if v, ok := s.feat.Value(localID, c.ID); ok {
			row[i] = v
		}
	}
	return row
}

// retrieve runs the shard's retrieval planes and returns the ranked candidate lists
// in local document ids together with the feature rows of every candidate. It is the
// shared core of Search and the broker fan-out: Search runs the cascade locally,
// while the broker gathers retrievals from many shards and runs one global rerank.
// retrievePreemptStride is how often the retrieval checks the deadline while gathering
// a plane's candidates: once every this-many-plus-one rows. The check is a context Err
// read, cheap but not free on a cancellable context (it touches the context's mutex), so
// it runs against a stride rather than every row, often enough to abandon a long plane
// soon after the budget runs out and rare enough to cost nothing on the common path where
// the budget holds. A power-of-two-minus-one so the test is a single bitwise and.
const retrievePreemptStride = 255

// retrieve runs the shard's retrieval planes and returns the candidates each produced
// with their feature rows, and whether it finished before the query's deadline. The
// deadline reaches the shard through ctx so a goroutine the broker dispatched before the
// budget ran out does not keep scanning after it: at each plane boundary, and on a stride
// while gathering a plane's rows, the shard checks whether ctx is done and abandons the
// rest of the work if it is. An abandoned retrieval returns completed=false, and the
// broker drops the shard from the merge rather than serving its half-built candidate set,
// so the partial answer stays honest (the shard is rolled up as not responded, exactly as
// a slow shard the collection stopped waiting for) and no CPU is spent on a plane whose
// result the collection has already stopped waiting for. The dense plane is the most
// expensive, so abandoning at the lexical-to-dense boundary when the deadline has already
// passed is where this saves the most. The returned slices on an abandoned retrieval are
// an arbitrary partial and must be discarded; only the completed flag is meaningful then.
func (s *Shard) retrieve(ctx context.Context, q Query) (lex, dense []scored, feats map[uint32][]float64, completed bool) {
	feats = make(map[uint32][]float64)
	l0 := s.l0
	if q.L0 > 0 {
		l0 = q.L0
	}
	k := q.K
	if k < l0 {
		k = l0
	}
	if ctx.Err() != nil {
		return lex, dense, feats, false
	}
	if s.lex != nil && len(q.lexTerms()) > 0 {
		cands, err := s.lexSearch(ctx, q, k)
		if errors.Is(err, context.Canceled) {
			// The deadline passed inside the WAND traversal, which abandoned its
			// postings walk and handed back a partial. Drop the whole shard.
			return lex, dense, feats, false
		}
		if err == nil {
			for i, c := range cands {
				if (i&retrievePreemptStride) == 0 && ctx.Err() != nil {
					return lex, dense, feats, false
				}
				lex = append(lex, scored{docID: c.DocID, score: float64(c.Score)})
				if _, ok := feats[c.DocID]; !ok {
					feats[c.DocID] = s.featureRow(c.DocID)
				}
			}
		}
	}
	if ctx.Err() != nil {
		return lex, dense, feats, false
	}
	if s.sp != nil && len(q.Sparse) > 0 {
		cands, completed := s.sp.SearchCtx(ctx, q.Sparse, k)
		if !completed {
			return lex, dense, feats, false
		}
		for i, c := range cands {
			if (i&retrievePreemptStride) == 0 && ctx.Err() != nil {
				return lex, dense, feats, false
			}
			lex = append(lex, scored{docID: c.DocID, score: float64(c.Score)})
			if _, ok := feats[c.DocID]; !ok {
				feats[c.DocID] = s.featureRow(c.DocID)
			}
		}
	}
	if ctx.Err() != nil {
		return lex, dense, feats, false
	}
	if s.vec != nil && len(q.Vector) > 0 {
		cands, completed := s.vec.SearchCtx(ctx, q.Vector, k, vector.DefaultEfSearch, vector.DefaultRerankDepth)
		if !completed {
			return lex, dense, feats, false
		}
		for i, c := range cands {
			if (i&retrievePreemptStride) == 0 && ctx.Err() != nil {
				return lex, dense, feats, false
			}
			dense = append(dense, scored{docID: c.DocID, score: c.Score})
			if _, ok := feats[c.DocID]; !ok {
				feats[c.DocID] = s.featureRow(c.DocID)
			}
		}
	}
	return lex, dense, feats, true
}

// lexSearch runs the lexical plane over the query's analyzed term set, scoring with
// the broker's pushed-down collection-wide idf when the query carries one and the
// shard's local idf otherwise. The term set is the broker's pre-analyzed Terms when
// present, so the shard does not re-run the analysis chain on the fan-out path.
func (s *Shard) lexSearch(ctx context.Context, q Query, k int) ([]lexical.Candidate, error) {
	terms := q.lexTerms()
	// An impact-ordered region scores query-term coverage weighted by the static rank the
	// postings are ordered by; it takes no idf, so the broker's pushed-down idf does not
	// apply and the impact traversal serves it directly.
	if s.lex.IsImpact() {
		return s.lex.SearchImpactTermsCtx(ctx, terms, k)
	}
	if q.TermIDF != nil {
		return s.lex.SearchTermsWithIDFCtx(ctx, terms, k, q.TermIDF)
	}
	return s.lex.SearchTermsCtx(ctx, terms, k)
}

// LexDocFreqs returns the local document frequency of each query term this shard's
// lexical region holds, the first phase of the broker's distributed exact-idf scoring.
// A shard with no lexical region contributes nothing.
func (s *Shard) LexDocFreqs(terms []string) map[string]uint32 {
	if s.lex == nil {
		return nil
	}
	return s.lex.DocFreqsTerms(terms)
}

// ForEachTerm calls fn for every term in this shard's lexical dictionary with its
// local document frequency. The broker builds the collection-wide spell-correction
// dictionary by merging this enumeration across the fleet's shards. A shard with no
// lexical region contributes nothing.
func (s *Shard) ForEachTerm(fn func(term string, docFreq uint32)) {
	if s.lex == nil {
		return
	}
	s.lex.ForEachTerm(fn)
}

// Search runs the full cascade over this one shard and returns the model-ranked
// top-k as global hits. It is the standalone single-shard search path and the M8
// cascade wired end to end: retrieve on every plane, fuse and cut and rerank, and
// shift the local ids into the global space by the shard's node base.
func (s *Shard) Search(q Query) []Hit {
	lex, dense, feats, _ := s.retrieve(context.Background(), q)
	lexIDs := localIDs(lex)
	denseIDs := localIDs(dense)
	// L1 reads the cheap matrix row; L2 reads the matrix row followed by the online
	// query-dependent features the extractor computes per survivor. The single-shard
	// path scores idf and the average body length against this shard's own counts,
	// the local statistics that are the collection statistics when there is one shard.
	ext := s.newOnline(q, q.TermIDF, s.localAvgFieldLen())
	l1feat := func(id uint32) []float64 { return feats[id] }
	l2feat := func(id uint32) []float64 { return s.l2Row(feats[id], ext, id) }
	cands := s.cascade.Rank(lexIDs, denseIDs, l1feat, l2feat, q.K)
	hits := make([]Hit, len(cands))
	for i, c := range cands {
		hits[i] = Hit{DocID: s.nodeBase + c.DocID, Score: c.Score}
	}
	return hits
}

// newOnline builds the per-query online feature extractor over this shard's
// forward and vector regions. idfOf is the per-term idf to score BM25 with, the
// broker's pushed-down collection idf on the fan-out path or the query's own
// override; a nil map falls back to the shard-local idf when the lexical region can
// supply it. avgField is the per-field average length BM25 normalizes each field by,
// the fleet averages on the fan-out path or this shard's own on the single-shard path.
func (s *Shard) newOnline(q Query, idfOf map[string]float64, avgField [4]float64) *onlineExtractor {
	if idfOf == nil && s.lex != nil {
		idfOf = s.localIDF(q.lexTerms())
	}
	return newOnlineExtractor(q, s.fwd, s.vec, idfOf, avgField)
}

// l2Row assembles the full L2 feature vector for a candidate: the query-independent
// matrix row followed by the online query-dependent features, the concatenated
// width the L2 model is trained against. base is the candidate's matrix row, reused
// without copying since the online features are appended onto a fresh backing array.
func (s *Shard) l2Row(base []float64, ext *onlineExtractor, localID uint32) []float64 {
	online := ext.features(localID)
	row := make([]float64, 0, len(base)+len(online))
	row = append(row, base...)
	row = append(row, online...)
	return row
}

// localIDF computes the shard-local idf of each query term from this shard's own
// document count and the term's local document frequency, the idf the single-shard
// path scores with when no collection-wide idf is pushed down.
func (s *Shard) localIDF(terms []string) map[string]float64 {
	if s.lex == nil || len(terms) == 0 {
		return nil
	}
	df := s.lex.DocFreqsTerms(terms)
	if len(df) == 0 {
		return nil
	}
	out := make(map[string]float64, len(df))
	for t, f := range df {
		out[t] = lexical.IDF(uint64(s.docCount), uint64(f))
	}
	return out
}

// localAvgFieldLen is the shard's per-field average length in tokens, the BM25 length
// normalizer for the single-shard path where this shard's own statistics are the
// collection statistics. It reads the per-field token sums the build recorded and
// divides by the document count, indexed in the online extractor's field order
// (title, body, url, anchor). The body falls back to the token_count sum when the per-field
// stat is absent, the way a shard built before the per-field sums normalized it;
// title, url, and anchor fall back to zero, leaving those fields unnormalized as they were.
func (s *Shard) localAvgFieldLen() [4]float64 {
	var avg [4]float64
	if s.docCount == 0 {
		return avg
	}
	n := float64(s.docCount)
	if v, ok := s.r.Stat(tsumugi.StatBodyTokenCount); ok {
		avg[fBody] = v / n
	} else if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
		avg[fBody] = v / n
	}
	if v, ok := s.r.Stat(tsumugi.StatTitleTokenCount); ok {
		avg[fTitle] = v / n
	}
	if v, ok := s.r.Stat(tsumugi.StatURLTokenCount); ok {
		avg[fURL] = v / n
	}
	if v, ok := s.r.Stat(tsumugi.StatAnchorTokenCount); ok {
		avg[fAnchor] = v / n
	}
	return avg
}

// localIDs drops the scores and returns the document ids in list order, the shape
// the cascade's fusion takes.
func localIDs(ss []scored) []uint32 {
	ids := make([]uint32, len(ss))
	for i, s := range ss {
		ids[i] = s.docID
	}
	return ids
}
