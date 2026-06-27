package search

import (
	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/sparse"
	"github.com/tamnd/tsumugi/vector"
)

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

// NodeBase is the global id of the shard's first document.
func (s *Shard) NodeBase() uint32 { return s.nodeBase }

// Close releases the shard's file mapping.
func (s *Shard) Close() error { return s.r.Close() }

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
func (s *Shard) retrieve(q Query) (lex, dense []scored, feats map[uint32][]float64) {
	feats = make(map[uint32][]float64)
	k := q.K
	if k < s.l0 {
		k = s.l0
	}
	if s.lex != nil && len(q.lexTerms()) > 0 {
		cands, err := s.lexSearch(q, k)
		if err == nil {
			for _, c := range cands {
				lex = append(lex, scored{docID: c.DocID, score: float64(c.Score)})
				if _, ok := feats[c.DocID]; !ok {
					feats[c.DocID] = s.featureRow(c.DocID)
				}
			}
		}
	}
	if s.sp != nil && len(q.Sparse) > 0 {
		for _, c := range s.sp.Search(q.Sparse, k) {
			lex = append(lex, scored{docID: c.DocID, score: float64(c.Score)})
			if _, ok := feats[c.DocID]; !ok {
				feats[c.DocID] = s.featureRow(c.DocID)
			}
		}
	}
	if s.vec != nil && len(q.Vector) > 0 {
		for _, c := range s.vec.Search(q.Vector, k, vector.DefaultEfSearch, vector.DefaultRerankDepth) {
			dense = append(dense, scored{docID: c.DocID, score: c.Score})
			if _, ok := feats[c.DocID]; !ok {
				feats[c.DocID] = s.featureRow(c.DocID)
			}
		}
	}
	return lex, dense, feats
}

// lexSearch runs the lexical plane over the query's analyzed term set, scoring with
// the broker's pushed-down collection-wide idf when the query carries one and the
// shard's local idf otherwise. The term set is the broker's pre-analyzed Terms when
// present, so the shard does not re-run the analysis chain on the fan-out path.
func (s *Shard) lexSearch(q Query, k int) ([]lexical.Candidate, error) {
	terms := q.lexTerms()
	if q.TermIDF != nil {
		return s.lex.SearchTermsWithIDF(terms, k, q.TermIDF)
	}
	return s.lex.SearchTerms(terms, k)
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

// Search runs the full cascade over this one shard and returns the model-ranked
// top-k as global hits. It is the standalone single-shard search path and the M8
// cascade wired end to end: retrieve on every plane, fuse and cut and rerank, and
// shift the local ids into the global space by the shard's node base.
func (s *Shard) Search(q Query) []Hit {
	lex, dense, feats := s.retrieve(q)
	lexIDs := localIDs(lex)
	denseIDs := localIDs(dense)
	// L1 reads the cheap matrix row; L2 reads the matrix row followed by the online
	// query-dependent features the extractor computes per survivor. The single-shard
	// path scores idf and the average body length against this shard's own counts,
	// the local statistics that are the collection statistics when there is one shard.
	ext := s.newOnline(q, q.TermIDF, s.localAvgBodyLen())
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
// supply it. avgBody is the average body length BM25 normalizes by.
func (s *Shard) newOnline(q Query, idfOf map[string]float64, avgBody float64) *onlineExtractor {
	if idfOf == nil && s.lex != nil {
		idfOf = s.localIDF(q.lexTerms())
	}
	return newOnlineExtractor(q, s.fwd, s.vec, idfOf, avgBody)
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

// localAvgBodyLen is the shard's average body length in tokens, the BM25 length
// normalizer for the single-shard path. It reads the token count the build recorded
// and divides by the document count; a shard with neither recorded falls back to no
// normalization.
func (s *Shard) localAvgBodyLen() float64 {
	if s.docCount == 0 {
		return 0
	}
	if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
		return v / float64(s.docCount)
	}
	return 0
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
