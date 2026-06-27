package search

import "github.com/tamnd/tsumugi"

// GlobalStats are the fleet-wide collection statistics, summed across every shard.
// A single shard's statistics describe only its slice of the collection, so scoring
// that needs collection-level terms, the average document length BM25 normalizes by
// above all, has to read the fleet number rather than any one shard's. The broker
// computes these once over its shards and exposes them so the global rerank scores
// against the whole collection, not a fragment.
type GlobalStats struct {
	DocCount   uint64
	TokenCount float64
	AvgDocLen  float64

	// AvgFieldLen is the fleet average length in tokens of each field, indexed by the
	// online extractor's field order (title, body, url). The per-field BM25F length
	// normalizer divides a candidate's own field length by these, so a term that
	// appears in a short title is normalized against the fleet's average title rather
	// than against the conflated average document length. A zero entry means no fleet
	// average is known for that field, and its BM25 falls back to no length
	// normalization. These are what make the merged top-k exact for the field-weighted
	// score: the candidate's field lengths come from its own matrix columns and the
	// denominators are fleet-wide, so a document's BM25F is identical regardless of the
	// shard it was retrieved from.
	AvgFieldLen [3]float64
}

// computeGlobalStats sums the per-shard statistics into fleet-wide ones. It reads the
// document and token counts each shard recorded at build time and derives the
// collection-wide average document length, the term every length normalization in the
// lexical plane depends on, and the per-field average lengths the online BM25F
// normalizes each field by.
func computeGlobalStats(shards []*Shard) GlobalStats {
	var gs GlobalStats
	var titleTok, bodyTok, urlTok float64
	for _, s := range shards {
		gs.DocCount += uint64(s.docCount)
		if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
			gs.TokenCount += v
		}
		// The body field falls back to token_count (title+body) for a shard built before
		// the per-field sums existed, so an older shard still normalizes the body the way
		// it did before; title and url fall back to zero, the no-normalization behavior
		// those fields had before this stat.
		if v, ok := s.r.Stat(tsumugi.StatBodyTokenCount); ok {
			bodyTok += v
		} else if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
			bodyTok += v
		}
		if v, ok := s.r.Stat(tsumugi.StatTitleTokenCount); ok {
			titleTok += v
		}
		if v, ok := s.r.Stat(tsumugi.StatURLTokenCount); ok {
			urlTok += v
		}
	}
	if gs.DocCount > 0 {
		n := float64(gs.DocCount)
		gs.AvgDocLen = gs.TokenCount / n
		gs.AvgFieldLen[fTitle] = titleTok / n
		gs.AvgFieldLen[fBody] = bodyTok / n
		gs.AvgFieldLen[fURL] = urlTok / n
	}
	return gs
}
