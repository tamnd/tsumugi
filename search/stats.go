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
	// online extractor's field order (title, body, url, anchor). The per-field BM25F length
	// normalizer divides a candidate's own field length by these, so a term that
	// appears in a short title is normalized against the fleet's average title rather
	// than against the conflated average document length. A zero entry means no fleet
	// average is known for that field, and its BM25 falls back to no length
	// normalization. These are what make the merged top-k exact for the field-weighted
	// score: the candidate's field lengths come from its own matrix columns and the
	// denominators are fleet-wide, so a document's BM25F is identical regardless of the
	// shard it was retrieved from.
	AvgFieldLen [4]float64
}

// statSums holds the raw additive accumulators the fleet statistics derive from: the
// document count, the total token count, and the per-field token totals. Keeping the raw
// sums rather than only the derived averages is what makes an incremental publish exact:
// folding one shard in adds its sums to the running totals and re-divides, so a broker that
// publishes shards one at a time arrives at the same averages a full rescan would, without
// rescanning the fleet on every publish (see incremental.go). The averages alone cannot be
// folded, because adding a shard to an average needs the count the average was taken over.
type statSums struct {
	docCount   uint64
	tokenCount float64
	titleTok   float64
	bodyTok    float64
	urlTok     float64
	anchorTok  float64
}

// shardSums reads one shard's raw statistics, the per-shard contribution computeStatSums
// accumulates and an incremental publish folds in. It mirrors the per-shard body of the
// old fleet scan exactly, including the body field's fallback to the combined token count
// for a shard built before the per-field sums existed.
func shardSums(s *Shard) statSums {
	var ss statSums
	ss.docCount = uint64(s.docCount)
	if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
		ss.tokenCount = v
	}
	// The body field falls back to token_count (title+body) for a shard built before the
	// per-field sums existed, so an older shard still normalizes the body the way it did
	// before; title and url fall back to zero, the no-normalization behavior those fields
	// had before this stat.
	if v, ok := s.r.Stat(tsumugi.StatBodyTokenCount); ok {
		ss.bodyTok = v
	} else if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
		ss.bodyTok = v
	}
	if v, ok := s.r.Stat(tsumugi.StatTitleTokenCount); ok {
		ss.titleTok = v
	}
	if v, ok := s.r.Stat(tsumugi.StatURLTokenCount); ok {
		ss.urlTok = v
	}
	// The anchor field falls back to zero for a shard built before the field was stored
	// forward, the no-normalization case its BM25F took before this stat existed.
	if v, ok := s.r.Stat(tsumugi.StatAnchorTokenCount); ok {
		ss.anchorTok = v
	}
	return ss
}

// add returns the sum of two stat accumulators, the fold an incremental publish runs to
// extend the running totals by one shard's contribution.
func (ss statSums) add(o statSums) statSums {
	return statSums{
		docCount:   ss.docCount + o.docCount,
		tokenCount: ss.tokenCount + o.tokenCount,
		titleTok:   ss.titleTok + o.titleTok,
		bodyTok:    ss.bodyTok + o.bodyTok,
		urlTok:     ss.urlTok + o.urlTok,
		anchorTok:  ss.anchorTok + o.anchorTok,
	}
}

// global derives the fleet-wide statistics from the raw sums, the same division the old
// fleet scan ran once at the end: the collection-wide average document length and the
// per-field average lengths the online BM25F normalizes each field by.
func (ss statSums) global() GlobalStats {
	gs := GlobalStats{DocCount: ss.docCount, TokenCount: ss.tokenCount}
	if ss.docCount > 0 {
		n := float64(ss.docCount)
		gs.AvgDocLen = ss.tokenCount / n
		gs.AvgFieldLen[fTitle] = ss.titleTok / n
		gs.AvgFieldLen[fBody] = ss.bodyTok / n
		gs.AvgFieldLen[fURL] = ss.urlTok / n
		gs.AvgFieldLen[fAnchor] = ss.anchorTok / n
	}
	return gs
}

// sumsFromStats reconstructs the raw accumulators from already-derived statistics, the
// values a collection artifact loads through NewBrokerWith: the counts are stored directly
// and the per-field totals are the averages times the document count. The reconstruction is
// exact up to the rounding of the division that produced the stored averages, a negligible
// relative error that the length normalization is insensitive to; it lets a broker started
// from a manifest still fold a later publish in incrementally rather than rescanning.
func sumsFromStats(gs GlobalStats) statSums {
	n := float64(gs.DocCount)
	return statSums{
		docCount:   gs.DocCount,
		tokenCount: gs.TokenCount,
		titleTok:   gs.AvgFieldLen[fTitle] * n,
		bodyTok:    gs.AvgFieldLen[fBody] * n,
		urlTok:     gs.AvgFieldLen[fURL] * n,
		anchorTok:  gs.AvgFieldLen[fAnchor] * n,
	}
}

// computeStatSums sums the per-shard statistics into the fleet-wide raw accumulators, the
// fleet scan a full broker build runs once over its shards.
func computeStatSums(shards []*Shard) statSums {
	var ss statSums
	for _, s := range shards {
		ss = ss.add(shardSums(s))
	}
	return ss
}

// computeGlobalStats sums the per-shard statistics into fleet-wide ones. It reads the
// document and token counts each shard recorded at build time and derives the
// collection-wide average document length, the term every length normalization in the
// lexical plane depends on, and the per-field average lengths the online BM25F
// normalizes each field by.
func computeGlobalStats(shards []*Shard) GlobalStats {
	return computeStatSums(shards).global()
}
