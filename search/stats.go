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
}

// computeGlobalStats sums the per-shard statistics into fleet-wide ones. It reads the
// document and token counts each shard recorded at build time and derives the
// collection-wide average document length, the term every length normalization in the
// lexical plane depends on.
func computeGlobalStats(shards []*Shard) GlobalStats {
	var gs GlobalStats
	for _, s := range shards {
		gs.DocCount += uint64(s.docCount)
		if v, ok := s.r.Stat(tsumugi.StatTokenCount); ok {
			gs.TokenCount += v
		}
	}
	if gs.DocCount > 0 {
		gs.AvgDocLen = gs.TokenCount / float64(gs.DocCount)
	}
	return gs
}
