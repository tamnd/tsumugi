package lexical

import "github.com/tamnd/tsumugi/codec"

// DefaultK is the L0 candidate count a shard returns, the value the ranking
// cascade consumes.
const DefaultK = 1000

// queryTerms analyzes a query string into the distinct terms present in this
// region, each with its dictionary entry and idf. A term absent from the shard
// (rejected by the bloom filter or missing from the dictionary) is dropped.
// Duplicate query terms collapse to one, which keeps a term from being
// double-counted in the score.
func (r *Region) queryTerms(query string) []termInfo {
	seen := map[string]bool{}
	var infos []termInfo
	for _, t := range Analyze(query) {
		if seen[t] {
			continue
		}
		seen[t] = true
		if info, ok := r.lookup(t); ok {
			infos = append(infos, info)
		}
	}
	return infos
}

// Search returns the top-k documents for a query using BlockMax-WAND, the pruned
// retrieval path. The result is strongest first.
func (r *Region) Search(query string, k int) ([]Candidate, error) {
	infos := r.queryTerms(query)
	if len(infos) == 0 {
		return nil, nil
	}
	cursors := make([]*cursor, 0, len(infos))
	for _, info := range infos {
		c, err := r.openCursor(info)
		if err != nil {
			return nil, err
		}
		cursors = append(cursors, c)
	}
	return r.blockMaxWAND(cursors, k), nil
}

// SearchExhaustive returns the top-k documents for a query by scoring every
// matching document with no pruning. It is the oracle the pruned path is checked
// against, not a serving path.
func (r *Region) SearchExhaustive(query string, k int) ([]Candidate, error) {
	infos := r.queryTerms(query)
	if len(infos) == 0 {
		return nil, nil
	}
	return r.exhaustive(infos, k)
}

// Term reconstructs a term string from its termID, for tooling and explanation.
func (r *Region) Term(termID uint32) (string, bool) { return r.dict.term(termID) }

// blockMaxInvariant decodes every block of every term and checks that the stored
// block-max is at least the true maximum contribution in that block. It is the
// safety invariant the pruning rests on, exposed for the test that asserts it
// directly rather than only as a consequence of the oracle passing.
func (r *Region) blockMaxInvariant() (bool, error) {
	for ti := uint32(0); ti < r.terms; ti++ {
		term, ok := r.dict.term(ti)
		if !ok {
			return false, errCorrupt
		}
		info, ok := r.lookup(term)
		if !ok {
			return false, errCorrupt
		}
		e := info.entry
		list := r.postings[e.postingsOff : e.postingsOff+e.postingsLen]
		off := 0
		var prevLast uint32
		for bi := uint32(0); bi < e.blockCount; bi++ {
			h, err := readBlockHeader(list, off)
			if err != nil {
				return false, err
			}
			ps, err := decodeBlock(h, prevLast)
			if err != nil {
				return false, err
			}
			stored := int32(codec.Uint32(r.blockMax[e.blockMaxOff+uint64(bi)*4:]))
			var trueMax int32
			for _, p := range ps {
				s := r.termScore(info.idf, &p.fieldTF, p.docID)
				if s > trueMax {
					trueMax = s
				}
			}
			if stored < trueMax {
				return false, nil
			}
			prevLast = h.lastDocID
			off = h.nextOffset
		}
	}
	return true, nil
}
