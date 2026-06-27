package lexical

import "github.com/tamnd/tsumugi/codec"

// DefaultK is the L0 candidate count a shard returns, the value the ranking
// cascade consumes.
const DefaultK = 1000

// queryTerms analyzes a query string into the distinct terms present in this region,
// each with its dictionary entry and idf. It is the convenience path for a caller that
// hands a raw string; the broker, which analyzes once and ships the term set to every
// shard, calls termInfos directly with the pre-analyzed terms so a shard never
// re-runs the analysis chain.
func (r *Region) queryTerms(query string, idfOf map[string]float64) []termInfo {
	return r.termInfos(Analyze(query), idfOf)
}

// termInfos turns an already-analyzed term set into the per-term dictionary entries
// and idf this region scores with. The idf is the shard-local one unless idfOf
// supplies one for the term, the path the broker uses to score every shard against the
// same collection-wide idf. A term absent from the shard (rejected by the bloom filter
// or missing from the dictionary) is dropped. Duplicate terms collapse to one, which
// keeps a term from being double-counted in the score.
func (r *Region) termInfos(terms []string, idfOf map[string]float64) []termInfo {
	seen := map[string]bool{}
	var infos []termInfo
	for _, t := range terms {
		if seen[t] {
			continue
		}
		seen[t] = true
		e, ok := r.lookupEntry(t)
		if !ok {
			continue
		}
		id, ok := idfOf[t]
		if !ok {
			id = idf(r.st.docCount, e.docFreq)
		}
		infos = append(infos, termInfo{entry: e, idf: id})
	}
	return infos
}

// Search returns the top-k documents for a query using BlockMax-WAND, the pruned
// retrieval path, scoring with shard-local idf. The result is strongest first.
func (r *Region) Search(query string, k int) ([]Candidate, error) {
	return r.search(query, k, nil)
}

// SearchWithIDF is Search with the per-term idf supplied from outside instead of
// computed from this shard's local statistics. The broker gathers each query term's
// document frequency across the routed shards, divides the collection-wide document
// count by it into one idf per term, and passes that map here so every shard scores
// the term identically. That is what makes the broker's merged top-k the result a
// single index over the whole collection would give: with shard-local idf a term that
// is rare across the collection but common in one shard is scored too weakly there and
// too strongly elsewhere, and the merge favors the wrong shard's documents. A term
// missing from idfOf falls back to the shard-local idf, so a caller can override only
// the terms it has gathered.
func (r *Region) SearchWithIDF(query string, k int, idfOf map[string]float64) ([]Candidate, error) {
	return r.search(query, k, idfOf)
}

func (r *Region) search(query string, k int, idfOf map[string]float64) ([]Candidate, error) {
	return r.searchInfos(r.queryTerms(query, idfOf), k)
}

// SearchTerms is Search over an already-analyzed term set, the path the broker takes
// after analyzing the query once at the front. It skips the analysis chain entirely,
// so a shard scores the terms the broker computed rather than re-deriving them, which
// is the spec's analyze-once-at-broker rule: the chain runs one time per query, not
// once per shard the fan-out visits.
func (r *Region) SearchTerms(terms []string, k int) ([]Candidate, error) {
	return r.searchInfos(r.termInfos(terms, nil), k)
}

// SearchTermsWithIDF is SearchTerms with the per-term idf supplied from outside, the
// broker's pushed-down collection-wide idf, so every shard scores a term against the
// same df and N over the term set the broker already analyzed.
func (r *Region) SearchTermsWithIDF(terms []string, k int, idfOf map[string]float64) ([]Candidate, error) {
	return r.searchInfos(r.termInfos(terms, idfOf), k)
}

func (r *Region) searchInfos(infos []termInfo, k int) ([]Candidate, error) {
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

// DocFreqs returns the local document frequency of each distinct analyzed query term
// present in this region, omitting terms the region does not hold. It reads only the
// bloom filter and the dictionary, never a posting list, so it is the cheap first phase
// of the broker's distributed exact-idf scoring: the broker sums these across the routed
// shards to learn the collection-wide df for each term without decoding any postings.
func (r *Region) DocFreqs(query string) map[string]uint32 {
	return r.DocFreqsTerms(Analyze(query))
}

// DocFreqsTerms is DocFreqs over an already-analyzed term set, the analyze-once path
// the broker uses to gather collection-wide df without re-running the chain per shard.
func (r *Region) DocFreqsTerms(terms []string) map[string]uint32 {
	seen := map[string]bool{}
	out := map[string]uint32{}
	for _, t := range terms {
		if seen[t] {
			continue
		}
		seen[t] = true
		if e, ok := r.lookupEntry(t); ok {
			out[t] = e.docFreq
		}
	}
	return out
}

// SearchExhaustive returns the top-k documents for a query by scoring every
// matching document with no pruning. It is the oracle the pruned path is checked
// against, not a serving path.
func (r *Region) SearchExhaustive(query string, k int) ([]Candidate, error) {
	infos := r.queryTerms(query, nil)
	if len(infos) == 0 {
		return nil, nil
	}
	return r.exhaustive(infos, k)
}

// Term reconstructs a term string from its termID, for tooling and explanation.
func (r *Region) Term(termID uint32) (string, bool) { return r.dict.term(termID) }

// ForEachTerm calls fn for every term in the region's dictionary with its local
// document frequency, in sorted order. The spell corrector builds its collection-wide
// correction dictionary by merging this enumeration across the fleet's shards, summing
// each term's per-shard df into the fleet-wide df the rank tie-break uses.
func (r *Region) ForEachTerm(fn func(term string, docFreq uint32)) { r.dict.forEach(fn) }

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
			ps, err := decodeBlock(h, prevLast, r.codec)
			if err != nil {
				return false, err
			}
			// The stored block-max is idf-free; the bound the traversal uses is it scaled
			// by the term's idf, so the invariant is checked against the scaled value, the
			// same one the cursor compares the per-document score to.
			stored := scaleBound(info.idf, int32(codec.Uint32(r.blockMax[e.blockMaxOff+uint64(bi)*4:])))
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
