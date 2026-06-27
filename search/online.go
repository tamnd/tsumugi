package search

import (
	"unicode/utf8"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/vector"
)

// This file is the online query-dependent feature extractor, the L2 half of the
// feature contract doc 09 pins. The feature matrix holds the query-independent
// signals, a strided byte read per candidate; everything that depends on the
// query-document pair is computed here, online, over the small L2 survivor set.
// The split is the whole reason these are L2 features: decoding a candidate's
// fields and positions and dotting its vector costs real work, so the cascade
// defers it to the two hundred survivors the L1 cut leaves, never the thousand the
// retrieval planes return.
//
// The full L2 vector a model scores is the query-independent matrix row followed
// by the online vector this file produces, in a fixed order. The L1 cut reads only
// the matrix row, the cheap features it can afford over its larger input; the
// model the L2 reranker scores is trained against the concatenated width.

// OnlineFeature indexes a column of the online query-dependent feature vector. The
// order is the contract the L2 model is trained against: it follows the matrix
// columns, so the full vector is the matrix row concatenated with this vector in
// this order. New online features append; an existing one never moves.
type OnlineFeature int

const (
	OnBM25FTotal         OnlineFeature = iota // field-weighted BM25F over all fields
	OnBM25Title                               // BM25 restricted to the title field
	OnBM25Body                                // BM25 restricted to the body field
	OnBM25URL                                 // BM25 over the url-token field
	OnDenseCosine                             // cosine of the query vector and the int8 rerank vector
	OnTermCoverage                            // fraction of query terms present anywhere in the document
	OnTermCoverageTitle                       // fraction of query terms present in the title
	OnExactMatch                              // 1 if the body contains the query as a contiguous phrase
	OnExactMatchTitle                         // 1 if the title contains the query as a contiguous phrase
	OnOrderedProximity                        // 1/(1+span) of the smallest in-order window over the body
	OnUnorderedProximity                      // 1/(1+span) of the smallest any-order window over the body
	OnMinPairDistance                         // 1/(1+gap) of the closest two distinct query terms in the body
	OnFirstOccurrence                         // 1/(1+pos) of the earliest query-term hit in the body
	OnFieldHitTitle                           // count of distinct query terms hitting the title
	OnFieldHitBody                            // count of distinct query terms hitting the body
	OnURLTermMatch                            // count of distinct query terms in the url path or host
	OnURLExactHost                            // 1 if a query term equals a host label
	OnQueryLength                             // number of distinct analyzed query terms, broadcast
	OnIdfSum                                  // sum of query-term idf, broadcast
	OnIdfMax                                  // the rarest query term's idf, broadcast
	NumOnline                                 // count of online features, the online vector width
)

// missingFeature is the sentinel an absent online feature takes, distinct from a
// real zero. doc 09 is explicit that absent is not zero: a zero dense cosine means
// the query and document vectors are orthogonal, an absent one means the region
// carries no rerank copy at all, and a tree must be able to split on the
// difference. The value sits below every real feature's range so a threshold split
// can isolate it, the same role LightGBM's missing-value branch plays.
const missingFeature = -1.0

// onlineExtractor computes the online feature vector for a candidate against one
// query. It is built once per query from the shard's regions and the query
// analysis, then called per L2 survivor. It holds no per-candidate state, so the
// same extractor serves every survivor of one query and is discarded when the
// query is done.
type onlineExtractor struct {
	terms      []string        // distinct analyzed query terms, the coverage and hit-count unit
	orderedIdx []int32         // query terms in query order as term indices, the phrase the exact-match check seeks
	qindex     map[string]int  // query term -> its index in terms, the per-field tf bucket
	idfTerm    []float64       // per-term idf aligned with terms, resolved once
	fwd        *forward.Region // the candidate text, decoded per survivor
	vec        *vector.Region  // the int8 rerank vectors, for the dense cosine
	qvec       []float32       // the dense query vector, nil when the query carries none
	params     lexical.Params  // the BM25F knobs, the same the lexical plane scores with
	avgField   [3]float64      // fleet average field length for title, body, url; zero means no normalization
	queryLen   float64         // broadcast: distinct query-term count
	idfSum     float64         // broadcast: sum of query-term idf
	idfMax     float64         // broadcast: the rarest query term's idf
	minTermLen int             // shortest query term in bytes, the lower length gate
	maxTermLen int             // longest query term in bytes, the upper length gate

	// Per-query scratch reused across the survivors one extractor scores, so the
	// per-candidate path allocates nothing for tokenization. An extractor serves one
	// query on one goroutine (the cascade reranks a query's survivors in sequence),
	// so reuse across calls is safe.
	tokBuf    []byte          // the current token's lowercased bytes, rebuilt per token
	tfBuf     [nField][]int   // per-field query-term frequencies, zeroed per scan
	streamBuf [nField][]int32 // per-field query-index stream, regrown per scan
	hitsBuf   []posHit        // body query-term occurrences, regrown per proximity call
	winCount  []int           // per-term window counter for the minimum-window sweep
}

// field indices into the extractor's per-field arrays, kept local so the online
// path names title, body, and url without importing the lexical field constants
// that also carry the anchor field this build does not materialize.
const (
	fTitle = 0
	fBody  = 1
	fURL   = 2
	nField = 3
)

// maxBodyScanRunes caps how far the online scan reads into the body field, the
// L2 body window. The online features the body feeds, BM25 and proximity and exact
// match, draw their signal from where the query terms cluster, which on a relevant
// page is the leading content; reading the whole of a multi-hundred-kilobyte page
// for every survivor would blow the L2 budget for no ranking gain. The cap bounds
// the per-candidate body cost to a constant regardless of document size, so the
// stage cost is the survivor count times a fixed window, not the corpus's longest
// document. Title and url are short and read in full.
const maxBodyScanRunes = 10000

// newOnlineExtractor analyzes the query once and binds the regions the online
// features read. idfOf is the per-term idf the broker pushed down, or nil to fall
// back to a neutral idf; avgBody is the fleet average body length BM25 normalizes
// the body field by, with the title and url fields left unnormalized until the
// fleet statistics carry their averages. A query with no text yields an extractor
// whose text features are all zero, which is the right answer for a pure-vector
// query: it has no terms to cover or locate.
func newOnlineExtractor(q Query, fwd *forward.Region, vec *vector.Region, idfOf map[string]float64, avgBody float64) *onlineExtractor {
	ordered := q.lexTerms()
	seen := make(map[string]bool, len(ordered))
	var terms []string
	for _, t := range ordered {
		if !seen[t] {
			seen[t] = true
			terms = append(terms, t)
		}
	}
	e := &onlineExtractor{
		terms:    terms,
		qindex:   make(map[string]int, len(terms)),
		idfTerm:  make([]float64, len(terms)),
		fwd:      fwd,
		vec:      vec,
		qvec:     q.Vector,
		params:   lexical.DefaultParams(),
		queryLen: float64(len(terms)),
	}
	e.avgField[fBody] = avgBody
	e.minTermLen, e.maxTermLen = 1<<31-1, 0
	for i, t := range terms {
		e.qindex[t] = i
		v := termIDF(idfOf, t)
		e.idfTerm[i] = v
		e.idfSum += v
		if v > e.idfMax {
			e.idfMax = v
		}
		if len(t) < e.minTermLen {
			e.minTermLen = len(t)
		}
		if len(t) > e.maxTermLen {
			e.maxTermLen = len(t)
		}
	}
	if len(terms) == 0 {
		e.minTermLen = 0
	}
	// The exact-match check seeks the query terms in query order; resolve them to
	// term indices once so the per-candidate phrase scan compares ints, not strings.
	e.orderedIdx = make([]int32, len(ordered))
	for i, t := range ordered {
		e.orderedIdx[i] = int32(e.qindex[t])
	}
	for f := 0; f < nField; f++ {
		e.tfBuf[f] = make([]int, len(terms))
	}
	e.winCount = make([]int, len(terms))
	return e
}

// termIDF returns a term's idf, the pushed-down collection value when the query
// carries one and a neutral one otherwise. A term the idf map does not name (it
// matched in this document but the gather missed it) takes the neutral value
// rather than zero, so its BM25 contribution is not silently erased.
func termIDF(idf map[string]float64, t string) float64 {
	if idf != nil {
		if v, ok := idf[t]; ok {
			return v
		}
	}
	return 1.0
}

// features returns the online feature vector for one candidate, in OnlineFeature
// order. It decodes the candidate's title, body, and url from the forward region,
// tokenizes each, and computes the per-field BM25, the coverage and exact-match
// signals, the proximity signals from the body token positions, and the url-match
// signals, then the dense cosine from the vector region and the broadcast
// query-shape signals. A query with no terms still returns the broadcast columns
// and the dense cosine, so a pure-vector query is scored on the features it has.
func (e *onlineExtractor) features(localID uint32) []float64 {
	out := make([]float64, NumOnline)

	// Scan each field once into a compact query-index stream and per-query-term
	// frequencies. The scan never materializes the field's tokens as strings: it
	// matches each token against the small query-term set by hashing the token's
	// bytes in place, so the per-candidate cost is the field length, not its
	// vocabulary, and the path allocates only the small streams it reuses.
	tfTitle, streamTitle, lenTitle := e.scanField(localID, "title", fTitle)
	tfBody, streamBody, lenBody := e.scanField(localID, "body", fBody)
	tfURL, _, lenURL := e.scanField(localID, "url", fURL)

	// Per-field BM25 and the field-weighted total. The total mirrors the lexical
	// plane's BM25F so the feature agrees with the retrieval score it refines.
	out[OnBM25Title] = e.bm25(tfTitle, lenTitle, fTitle)
	out[OnBM25Body] = e.bm25(tfBody, lenBody, fBody)
	out[OnBM25URL] = e.bm25(tfURL, lenURL, fURL)
	out[OnBM25FTotal] = e.bm25f(tfTitle, tfBody, tfURL, lenTitle, lenBody, lenURL)

	// Coverage and field hit counts from the per-field tf: a query term hits a field
	// when its count there is non-zero, and it is covered when it hits any field.
	titleHits, bodyHits, urlHits, covered, titleCovered := 0, 0, 0, 0, 0
	for i := range e.terms {
		hit := false
		if tfTitle[i] > 0 {
			titleHits++
			titleCovered++
			hit = true
		}
		if tfBody[i] > 0 {
			bodyHits++
			hit = true
		}
		if tfURL[i] > 0 {
			urlHits++
			hit = true
		}
		if hit {
			covered++
		}
	}
	if e.queryLen > 0 {
		out[OnTermCoverage] = float64(covered) / e.queryLen
		out[OnTermCoverageTitle] = float64(titleCovered) / e.queryLen
	}
	out[OnFieldHitTitle] = float64(titleHits)
	out[OnFieldHitBody] = float64(bodyHits)
	out[OnURLTermMatch] = float64(urlHits)

	// Exact phrase match: the query terms as a contiguous run in a field, read off
	// the field's query-index stream so the check compares ints, not strings.
	out[OnExactMatch] = boolFeat(phraseMatch(streamBody, e.orderedIdx))
	out[OnExactMatchTitle] = boolFeat(phraseMatch(streamTitle, e.orderedIdx))

	// Url host match: a query term equal to a host label.
	out[OnURLExactHost] = boolFeat(e.hostMatch(localID))

	// Proximity over the body positions, the most expensive online features and the
	// ones the lexical plane explicitly deferred to L2.
	prox := e.proximity(streamBody, tfBody)
	out[OnOrderedProximity] = prox.ordered
	out[OnUnorderedProximity] = prox.unordered
	out[OnMinPairDistance] = prox.minPair
	out[OnFirstOccurrence] = prox.first

	// Dense cosine from the int8 rerank vector, absent when the query carries no
	// vector or the region keeps no rerank copy.
	if e.vec != nil && len(e.qvec) > 0 {
		if c, ok := e.vec.Cosine(e.qvec, localID); ok {
			out[OnDenseCosine] = c
		} else {
			out[OnDenseCosine] = missingFeature
		}
	} else {
		out[OnDenseCosine] = missingFeature
	}

	// Broadcast query-shape signals, the same for every candidate of one query.
	out[OnQueryLength] = e.queryLen
	out[OnIdfSum] = e.idfSum
	out[OnIdfMax] = e.idfMax
	return out
}

// scanField decodes a forward-region field for a candidate and walks it once,
// applying the same lowercase-and-split chain lexical.Analyze uses so a field token
// matches a query term byte for byte, but without materializing the field's tokens
// as strings. For each token it folds the bytes into a reused buffer and looks the
// buffer up in the query-term index with a map read that the compiler keeps
// allocation-free; the result is the per-query-term frequency tf and a stream that
// records, per field token in order, the query-term index it carries or -1. The tf
// slice and the stream backing array are reused across the survivors one extractor
// scores, so a candidate's scan allocates nothing. It returns the field length in
// tokens alongside, the BM25 normalizer. A shard with no forward region, a missing
// column, or a query with no terms yields an empty field.
func (e *onlineExtractor) scanField(localID uint32, col string, f int) (tf []int, stream []int32, flen int) {
	tf = e.tfBuf[f]
	for i := range tf {
		tf[i] = 0
	}
	stream = e.streamBuf[f][:0]
	if e.fwd == nil || len(e.terms) == 0 {
		e.streamBuf[f] = stream
		return tf, stream, 0
	}
	b, ok := e.fwd.Column(col, localID)
	if !ok || len(b) == 0 {
		e.streamBuf[f] = stream
		return tf, stream, 0
	}
	buf := e.tokBuf[:0]
	overflow := false // the current token is longer than any query term, so cannot match
	emit := func() {
		if len(buf) == 0 && !overflow {
			return
		}
		flen++
		idx := int32(-1)
		// A token can match a query term only when its byte length falls in the query's
		// term-length range; outside it the lookup cannot hit, so it is skipped. This
		// keeps the dominant cost, hashing the token bytes, off the long tokens raw text
		// is full of, the CJK runs the splitter leaves whole above all. The map read
		// keyed by string(buf) does not copy the bytes: the compiler special cases
		// m[string(b)] so the lookup is allocation-free.
		if !overflow && len(buf) >= e.minTermLen && len(buf) <= e.maxTermLen {
			if i, ok := e.qindex[string(buf)]; ok {
				tf[i]++
				idx = int32(i)
			}
		}
		stream = append(stream, idx)
		buf = buf[:0]
		overflow = false
	}
	scanCap := 0
	if f == fBody {
		scanCap = maxBodyScanRunes
	}
	runes := 0
	for _, r := range string(b) {
		if lexical.IsTokenRune(r) {
			if !overflow {
				buf = utf8.AppendRune(buf, lexical.FoldRune(r))
				if len(buf) > e.maxTermLen {
					// The token already exceeds the longest query term; stop folding and
					// holding it, just consume the rest of its runes to find the boundary.
					overflow = true
					buf = buf[:0]
				}
			}
		} else {
			emit()
		}
		runes++
		if scanCap > 0 && runes >= scanCap {
			break
		}
	}
	emit()
	e.tokBuf = buf[:0]
	e.streamBuf[f] = stream
	return tf, stream, flen
}

// fieldNorm is the BM25 length-normalization denominator for a field, 1 when the
// field has no fleet average to normalize against.
func (e *onlineExtractor) fieldNorm(flen, f int) float64 {
	avg := e.avgField[f]
	if avg <= 0 {
		return 1.0
	}
	b := e.params.B[f]
	return 1 - b + b*float64(flen)/avg
}

// bm25 is the single-field BM25 over the candidate's query-term frequencies: the
// classic idf * tf*(k1+1) / (tf + k1*(1-b+b*len/avg)) summed over the query terms,
// using the lexical plane's k1 and per-field b. With no fleet average for the field
// the length normalization falls away (norm = 1), a simplified but valid BM25 that
// still rewards term frequency and idf.
func (e *onlineExtractor) bm25(tf []int, flen, f int) float64 {
	if flen == 0 || len(e.terms) == 0 {
		return 0
	}
	k1 := e.params.K1
	norm := e.fieldNorm(flen, f)
	var s float64
	for i, c := range tf {
		if c == 0 {
			continue
		}
		cf := float64(c)
		s += e.idfTerm[i] * cf * (k1 + 1) / (cf + k1*norm)
	}
	return s
}

// bm25f mirrors the lexical plane's BM25F: each field's term frequency is weighted
// and length-normalized, the weighted frequencies are summed across fields into tfF,
// and the term contributes idf * tfF/(k1+tfF). Summed over the query terms this is
// the field-weighted total the retrieval score is built from, so the feature and the
// L0 score move together.
func (e *onlineExtractor) bm25f(tfT, tfB, tfU []int, lenT, lenB, lenU int) float64 {
	if len(e.terms) == 0 {
		return 0
	}
	k1 := e.params.K1
	wT := e.params.Weight[lexical.FieldTitle]
	wB := e.params.Weight[lexical.FieldBody]
	wU := e.params.Weight[lexical.FieldURL]
	normT := e.fieldNorm(lenT, fTitle)
	normB := e.fieldNorm(lenB, fBody)
	normU := e.fieldNorm(lenU, fURL)
	var total float64
	for i := range e.terms {
		var tfF float64
		if tfT[i] > 0 {
			tfF += wT * float64(tfT[i]) / normT
		}
		if tfB[i] > 0 {
			tfF += wB * float64(tfB[i]) / normB
		}
		if tfU[i] > 0 {
			tfF += wU * float64(tfU[i]) / normU
		}
		if tfF == 0 {
			continue
		}
		total += e.idfTerm[i] * tfF / (k1 + tfF)
	}
	return total
}

// hostMatch reports whether a query term equals one of the candidate's host
// labels, the url_exact_host_match signal: a query that names the site, like
// "wikipedia", should be able to reward the page on that host. The host comes from
// the candidate's stored url; its labels are split on dots and matched against the
// query terms.
func (e *onlineExtractor) hostMatch(localID uint32) bool {
	if e.fwd == nil || len(e.terms) == 0 {
		return false
	}
	b, ok := e.fwd.Column("url", localID)
	if !ok || len(b) == 0 {
		return false
	}
	host := analyze.HostOf(string(b))
	if host == "" {
		return false
	}
	for _, label := range lexical.Analyze(host) {
		if _, ok := e.qindex[label]; ok {
			return true
		}
	}
	return false
}

// proxResult holds the four proximity features, each already transformed so that
// closer is larger.
type proxResult struct {
	ordered   float64
	unordered float64
	minPair   float64
	first     float64
}

// proximity computes the body-position proximity features. It builds the merged
// list of query-term occurrences in the body, then derives the ordered span, the
// unordered minimum window, the closest distinct pair, and the first occurrence.
// tfBody is the body's per-query-term frequency, so the count of present terms the
// minimum window must cover is read off it rather than rescanned. A query with
// fewer than two present terms has no pair or window to measure, so those features
// stay zero while first-occurrence still reports the lone term's position.
func (e *onlineExtractor) proximity(stream []int32, tfBody []int) proxResult {
	if len(stream) == 0 || len(e.terms) == 0 {
		return proxResult{}
	}
	// Merge the body hits of the query terms into one position-sorted list, the raw
	// material every proximity feature reads. The stream is in token order, so the
	// merged list is already sorted by position. A -1 entry is a non-query token.
	hits := e.hitsBuf[:0]
	firstAt := -1
	for i, id := range stream {
		if id >= 0 {
			hits = append(hits, posHit{at: i, term: int(id)})
			if firstAt < 0 {
				firstAt = i
			}
		}
	}
	e.hitsBuf = hits
	var res proxResult
	if firstAt >= 0 {
		res.first = 1.0 / (1.0 + float64(firstAt))
	}
	if len(hits) < 2 {
		return res
	}
	// Minimum pair distance: the smallest gap between two adjacent hits of distinct
	// terms in the merged list.
	minGap := -1
	for i := 1; i < len(hits); i++ {
		if hits[i].term == hits[i-1].term {
			continue
		}
		g := hits[i].at - hits[i-1].at
		if minGap < 0 || g < minGap {
			minGap = g
		}
	}
	if minGap >= 0 {
		res.minPair = 1.0 / (1.0 + float64(minGap))
	}
	// Unordered minimum window covering every distinct present query term, the
	// classic two-pointer sweep over the merged hits. The number of distinct present
	// terms is the count of non-zero body frequencies.
	distinct := 0
	for _, c := range tfBody {
		if c > 0 {
			distinct++
		}
	}
	if w := e.minWindow(hits, distinct); w > 0 {
		res.unordered = 1.0 / (1.0 + float64(w))
	}
	// Ordered span: the smallest window that contains the query terms in query
	// order. It is the minimum over end positions of an in-order match length.
	if span := orderedSpan(stream, e.orderedIdx); span > 0 {
		res.ordered = 1.0 / (1.0 + float64(span))
	}
	return res
}

// posHit is one query-term occurrence in the body: its token position and which
// distinct query term it belongs to, the merge label the window sweep groups on.
type posHit struct {
	at   int
	term int
}

// minWindow is the minimum-width window over the position-sorted hits that covers
// all distinct present terms, the standard minimum-window two-pointer sweep. The
// width is the number of tokens the window spans, the difference of the bounding
// positions plus one. It returns zero when no window covers every distinct term.
// The per-term occupancy counter is the extractor's reused winCount slice, indexed
// by term id and zeroed over just the terms the window touched, so the sweep
// allocates nothing.
func (e *onlineExtractor) minWindow(hits []posHit, distinct int) int {
	if distinct <= 0 {
		return 0
	}
	count := e.winCount
	covered := 0
	best := 0
	left := 0
	for right := 0; right < len(hits); right++ {
		if count[hits[right].term] == 0 {
			covered++
		}
		count[hits[right].term]++
		for covered == distinct {
			w := hits[right].at - hits[left].at + 1
			if best == 0 || w < best {
				best = w
			}
			count[hits[left].term]--
			if count[hits[left].term] == 0 {
				covered--
			}
			left++
		}
	}
	// Reset only the slots the remaining open window left non-zero, the terms from
	// left to the end, so the next call starts from an all-zero counter.
	for i := left; i < len(hits); i++ {
		count[hits[i].term] = 0
	}
	return best
}

// orderedSpan is the smallest window in the body that contains the query terms in
// query order, allowing other tokens between them, the minimum-window-subsequence
// span. It sweeps the stream once with a forward cursor through the ordered query
// terms; when the cursor completes a match it walks back from that end to the
// tightest start, records the span, and resumes the forward sweep just past that
// start. The backward walk runs only on a completed match, so the work is the body
// length plus one back-scan per match rather than a forward scan from every start,
// the bound the L2 budget needs over the survivor set. It returns zero when the
// body does not contain the terms in order. Repeated query terms are handled
// because the cursor advances one term per matching token. The stream carries -1
// for non-query tokens, which never equal a want index, so they fall between terms.
func orderedSpan(stream, want []int32) int {
	if len(want) == 0 {
		return 0
	}
	best := 0
	ti := 0
	for si := 0; si < len(stream); si++ {
		if stream[si] != want[ti] {
			continue
		}
		ti++
		if ti < len(want) {
			continue
		}
		// A full in-order match ends at si. Walk back through the ordered terms to
		// the tightest start that still matches, the minimal span ending here.
		end := si
		j := len(want) - 1
		back := si
		for j >= 0 {
			if stream[back] == want[j] {
				j--
			}
			if j < 0 {
				break
			}
			back--
		}
		span := end - back + 1
		if best == 0 || span < best {
			best = span
		}
		// Resume the forward sweep just past the tightest start so the next match
		// cannot reuse a token before it, the move that keeps the sweep linear.
		si = back
		ti = 0
	}
	return best
}

// phraseMatch reports whether want appears as a contiguous run in the field's
// query-index stream, the exact-match check. The stream carries -1 for non-query
// tokens, so any token between the wanted terms breaks the run, exactly the
// contiguous-phrase semantics. An empty phrase never matches, so a query with no
// terms is not credited an exact match on every document.
func phraseMatch(stream, want []int32) bool {
	if len(want) == 0 || len(want) > len(stream) {
		return false
	}
	for i := 0; i+len(want) <= len(stream); i++ {
		match := true
		for j := range want {
			if stream[i+j] != want[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// boolFeat maps a predicate to the 1.0/0.0 a tree splits on.
func boolFeat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
