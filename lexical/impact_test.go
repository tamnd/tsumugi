package lexical

import (
	"sort"
	"testing"
)

// impactFor is the per-document impact the tests order lists by. It hashes the docID so
// the impacts spread across the whole byte range and collide, exercising both the
// descending order and the docID tie-break within an impact value.
func impactFor(docID uint32) uint8 {
	h := docID*2654435761 + 12345
	return uint8(h >> 24)
}

// buildImpactRegion indexes a corpus impact-ordered and reopens it, the round trip the
// impact tests run through so they exercise the real encode and decode paths.
func buildImpactRegion(t *testing.T, docs []map[Field]string) *Region {
	t.Helper()
	b := NewBuilder(DefaultParams())
	for i, d := range docs {
		b.AddDoc(uint32(i), d)
	}
	r, err := Open(b.BuildImpact(impactFor))
	if err != nil {
		t.Fatalf("open impact region: %v", err)
	}
	if !r.impact {
		t.Fatalf("region did not open in impact mode")
	}
	if got := r.DocCount(); got != uint32(len(docs)) {
		t.Fatalf("doc count: got %d want %d", got, len(docs))
	}
	return r
}

// TestImpactBlockRoundTrip encodes a block whose docIDs are not monotone, the case the
// signed-delta codec exists for, and checks the decode recovers the exact postings, both
// as a first block and threaded after a previous block's last docID.
func TestImpactBlockRoundTrip(t *testing.T) {
	// Impact-descending, docIDs deliberately out of order so some deltas are negative.
	ps := []impactPosting{
		{docID: 900, impact: 250},
		{docID: 12, impact: 250},
		{docID: 500, impact: 200},
		{docID: 3, impact: 200},
		{docID: 4000, impact: 100},
		{docID: 4001, impact: 7},
	}
	for _, prevLast := range []uint32{0, 42, 5000} {
		buf := encodeImpactBlock(nil, ps, prevLast)
		h, err := readBlockHeader(buf, 0)
		if err != nil {
			t.Fatalf("read header (prevLast=%d): %v", prevLast, err)
		}
		if h.nextOffset != len(buf) {
			t.Fatalf("nextOffset=%d, want %d", h.nextOffset, len(buf))
		}
		// The header's first field carries the block's min impact, the last posting's.
		if uint8(h.lastDocID) != ps[len(ps)-1].impact {
			t.Fatalf("header min impact=%d, want %d", h.lastDocID, ps[len(ps)-1].impact)
		}
		got, err := decodeImpactBlock(h, prevLast)
		if err != nil {
			t.Fatalf("decode (prevLast=%d): %v", prevLast, err)
		}
		if len(got) != len(ps) {
			t.Fatalf("decoded %d postings, want %d", len(got), len(ps))
		}
		for i := range ps {
			if got[i] != ps[i] {
				t.Fatalf("posting %d: got %+v want %+v", i, got[i], ps[i])
			}
		}
	}
}

// TestImpactBlockThreading checks that two blocks decoded back to back, the second
// delta-coded against the first's last docID, recover the full list, the threading the
// list reader depends on.
func TestImpactBlockThreading(t *testing.T) {
	b1 := []impactPosting{{docID: 10, impact: 200}, {docID: 7, impact: 150}}
	b2 := []impactPosting{{docID: 9000, impact: 140}, {docID: 8, impact: 3}}

	var buf []byte
	buf = encodeImpactBlock(buf, b1, 0)
	mid := len(buf)
	buf = encodeImpactBlock(buf, b2, b1[len(b1)-1].docID)

	h1, err := readBlockHeader(buf, 0)
	if err != nil {
		t.Fatalf("header 1: %v", err)
	}
	got1, err := decodeImpactBlock(h1, 0)
	if err != nil {
		t.Fatalf("decode 1: %v", err)
	}
	if h1.nextOffset != mid {
		t.Fatalf("block 1 nextOffset=%d, want %d", h1.nextOffset, mid)
	}
	h2, err := readBlockHeader(buf, h1.nextOffset)
	if err != nil {
		t.Fatalf("header 2: %v", err)
	}
	got2, err := decodeImpactBlock(h2, got1[len(got1)-1].docID)
	if err != nil {
		t.Fatalf("decode 2: %v", err)
	}
	all := append(got1, got2...)
	want := append(append([]impactPosting{}, b1...), b2...)
	for i := range want {
		if all[i] != want[i] {
			t.Fatalf("posting %d: got %+v want %+v", i, all[i], want[i])
		}
	}
}

// TestImpactRegionInvariant builds a region large enough to span many posting blocks and
// asserts the ordering properties directly: within-block descending, stored block-max is
// the leading impact, min-impact header is the trailing impact, block maxima monotone
// non-increasing, and maxContrib the list's leading impact.
func TestImpactRegionInvariant(t *testing.T) {
	docs := genCorpus(7, 4000, 200)
	r := buildImpactRegion(t, docs)
	ok, err := r.impactBlockInvariant()
	if err != nil {
		t.Fatalf("invariant: %v", err)
	}
	if !ok {
		t.Fatalf("impact block invariant violated")
	}
}

// TestImpactDictMatchesBM25 builds the same corpus both ways off one builder and checks the
// dictionary and bloom filter are shared: impact ordering changes the posting bodies, not
// the term set, so the two regions agree on term count and on every term's document
// frequency.
func TestImpactDictMatchesBM25(t *testing.T) {
	docs := genCorpus(11, 3000, 150)
	b := NewBuilder(DefaultParams())
	for i, d := range docs {
		b.AddDoc(uint32(i), d)
	}
	bm, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open bm25 region: %v", err)
	}
	im, err := Open(b.BuildImpact(impactFor))
	if err != nil {
		t.Fatalf("open impact region: %v", err)
	}
	if bm.TermCount() != im.TermCount() {
		t.Fatalf("term count: bm25 %d, impact %d", bm.TermCount(), im.TermCount())
	}
	for ti := uint32(0); ti < bm.TermCount(); ti++ {
		term, ok := bm.Term(ti)
		if !ok {
			t.Fatalf("bm25 term %d missing", ti)
		}
		be, bok := bm.lookupEntry(term)
		ie, iok := im.lookupEntry(term)
		if !bok || !iok {
			t.Fatalf("term %q lookup: bm25 %v, impact %v", term, bok, iok)
		}
		if be.docFreq != ie.docFreq {
			t.Fatalf("term %q docFreq: bm25 %d, impact %d", term, be.docFreq, ie.docFreq)
		}
	}
}

// naiveImpactTopK is the independent oracle: for each document it counts the query terms it
// carries that the region also holds, multiplies by the document's impact, and keeps the
// top-k under the same score-descending, docID-ascending order the region's top-k uses.
func naiveImpactTopK(r *Region, docs []map[Field]string, query string, k int) []Candidate {
	qterms := map[string]bool{}
	for _, t := range Analyze(query) {
		if _, ok := r.lookupEntry(t); ok {
			qterms[t] = true
		}
	}
	var cands []Candidate
	for id, d := range docs {
		docID := uint32(id)
		present := map[string]bool{}
		for _, text := range d {
			for _, t := range Analyze(text) {
				if qterms[t] {
					present[t] = true
				}
			}
		}
		if len(present) == 0 {
			continue
		}
		score := int32(len(present)) * int32(impactFor(docID))
		cands = append(cands, Candidate{DocID: docID, Score: score})
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Score != cands[j].Score {
			return cands[i].Score > cands[j].Score
		}
		return cands[i].DocID < cands[j].DocID
	})
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands
}

// TestImpactExhaustiveMatchesNaive is the correctness gate: over many random multi-term
// queries the region's impact scorer must return exactly the naive coverage-times-impact
// oracle, both the set of documents and their scores and order.
func TestImpactExhaustiveMatchesNaive(t *testing.T) {
	docs := genCorpus(23, 4000, 200)
	r := buildImpactRegion(t, docs)
	queries := genQueries(29, 200, 200, 4)
	const k = 50
	for _, q := range queries {
		got, err := r.SearchImpact(q, k)
		if err != nil {
			t.Fatalf("search %q: %v", q, err)
		}
		want := naiveImpactTopK(r, docs, q, k)
		if len(got) != len(want) {
			t.Fatalf("query %q: got %d results, want %d", q, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("query %q result %d: got %+v want %+v", q, i, got[i], want[i])
			}
		}
	}
}

// TestImpactSearchRejectsBM25Region checks the cross-mode guard: a docID-ordered region
// refuses an impact search rather than misreading its postings.
func TestImpactSearchRejectsBM25Region(t *testing.T) {
	docs := genCorpus(5, 100, 40)
	r := buildRegion(t, docs)
	if _, err := r.SearchImpact("term0001", 10); err != errNotImpactRegion {
		t.Fatalf("SearchImpact on bm25 region: got %v, want errNotImpactRegion", err)
	}
}

// TestImpactEmpty checks the degenerate corpus: no documents builds a region that opens in
// impact mode and returns nothing.
func TestImpactEmpty(t *testing.T) {
	b := NewBuilder(DefaultParams())
	r, err := Open(b.BuildImpact(impactFor))
	if err != nil {
		t.Fatalf("open empty impact region: %v", err)
	}
	if !r.impact {
		t.Fatalf("empty region not in impact mode")
	}
	got, err := r.SearchImpact("anything", 10)
	if err != nil {
		t.Fatalf("search empty: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("empty region returned %d results", len(got))
	}
}

// TestImpactRegionSize measures the impact region against the BM25F region over the same
// corpus, reporting bytes per posting for each so the format's size is on the record
// rather than assumed. It asserts nothing about which is smaller: impact ordering trades a
// one-byte impact payload, smaller than BM25F's field mask and frequency varints, against
// signed docID deltas that are wider than the monotone gaps docID order produces, so the
// net depends on the corpus.
func TestImpactRegionSize(t *testing.T) {
	docs := genCorpus(41, 6000, 300)
	b := NewBuilder(DefaultParams())
	for i, d := range docs {
		b.AddDoc(uint32(i), d)
	}
	bmBytes := b.Build()
	imBytes := b.BuildImpact(impactFor)

	bm, err := Open(bmBytes)
	if err != nil {
		t.Fatalf("open bm25: %v", err)
	}
	var postings int64
	for ti := uint32(0); ti < bm.TermCount(); ti++ {
		term, _ := bm.Term(ti)
		e, _ := bm.lookupEntry(term)
		postings += int64(e.docFreq)
	}
	if postings == 0 {
		t.Fatal("no postings")
	}
	t.Logf("bm25 region %d bytes (%.2f B/posting), impact region %d bytes (%.2f B/posting), %d postings",
		len(bmBytes), float64(len(bmBytes))/float64(postings),
		len(imBytes), float64(len(imBytes))/float64(postings), postings)
}
