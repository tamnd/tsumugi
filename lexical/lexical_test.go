package lexical

import (
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// buildRegion indexes a corpus and reopens it as a Region, the round trip every
// test below runs through so it exercises the real encode and decode paths.
func buildRegion(t *testing.T, docs []map[Field]string) *Region {
	t.Helper()
	b := NewBuilder(DefaultParams())
	for i, d := range docs {
		b.AddDoc(uint32(i), d)
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open region: %v", err)
	}
	if got := r.DocCount(); got != uint32(len(docs)) {
		t.Fatalf("doc count: got %d want %d", got, len(docs))
	}
	return r
}

// genCorpus builds a deterministic random corpus large enough to span many
// dictionary blocks and many posting blocks per term, so the skip structure and
// the front coding are both stressed.
func genCorpus(seed int64, nDocs, vocab int) []map[Field]string {
	rng := rand.New(rand.NewSource(seed))
	terms := make([]string, vocab)
	for i := range terms {
		terms[i] = fmt.Sprintf("term%04d", i)
	}
	docs := make([]map[Field]string, nDocs)
	for i := range docs {
		fields := map[Field]string{}
		for _, f := range []Field{FieldTitle, FieldBody, FieldURL, FieldAnchor} {
			n := rng.Intn(8)
			var toks []string
			for j := 0; j < n; j++ {
				toks = append(toks, terms[rng.Intn(vocab)])
			}
			fields[f] = join(toks)
		}
		docs[i] = fields
	}
	return docs
}

func join(toks []string) string {
	out := ""
	for i, t := range toks {
		if i > 0 {
			out += " "
		}
		out += t
	}
	return out
}

// genQueries draws random multi-term queries from the same vocabulary so most
// queries hit several posting lists and the pivoting actually prunes.
func genQueries(seed int64, n, vocab, maxTerms int) []string {
	rng := rand.New(rand.NewSource(seed))
	out := make([]string, n)
	for i := range out {
		nt := 1 + rng.Intn(maxTerms)
		var toks []string
		for j := 0; j < nt; j++ {
			toks = append(toks, fmt.Sprintf("term%04d", rng.Intn(vocab)))
		}
		out[i] = join(toks)
	}
	return out
}

// TestPrunedMatchesExhaustive is the M1 oracle: BlockMax-WAND must return the
// identical ranked list the no-pruning scan returns, for every query, ties
// included. The integer score domain makes the two sums bit-equal, so this is an
// exact equality, not an approximate one.
func TestPrunedMatchesExhaustive(t *testing.T) {
	docs := genCorpus(1, 3000, 250)
	r := buildRegion(t, docs)
	queries := genQueries(2, 500, 250, 4)

	for _, k := range []int{1, 5, 10, 100} {
		for _, q := range queries {
			pruned, err := r.Search(q, k)
			if err != nil {
				t.Fatalf("search %q: %v", q, err)
			}
			exhaustive, err := r.SearchExhaustive(q, k)
			if err != nil {
				t.Fatalf("exhaustive %q: %v", q, err)
			}
			if !reflect.DeepEqual(pruned, exhaustive) {
				t.Fatalf("k=%d q=%q\n pruned=%v\n  exact=%v", k, q, pruned, exhaustive)
			}
		}
	}
}

// TestBlockMaxUpperBound asserts the safety invariant the pruning rests on: each
// stored block-max is at least the true maximum contribution within its block. If
// this ever fails, a skip could drop a real top-k document.
func TestBlockMaxUpperBound(t *testing.T) {
	r := buildRegion(t, genCorpus(3, 2000, 200))
	ok, err := r.blockMaxInvariant()
	if err != nil {
		t.Fatalf("invariant check: %v", err)
	}
	if !ok {
		t.Fatal("a stored block-max under-counted its block")
	}
}

// TestBM25FFieldWeighting checks the scorer ranks a title hit above an otherwise
// identical body-only hit, which is the whole point of carrying fields. The
// default params weight the title above the body.
func TestBM25FFieldWeighting(t *testing.T) {
	docs := []map[Field]string{
		{FieldTitle: "wombat", FieldBody: "filler filler filler"},
		{FieldBody: "wombat filler filler"},
	}
	r := buildRegion(t, docs)
	res, err := r.Search("wombat", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("want 2 hits, got %d: %v", len(res), res)
	}
	if res[0].DocID != 0 {
		t.Fatalf("title hit should rank first, got order %v", res)
	}
	if res[0].Score <= res[1].Score {
		t.Fatalf("title hit should outscore body hit: %v", res)
	}
}

// TestSearchMissingTerm returns nothing for a term no document holds, taking the
// bloom-reject path.
func TestSearchMissingTerm(t *testing.T) {
	r := buildRegion(t, genCorpus(5, 500, 100))
	res, err := r.Search("notavocabularyterm", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if res != nil {
		t.Fatalf("want no hits, got %v", res)
	}
}

// blockCodecs is every docID gap codec, so the round-trip tests cover each one and
// a new codec is exercised the moment it is added to codecByID.
var blockCodecs = []struct {
	name string
	dc   docCodec
}{
	{"varint", varintCodec{}},
	{"streamvbyte", streamVByteCodec{}},
}

// TestBlockCodecRoundTrip drives encodeBlock and decodeBlock across the awkward
// shapes: a single posting, a full block, a short trailing block, and field
// masks with gaps. The decoded postings must equal the originals exactly, under
// every gap codec.
func TestBlockCodecRoundTrip(t *testing.T) {
	cases := [][]posting{
		{{docID: 0, fieldTF: [numFields]uint32{1, 0, 0, 0}}},
		{{docID: 7, fieldTF: [numFields]uint32{0, 0, 0, 5}}},
		mkPostings(1, blockSize),     // full block
		mkPostings(100, blockSize-1), // short block
		mkPostings(1, 3),             // tiny block
		largeGapPostings(),           // gaps spanning all 1..4 byte widths
	}
	for _, codec := range blockCodecs {
		for ci, ps := range cases {
			var buf []byte
			buf = encodeBlock(buf, ps, 0, codec.dc)
			h, err := readBlockHeader(buf, 0)
			if err != nil {
				t.Fatalf("%s case %d header: %v", codec.name, ci, err)
			}
			if h.nextOffset != len(buf) {
				t.Fatalf("%s case %d nextOffset %d want %d", codec.name, ci, h.nextOffset, len(buf))
			}
			got, err := decodeBlock(h, 0, codec.dc)
			if err != nil {
				t.Fatalf("%s case %d decode: %v", codec.name, ci, err)
			}
			if !reflect.DeepEqual(got, ps) {
				t.Fatalf("%s case %d round trip\n got=%v\nwant=%v", codec.name, ci, got, ps)
			}
		}
	}
}

// TestBlockChainPrevLast encodes two consecutive blocks the way the builder does,
// delta-coding the second against the first block's last docID, and decodes them
// back to the original absolute docIDs, under every gap codec.
func TestBlockChainPrevLast(t *testing.T) {
	for _, codec := range blockCodecs {
		first := mkPostings(10, blockSize)
		second := mkPostings(int(first[len(first)-1].docID)+5, 20)
		var buf []byte
		buf = encodeBlock(buf, first, 0, codec.dc)
		prevLast := first[len(first)-1].docID
		buf = encodeBlock(buf, second, prevLast, codec.dc)

		h1, err := readBlockHeader(buf, 0)
		if err != nil {
			t.Fatalf("%s h1: %v", codec.name, err)
		}
		got1, err := decodeBlock(h1, 0, codec.dc)
		if err != nil {
			t.Fatalf("%s decode1: %v", codec.name, err)
		}
		if !reflect.DeepEqual(got1, first) {
			t.Fatalf("%s first block mismatch", codec.name)
		}
		h2, err := readBlockHeader(buf, h1.nextOffset)
		if err != nil {
			t.Fatalf("%s h2: %v", codec.name, err)
		}
		got2, err := decodeBlock(h2, h1.lastDocID, codec.dc)
		if err != nil {
			t.Fatalf("%s decode2: %v", codec.name, err)
		}
		if !reflect.DeepEqual(got2, second) {
			t.Fatalf("%s second block mismatch", codec.name)
		}
	}
}

// largeGapPostings builds a block whose docID gaps land in each StreamVByte length
// class, 1 through 4 bytes, so the codec's per-gap length selection is exercised at
// every width rather than only the small gaps a dense list produces.
func largeGapPostings() []posting {
	gaps := []uint32{1, 200, 1 << 9, 1 << 17, 1 << 25, 3}
	ps := make([]posting, len(gaps))
	var d uint32
	for i, g := range gaps {
		d += g
		ps[i].docID = d
		ps[i].fieldTF[FieldTitle] = uint32(i%3) + 1
	}
	return ps
}

func mkPostings(start, n int) []posting {
	ps := make([]posting, n)
	d := uint32(start)
	for i := range ps {
		ps[i].docID = d
		// vary the field mask and frequencies so the payload codec is exercised.
		ps[i].fieldTF[FieldTitle] = uint32(i%3) + 1
		if i%2 == 0 {
			ps[i].fieldTF[FieldBody] = uint32(i%4) + 1
		}
		if i%5 == 0 {
			ps[i].fieldTF[FieldAnchor] = 2
		}
		d += uint32(1 + i%7) // strictly increasing gaps
	}
	return ps
}

// TestDictRoundTrip builds a front-coded dictionary over a sorted vocabulary and
// checks both directions: every term resolves to its entry, and every termID
// reconstructs its term. It also confirms an absent term is rejected.
func TestDictRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	set := map[string]bool{}
	for len(set) < 500 {
		set[fmt.Sprintf("w%d-%s", rng.Intn(100), randSuffix(rng))] = true
	}
	terms := make([]string, 0, len(set))
	for w := range set {
		terms = append(terms, w)
	}
	sort.Strings(terms)

	entries := make([]termEntry, len(terms))
	for i := range entries {
		entries[i] = termEntry{
			termID:      uint32(i),
			postingsOff: uint64(i * 13),
			postingsLen: uint64(i + 1),
			docFreq:     uint32(i*2 + 1),
			blockCount:  uint32(i%4 + 1),
			blockMaxOff: uint64(i * 4),
			maxContrib:  int32(i * 100),
		}
	}

	d, err := decodeDict(encodeDict(terms, entries))
	if err != nil {
		t.Fatalf("decode dict: %v", err)
	}
	for i, term := range terms {
		e, ok := d.lookup(term)
		if !ok {
			t.Fatalf("term %q missing", term)
		}
		if !reflect.DeepEqual(e, entries[i]) {
			t.Fatalf("term %q entry\n got=%+v\nwant=%+v", term, e, entries[i])
		}
		back, ok := d.term(uint32(i))
		if !ok || back != term {
			t.Fatalf("termID %d reverse: got %q ok=%v want %q", i, back, ok, term)
		}
	}
	if _, ok := d.lookup("definitely-not-present"); ok {
		t.Fatal("absent term resolved")
	}
}

func randSuffix(rng *rand.Rand) string {
	const alpha = "abcdefghijklmnop"
	n := 1 + rng.Intn(6)
	b := make([]byte, n)
	for i := range b {
		b[i] = alpha[rng.Intn(len(alpha))]
	}
	return string(b)
}

// TestBloomNoFalseNegative is the property the prefilter must hold: every term
// added reports present. A false negative would silently drop a real posting
// list, so this is the one bloom error that is never acceptable.
func TestBloomNoFalseNegative(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	terms := make([]string, 5000)
	for i := range terms {
		terms[i] = fmt.Sprintf("tok-%d-%s", i, randSuffix(rng))
	}
	bf := newBloom(len(terms), 0.01)
	for _, term := range terms {
		bf.add(term)
	}
	for _, term := range terms {
		if !bf.mayContain(term) {
			t.Fatalf("false negative for %q", term)
		}
	}

	// And the same after an encode/decode round trip.
	dec, err := decodeBloom(bf.encode())
	if err != nil {
		t.Fatalf("decode bloom: %v", err)
	}
	for _, term := range terms {
		if !dec.mayContain(term) {
			t.Fatalf("false negative after round trip for %q", term)
		}
	}
}

// TestBloomFalsePositiveRate is a soft check that the filter is doing real work:
// the observed false-positive rate on absent terms should be in the ballpark of
// the configured target, not near one.
func TestBloomFalsePositiveRate(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	bf := newBloom(10000, 0.01)
	for i := 0; i < 10000; i++ {
		bf.add(fmt.Sprintf("present-%d", i))
	}
	fp := 0
	const trials = 10000
	for i := 0; i < trials; i++ {
		if bf.mayContain(fmt.Sprintf("absent-%d-%s", i, randSuffix(rng))) {
			fp++
		}
	}
	rate := float64(fp) / trials
	if rate > 0.05 {
		t.Fatalf("false-positive rate %.3f far above target 0.01", rate)
	}
}

// TestEmptyRegion builds a region with no documents and confirms it opens and
// answers a query without panicking.
func TestEmptyRegion(t *testing.T) {
	b := NewBuilder(DefaultParams())
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r.DocCount() != 0 || r.TermCount() != 0 {
		t.Fatalf("empty region not empty: docs=%d terms=%d", r.DocCount(), r.TermCount())
	}
	res, err := r.Search("anything", 10)
	if err != nil {
		t.Fatalf("search empty: %v", err)
	}
	if res != nil {
		t.Fatalf("empty region returned hits: %v", res)
	}
}
