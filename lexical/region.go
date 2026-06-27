package lexical

import (
	"math"

	"github.com/tamnd/tsumugi/codec"
)

// regionMagic marks the lexical region, distinct from the file's TSM1 so a tool
// dumping a raw region knows what it is looking at.
const regionMagic = "LEX1"

const regionVersion = 1

// flagIDFFreeBlockMax marks a region whose block-max and list-max bounds are stored
// idf-free, the M13 format. The query path scales these bounds by the idf it is given,
// shard-local or collection-wide, so the same posting lists serve both the single-shard
// path and the broker's distributed exact-idf scoring. A region without the flag predates
// the change and bakes a shard-local idf into the bounds, so scoring it with a pushed-down
// idf would double-count; Open refuses such a region and asks for a rebuild rather than
// returning silently wrong scores.
const flagIDFFreeBlockMax = 1 << 0

// norm record width: four uint32 field lengths per document, fixed-width so the
// scorer reaches a document's lengths in O(1).
const normRecord = numFields * 4

// Region is a parsed, read-only lexical region. It answers bloom checks and
// dictionary lookups, opens cursors over posting lists, and carries the BM25F
// parameters and shard statistics the scorer needs. The bytes are the region's
// decompressed contents from the container; Region holds sub-slices of them and
// copies nothing large.
type Region struct {
	bloom    *bloom
	dict     *dict
	blockMax []byte // per-list uint32 arrays, indexed by term entry blockMaxOff
	postings []byte // per-list block runs, indexed by term entry postingsOff
	norms    []byte // fixed-width per-doc field lengths

	terms  uint32
	st     stats
	params Params
}

// header sub-offsets, all relative to the region start.
type regionHeader struct {
	flags       uint16
	termCount   uint32
	docCount    uint32
	bloomOff    uint64
	bloomLen    uint64
	dictOff     uint64
	dictLen     uint64
	blockMaxOff uint64
	blockMaxLen uint64
	postingsOff uint64
	postingsLen uint64
	normsOff    uint64
	normsLen    uint64
	avgFieldLen [numFields]float64
	params      Params
}

func (h *regionHeader) encode() []byte {
	b := make([]byte, 0, 256)
	b = append(b, regionMagic...)
	b = codec.AppendUint32(b, uint32(regionVersion))
	b = codec.AppendUint32(b, uint32(h.flags))
	b = codec.AppendUint32(b, h.termCount)
	b = codec.AppendUint32(b, h.docCount)
	for _, p := range []uint64{h.bloomOff, h.bloomLen, h.dictOff, h.dictLen,
		h.blockMaxOff, h.blockMaxLen, h.postingsOff, h.postingsLen, h.normsOff, h.normsLen} {
		b = codec.AppendUint64(b, p)
	}
	for f := 0; f < numFields; f++ {
		b = codec.AppendUint64(b, math.Float64bits(h.avgFieldLen[f]))
	}
	b = codec.AppendUint64(b, math.Float64bits(h.params.K1))
	for f := 0; f < numFields; f++ {
		b = codec.AppendUint64(b, math.Float64bits(h.params.Weight[f]))
	}
	for f := 0; f < numFields; f++ {
		b = codec.AppendUint64(b, math.Float64bits(h.params.B[f]))
	}
	// header_crc over everything written so far, then four reserved bytes.
	b = codec.AppendUint32(b, codec.CRC32C(b))
	b = codec.AppendUint32(b, 0)
	return b
}

// headerLen is the fixed encoded header length: 4 magic + 4 ver + 4 flags +
// 4 term + 4 doc + 10*8 offsets + 4*8 avg + 8 k1 + 4*8 w + 4*8 b + 4 crc + 4 rsv.
const headerLen = 4 + 4 + 4 + 4 + 4 + 10*8 + numFields*8 + 8 + numFields*8 + numFields*8 + 4 + 4

func decodeHeader(b []byte) (regionHeader, error) {
	var h regionHeader
	if len(b) < headerLen {
		return h, errCorrupt
	}
	if string(b[0:4]) != regionMagic {
		return h, errBadMagic
	}
	if codec.Uint32(b[headerLen-8:]) != codec.CRC32C(b[:headerLen-8]) {
		return h, errCorrupt
	}
	off := 4
	if codec.Uint32(b[off:]) != regionVersion {
		return h, errCorrupt
	}
	off += 4
	h.flags = uint16(codec.Uint32(b[off:]))
	off += 4
	h.termCount = codec.Uint32(b[off:])
	off += 4
	h.docCount = codec.Uint32(b[off:])
	off += 4
	ptrs := []*uint64{&h.bloomOff, &h.bloomLen, &h.dictOff, &h.dictLen,
		&h.blockMaxOff, &h.blockMaxLen, &h.postingsOff, &h.postingsLen, &h.normsOff, &h.normsLen}
	for _, p := range ptrs {
		*p = codec.Uint64(b[off:])
		off += 8
	}
	for f := 0; f < numFields; f++ {
		h.avgFieldLen[f] = math.Float64frombits(codec.Uint64(b[off:]))
		off += 8
	}
	h.params.K1 = math.Float64frombits(codec.Uint64(b[off:]))
	off += 8
	for f := 0; f < numFields; f++ {
		h.params.Weight[f] = math.Float64frombits(codec.Uint64(b[off:]))
		off += 8
	}
	for f := 0; f < numFields; f++ {
		h.params.B[f] = math.Float64frombits(codec.Uint64(b[off:]))
		off += 8
	}
	return h, nil
}

// Open parses a lexical region from its decompressed bytes.
func Open(b []byte) (*Region, error) {
	h, err := decodeHeader(b)
	if err != nil {
		return nil, err
	}
	if h.flags&flagIDFFreeBlockMax == 0 {
		return nil, errLegacyBlockMax
	}
	slice := func(off, length uint64) ([]byte, bool) {
		if off+length > uint64(len(b)) {
			return nil, false
		}
		return b[off : off+length], true
	}
	bl, ok := slice(h.bloomOff, h.bloomLen)
	if !ok {
		return nil, errCorrupt
	}
	bf, err := decodeBloom(bl)
	if err != nil {
		return nil, err
	}
	db, ok := slice(h.dictOff, h.dictLen)
	if !ok {
		return nil, errCorrupt
	}
	dc, err := decodeDict(db)
	if err != nil {
		return nil, err
	}
	bm, ok := slice(h.blockMaxOff, h.blockMaxLen)
	if !ok {
		return nil, errCorrupt
	}
	pl, ok := slice(h.postingsOff, h.postingsLen)
	if !ok {
		return nil, errCorrupt
	}
	nm, ok := slice(h.normsOff, h.normsLen)
	if !ok {
		return nil, errCorrupt
	}
	r := &Region{
		bloom:    bf,
		dict:     dc,
		blockMax: bm,
		postings: pl,
		norms:    nm,
		params:   h.params,
	}
	r.terms = h.termCount
	r.st.docCount = h.docCount
	r.st.avgFieldLen = h.avgFieldLen
	return r, nil
}

// TermCount returns the number of distinct terms in the dictionary.
func (r *Region) TermCount() uint32 { return r.terms }

// DocCount returns N, the shard's document count.
func (r *Region) DocCount() uint32 { return r.st.docCount }

// fieldLen reads a document's per-field lengths from the norms table.
func (r *Region) fieldLen(docID uint32) [numFields]uint32 {
	var out [numFields]uint32
	off := int(docID) * normRecord
	if off+normRecord > len(r.norms) {
		return out
	}
	for f := 0; f < numFields; f++ {
		out[f] = codec.Uint32(r.norms[off+f*4:])
	}
	return out
}

// termInfo bundles a term's dictionary entry with its computed idf.
type termInfo struct {
	entry termEntry
	idf   float64
}

// lookupEntry resolves an analyzed term to its dictionary entry, using the bloom
// filter to reject absent terms before touching the dictionary. It returns only the
// entry, not an idf, so a caller can choose the idf: the shard-local one or a
// collection-wide one pushed down by the broker.
func (r *Region) lookupEntry(term string) (termEntry, bool) {
	if !r.bloom.mayContain(term) {
		return termEntry{}, false
	}
	return r.dict.lookup(term)
}

// lookup resolves an analyzed term to its dictionary entry and shard-local idf, the
// single-shard scoring path.
func (r *Region) lookup(term string) (termInfo, bool) {
	e, ok := r.lookupEntry(term)
	if !ok {
		return termInfo{}, false
	}
	return termInfo{entry: e, idf: idf(r.st.docCount, e.docFreq)}, true
}
