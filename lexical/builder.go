package lexical

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// Builder accumulates documents into a BM25F lexical region. It is the in-memory
// M1 build: tokenize each document's four fields, invert into term-to-postings,
// then encode the dictionary, postings, block-max metadata, bloom filter, and
// the per-document field-length norms into one region byte run the container
// stores as RegionLexical. It holds the whole inverted index in maps, so it suits a
// corpus that fits in RAM. SpimiBuilder is the external-merge build for a corpus past
// RAM; it shares this builder's encoder through assembleRegion, so the two emit
// byte-identical regions.
type Builder struct {
	params Params
	// term -> docID -> per-field term frequencies
	terms   map[string]map[uint32]*[numFields]uint32
	norms   map[uint32]*[numFields]uint32
	maxDoc  uint32
	hasDocs bool
}

// NewBuilder starts a builder with the given BM25F parameters. The same params
// are written into the region and used by the traversal, so the block-max bounds
// the build computes match the contributions the traversal computes.
func NewBuilder(params Params) *Builder {
	return &Builder{
		params: params,
		terms:  map[string]map[uint32]*[numFields]uint32{},
		norms:  map[uint32]*[numFields]uint32{},
	}
}

// AddDoc indexes one document. fields maps each field to its raw text; the
// builder analyzes each field, counts per-field term frequencies, and records the
// document's per-field token lengths for length normalization. docID is the
// dense id the document occupies in this shard.
func (b *Builder) AddDoc(docID uint32, fields map[Field]string) {
	if !b.hasDocs || docID > b.maxDoc {
		b.maxDoc = docID
		b.hasDocs = true
	}
	dn := b.norms[docID]
	if dn == nil {
		dn = &[numFields]uint32{}
		b.norms[docID] = dn
	}
	for f, text := range fields {
		toks := Analyze(text)
		dn[f] += uint32(len(toks))
		for _, tok := range toks {
			docs := b.terms[tok]
			if docs == nil {
				docs = map[uint32]*[numFields]uint32{}
				b.terms[tok] = docs
			}
			tf := docs[docID]
			if tf == nil {
				tf = &[numFields]uint32{}
				docs[docID] = tf
			}
			tf[f]++
		}
	}
}

// docCount returns N, the dense docID space size.
func (b *Builder) docCount() uint32 {
	if !b.hasDocs {
		return 0
	}
	return b.maxDoc + 1
}

// Build encodes the accumulated documents into a lexical region byte run.
func (b *Builder) Build() []byte {
	n := b.docCount()
	st := computeStats(n, b.norms)
	norms := normsTable(n, b.norms)

	// Sort terms; termID is dictionary position. The stream hands assembleRegion one
	// term at a time with its postings docID-sorted, the order the encoder expects.
	sorted := make([]string, 0, len(b.terms))
	for t := range b.terms {
		sorted = append(sorted, t)
	}
	sort.Strings(sorted)

	i := 0
	next := func() (string, []posting, bool) {
		if i >= len(sorted) {
			return "", nil, false
		}
		term := sorted[i]
		i++
		docs := b.terms[term]
		ps := make([]posting, 0, len(docs))
		for docID, tf := range docs {
			ps = append(ps, posting{docID: docID, fieldTF: *tf})
		}
		sort.Slice(ps, func(a, c int) bool { return ps[a].docID < ps[c].docID })
		return term, ps, true
	}

	return assembleRegion(n, norms, st, b.params, fieldLenFrom(b.norms), next)
}

// termSource yields terms in ascending dictionary order, each with its postings
// already sorted by docID. It returns ok false once the stream is exhausted. Both
// the in-memory builder and the SPIMI external-merge build drive assembleRegion
// through this one interface, so the two produce byte-identical regions.
type termSource func() (term string, ps []posting, ok bool)

// assembleRegion encodes a lexical region from a sorted term stream, the per-doc
// norms table, the shard statistics, and the BM25F params. It is the single encoder
// path: the in-memory Builder feeds it from its maps, the SpimiBuilder feeds it from
// an external k-way merge, and because both deliver terms in the same order with
// docID-sorted postings the output bytes match exactly. fieldLenOf reads a document's
// per-field lengths so the block-max bound can be computed at build time.
func assembleRegion(n uint32, norms []byte, st stats, params Params, fieldLenOf func(uint32) [numFields]uint32, next termSource) []byte {
	var postings []byte
	var blockMax []byte
	var sorted []string
	var entries []termEntry

	for {
		term, ps, ok := next()
		if !ok {
			break
		}
		ti := len(sorted)
		df := uint32(len(ps))

		entry := termEntry{
			termID:      uint32(ti),
			postingsOff: uint64(len(postings)),
			docFreq:     df,
			blockMaxOff: uint64(len(blockMax)),
		}

		var prevLast uint32
		var listMax int32
		var blockCount uint32
		for start := 0; start < len(ps); start += blockSize {
			end := start + blockSize
			if end > len(ps) {
				end = len(ps)
			}
			block := ps[start:end]
			// block-max: the largest idf-free term-frequency contribution any posting
			// in the block makes, rounded up so it is a true upper bound. It is stored
			// without idf so the query path can scale it by either the shard-local idf
			// or the collection-wide idf the broker pushes down; the cursor applies the
			// chosen idf at open. Computing the contribution with an idf of 1 leaves
			// exactly the field-weighted, length-normalized, saturation-capped term
			// component.
			var bmax int32
			for i := range block {
				fl := fieldLenOf(block[i].docID)
				c := quantizeCeil(contribution(1, &block[i].fieldTF, &fl, &st, &params))
				if c > bmax {
					bmax = c
				}
			}
			if bmax > listMax {
				listMax = bmax
			}
			blockMax = codec.AppendUint32(blockMax, uint32(bmax))
			postings = encodeBlock(postings, block, prevLast)
			prevLast = block[len(block)-1].docID
			blockCount++
		}

		entry.postingsLen = uint64(len(postings)) - entry.postingsOff
		entry.blockCount = blockCount
		entry.maxContrib = listMax

		sorted = append(sorted, term)
		entries = append(entries, entry)
	}

	dictBytes := encodeDict(sorted, entries)

	bf := newBloom(len(sorted), 0.01)
	for _, t := range sorted {
		bf.add(t)
	}
	bloomBytes := bf.encode()

	// Lay the parts out after the header and fill in the sub-offsets.
	h := regionHeader{
		flags:       flagIDFFreeBlockMax,
		termCount:   uint32(len(sorted)),
		docCount:    n,
		avgFieldLen: st.avgFieldLen,
		params:      params,
	}
	off := uint64(headerLen)
	h.bloomOff = off
	h.bloomLen = uint64(len(bloomBytes))
	off += h.bloomLen
	h.dictOff = off
	h.dictLen = uint64(len(dictBytes))
	off += h.dictLen
	h.blockMaxOff = off
	h.blockMaxLen = uint64(len(blockMax))
	off += h.blockMaxLen
	h.postingsOff = off
	h.postingsLen = uint64(len(postings))
	off += h.postingsLen
	h.normsOff = off
	h.normsLen = uint64(len(norms))

	out := h.encode()
	out = append(out, bloomBytes...)
	out = append(out, dictBytes...)
	out = append(out, blockMax...)
	out = append(out, postings...)
	out = append(out, norms...)
	return out
}

// computeStats derives the shard statistics from the per-doc norms: N and the
// average length of each field over the full dense docID space.
func computeStats(n uint32, norms map[uint32]*[numFields]uint32) stats {
	var totalLen [numFields]uint64
	for _, dn := range norms {
		for f := 0; f < numFields; f++ {
			totalLen[f] += uint64(dn[f])
		}
	}
	var st stats
	st.docCount = n
	if n > 0 {
		for f := 0; f < numFields; f++ {
			st.avgFieldLen[f] = float64(totalLen[f]) / float64(n)
		}
	}
	return st
}

// normsTable encodes the fixed-width per-doc field-length table, one record per
// dense docID, the form the region stores and the scorer reads in O(1).
func normsTable(n uint32, norms map[uint32]*[numFields]uint32) []byte {
	out := make([]byte, int(n)*normRecord)
	for docID, dn := range norms {
		off := int(docID) * normRecord
		for f := 0; f < numFields; f++ {
			codec.PutUint32(out[off+f*4:], dn[f])
		}
	}
	return out
}

// fieldLenFrom returns a lookup over a norms map, the per-doc field lengths the
// block-max computation needs during a build.
func fieldLenFrom(norms map[uint32]*[numFields]uint32) func(uint32) [numFields]uint32 {
	return func(docID uint32) [numFields]uint32 {
		if dn := norms[docID]; dn != nil {
			return *dn
		}
		return [numFields]uint32{}
	}
}
