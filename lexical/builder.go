package lexical

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// Builder accumulates documents into a BM25F lexical region. It is the in-memory
// M1 build: tokenize each document's four fields, invert into term-to-postings,
// then encode the dictionary, postings, block-max metadata, bloom filter, and
// the per-document field-length norms into one region byte run the container
// stores as RegionLexical. The external-sort SPIMI build for corpora past RAM is
// a later concern; this builds a shard that fits in memory, which is the unit a
// single .tsumugi file holds.
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

	// Shard-wide average field lengths over the full dense space.
	var totalLen [numFields]uint64
	for _, dn := range b.norms {
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

	// Fixed-width norms table, one record per dense docID.
	norms := make([]byte, int(n)*normRecord)
	for docID, dn := range b.norms {
		off := int(docID) * normRecord
		for f := 0; f < numFields; f++ {
			codec.PutUint32(norms[off+f*4:], dn[f])
		}
	}

	// Sort terms; termID is dictionary position.
	sorted := make([]string, 0, len(b.terms))
	for t := range b.terms {
		sorted = append(sorted, t)
	}
	sort.Strings(sorted)

	var postings []byte
	var blockMax []byte
	entries := make([]termEntry, len(sorted))

	for ti, term := range sorted {
		docs := b.terms[term]
		ps := make([]posting, 0, len(docs))
		for docID, tf := range docs {
			ps = append(ps, posting{docID: docID, fieldTF: *tf})
		}
		sort.Slice(ps, func(i, j int) bool { return ps[i].docID < ps[j].docID })

		df := uint32(len(ps))
		termIDF := idf(n, df)

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
			// block-max: the largest contribution any posting in the block makes,
			// rounded up so it is a true upper bound.
			var bmax int32
			for i := range block {
				fl := b.fieldLenOf(block[i].docID)
				c := quantizeCeil(contribution(termIDF, &block[i].fieldTF, &fl, &st, &b.params))
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
		entries[ti] = entry
	}

	dictBytes := encodeDict(sorted, entries)

	bf := newBloom(len(sorted), 0.01)
	for _, t := range sorted {
		bf.add(t)
	}
	bloomBytes := bf.encode()

	// Lay the parts out after the header and fill in the sub-offsets.
	h := regionHeader{
		flags:       0,
		termCount:   uint32(len(sorted)),
		docCount:    n,
		avgFieldLen: st.avgFieldLen,
		params:      b.params,
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

// fieldLenOf returns a document's per-field lengths during the build.
func (b *Builder) fieldLenOf(docID uint32) [numFields]uint32 {
	if dn := b.norms[docID]; dn != nil {
		return *dn
	}
	return [numFields]uint32{}
}
