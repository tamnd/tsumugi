package sparse

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// posting is one raw (docID, weight) before quantization.
type posting struct {
	docID  uint32
	weight float64
}

// Builder accumulates impact postings and encodes them into an IMP1 region.
// Postings arrive as (term, docID, weight) in any order; Build finds the global
// weight range, quantizes, groups each term's postings into docID-aligned blocks,
// and writes the per-block max impact the BMP traversal prunes on.
type Builder struct {
	terms     map[string][]posting
	docCount  uint32
	blockSize uint32
}

// NewBuilder returns a builder over a dense docID space of size docCount.
func NewBuilder(docCount uint32) *Builder {
	return &Builder{terms: map[string][]posting{}, docCount: docCount, blockSize: DefaultBlockSize}
}

// WithBlockSize overrides the range width before any postings are added.
func (b *Builder) WithBlockSize(n uint32) *Builder {
	b.blockSize = n
	return b
}

// Add records a learned impact weight for a term in a document.
func (b *Builder) Add(term string, docID uint32, weight float64) {
	if weight <= 0 || docID >= b.docCount {
		return
	}
	b.terms[term] = append(b.terms[term], posting{docID: docID, weight: weight})
}

// Build quantizes every posting and frames the region.
func (b *Builder) Build() []byte {
	// Global weight range for the log quantizer.
	wmin, wmax := 0.0, 0.0
	first := true
	for _, ps := range b.terms {
		for _, p := range ps {
			if first || p.weight < wmin {
				wmin = p.weight
			}
			if first || p.weight > wmax {
				wmax = p.weight
			}
			first = false
		}
	}
	q := newQuantizer(wmin, wmax)

	// Sorted term list for binary search at query time.
	names := make([]string, 0, len(b.terms))
	for t := range b.terms {
		names = append(names, t)
	}
	sort.Strings(names)

	var idx, nameBlob, post []byte
	for _, name := range names {
		ps := b.terms[name]
		sort.Slice(ps, func(i, j int) bool {
			if ps[i].docID != ps[j].docID {
				return ps[i].docID < ps[j].docID
			}
			return ps[i].weight > ps[j].weight
		})
		// One impact per (term, docID). Postings are sorted by docID then weight
		// descending, so the first of each docID run is the strongest; drop the
		// rest. Duplicates would double-count a doc past its own block-max bound
		// and break the pruned == exhaustive guarantee.
		ps = dedupByDoc(ps)

		nameOff := uint32(len(nameBlob))
		nameBlob = append(nameBlob, name...)
		postOff := uint32(len(post))

		blockCount := encodeTermBlocks(&post, ps, q, b.blockSize)

		idx = codec.AppendUint32(idx, nameOff)
		idx = codec.AppendUint16(idx, uint16(len(name)))
		idx = codec.AppendUint32(idx, postOff)
		idx = codec.AppendUint32(idx, blockCount)
	}

	h := header{
		version:   regionVersion,
		blockSize: b.blockSize,
		termCount: uint32(len(names)),
		docCount:  b.docCount,
		lnMin:     q.lnMin,
		lnMax:     q.lnMax,
		idxLen:    uint64(len(idx)),
		nameLen:   uint64(len(nameBlob)),
		postLen:   uint64(len(post)),
	}
	region := h.encode()
	region = append(region, idx...)
	region = append(region, nameBlob...)
	region = append(region, post...)
	return region
}

// dedupByDoc keeps one posting per docID, the first of each run, given input
// sorted by docID then weight descending.
func dedupByDoc(ps []posting) []posting {
	if len(ps) == 0 {
		return ps
	}
	out := ps[:1]
	for _, p := range ps[1:] {
		if p.docID != out[len(out)-1].docID {
			out = append(out, p)
		}
	}
	return out
}

// encodeTermBlocks writes one term's postings as docID-aligned blocks and returns
// the block count. Each block covers the range [blockID*B, (blockID+1)*B); its
// header is the block id, the posting count, and the block-max impact (the
// largest quantized impact in the block, the BMP upper bound for the range).
func encodeTermBlocks(dst *[]byte, ps []posting, q quantizer, blockSize uint32) uint32 {
	type blk struct {
		id      uint32
		docIDs  []uint32
		impacts []uint8
		max     uint8
	}
	var blocks []blk
	var cur *blk
	for _, p := range ps {
		id := p.docID / blockSize
		if cur == nil || cur.id != id {
			blocks = append(blocks, blk{id: id})
			cur = &blocks[len(blocks)-1]
		}
		imp := q.quantize(p.weight)
		cur.docIDs = append(cur.docIDs, p.docID)
		cur.impacts = append(cur.impacts, imp)
		if imp > cur.max {
			cur.max = imp
		}
	}
	for _, blk := range blocks {
		// Encode the body (gap varints then impacts) first so its byte length can
		// be written in the header. The length lets a reader skip a whole block
		// without decoding it, which is what makes BMP pruning pay off.
		var body []byte
		prev := uint32(0)
		for i, d := range blk.docIDs {
			if i == 0 {
				body = codec.AppendUvarint(body, uint64(d))
			} else {
				body = codec.AppendUvarint(body, uint64(d-prev))
			}
			prev = d
		}
		body = append(body, blk.impacts...)

		*dst = codec.AppendUvarint(*dst, uint64(blk.id))
		*dst = codec.AppendUvarint(*dst, uint64(len(blk.docIDs)))
		*dst = append(*dst, blk.max)
		*dst = codec.AppendUvarint(*dst, uint64(len(body)))
		*dst = append(*dst, body...)
	}
	return uint32(len(blocks))
}
