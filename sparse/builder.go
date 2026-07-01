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
// weight range, quantizes, orders each term's postings impact-descending, groups
// them into fixed-count blocks, and writes the per-block leading (max) impact the
// anytime traversal prunes on.
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

	i := 0
	next := func() (string, []posting, bool) {
		if i >= len(names) {
			return "", nil, false
		}
		name := names[i]
		i++
		ps := b.terms[name]
		sort.Slice(ps, func(a, c int) bool {
			if ps[a].docID != ps[c].docID {
				return ps[a].docID < ps[c].docID
			}
			return ps[a].weight > ps[c].weight
		})
		// One impact per (term, docID). Postings are sorted by docID then weight
		// descending, so the first of each docID run is the strongest; drop the
		// rest. Deduping here means encodeTermBlocks sees each doc once, so a doc's
		// whole contribution for the term is a single impact in a single block.
		return name, dedupByDoc(ps), true
	}

	return assembleRegion(b.docCount, b.blockSize, q, next)
}

// termSource yields terms in ascending order, each with its postings already deduped
// to one per docID and sorted by docID, the form encodeTermBlocks expects. Both the
// in-memory Builder and the SPIMI external-merge build drive assembleRegion through
// this interface, so the two emit byte-identical regions.
type termSource func() (name string, ps []posting, ok bool)

// assembleRegion frames an IMP1 region from a sorted term stream, the dense docID
// space, the block width, and the quantizer the global weight range fixed. It is the
// single encoder path the in-memory and external-merge builds share.
func assembleRegion(docCount, blockSize uint32, q quantizer, next termSource) []byte {
	var idx, nameBlob, post []byte
	var termCount uint32
	for {
		name, ps, ok := next()
		if !ok {
			break
		}
		nameOff := uint32(len(nameBlob))
		nameBlob = append(nameBlob, name...)
		postOff := uint32(len(post))

		blockCount := encodeTermBlocks(&post, ps, q, blockSize)

		idx = codec.AppendUint32(idx, nameOff)
		idx = codec.AppendUint16(idx, uint16(len(name)))
		idx = codec.AppendUint32(idx, postOff)
		idx = codec.AppendUint32(idx, blockCount)
		termCount++
	}

	h := header{
		version:   regionVersion,
		blockSize: blockSize,
		termCount: termCount,
		docCount:  docCount,
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

// encodeTermBlocks writes one term's postings impact-descending and returns the
// block count. This is the doc-04 impact ordering: the postings are quantized,
// sorted by impact descending (docID ascending on a tie), and cut into fixed-count
// blocks of blockSize postings. Because the list is impact-sorted, each block's
// leading impact is its maximum and those leading impacts are monotone
// non-increasing down the list, so the traversal can walk blocks in descending
// bound order and stop once no unvisited block can lift a document past the top-k.
//
// A block header is the trailing (min) impact, the posting count, the leading
// (max) impact, and the body length; the body is the block's docIDs as zig-zag
// signed deltas (impact order is not docID monotone, so gaps run both ways)
// followed by one impact byte per posting. The body length lets a reader skip a
// whole block without decoding it, which is what makes the pruning pay off.
func encodeTermBlocks(dst *[]byte, ps []posting, q quantizer, blockSize uint32) uint32 {
	if len(ps) == 0 {
		return 0
	}
	type qp struct {
		docID  uint32
		impact uint8
	}
	qps := make([]qp, len(ps))
	for i, p := range ps {
		qps[i] = qp{docID: p.docID, impact: q.quantize(p.weight)}
	}
	sort.Slice(qps, func(a, c int) bool {
		if qps[a].impact != qps[c].impact {
			return qps[a].impact > qps[c].impact
		}
		return qps[a].docID < qps[c].docID
	})

	if blockSize == 0 {
		blockSize = DefaultBlockSize
	}
	var blocks uint32
	for start := 0; start < len(qps); start += int(blockSize) {
		end := start + int(blockSize)
		if end > len(qps) {
			end = len(qps)
		}
		blk := qps[start:end]
		lead := blk[0].impact            // block-max: first, since impact-descending
		minImp := blk[len(blk)-1].impact // block-min: trailing impact

		var body []byte
		prev := int64(0)
		for _, e := range blk {
			body = codec.AppendVarint(body, int64(e.docID)-prev)
			prev = int64(e.docID)
		}
		for _, e := range blk {
			body = append(body, e.impact)
		}

		*dst = codec.AppendUvarint(*dst, uint64(minImp))
		*dst = codec.AppendUvarint(*dst, uint64(len(blk)))
		*dst = append(*dst, lead)
		*dst = codec.AppendUvarint(*dst, uint64(len(body)))
		*dst = append(*dst, body...)
		blocks++
	}
	return blocks
}
