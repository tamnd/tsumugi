// Package sparse implements the .tsumugi learned-sparse retrieval region: an
// impact-ordered index with an anytime traversal. Where the M1 lexical region
// scores BM25F over docID-ordered postings, this region stores a per-term
// per-document learned impact weight, quantized to one byte, and retrieves by
// summing the impacts of the query terms. The query carries small integer term
// weights (one for the inference-free case), so there is no transformer on the hot
// path; the heavy lifting happened offline when the doc-side model produced the
// impacts.
//
// Retrieval is the doc-04 impact ordering: each term's postings are stored
// impact-descending and cut into fixed-count blocks, so a block's leading impact
// is its maximum and those maxima fall monotonically down the list. Every
// query-term block becomes a work item bounded by weight*leadImpact; the items are
// walked in descending bound order, accumulating weight*impact per document, and
// the walk stops once the top-k is full and the block at the cursor cannot lift a
// fresh document past the weakest kept score. This anytime cutoff is exact for a
// single-term query, where a document's whole score lives in one block, and
// near-exact for multi-term queries, where a document's contributions scatter
// across impact-descending blocks with no docID index to complete a partially-seen
// winner; recall is the gate there, not bit-equality. SearchExhaustive keeps the
// exact oracle. The bytes are the IMP1 format, framed by the M0 container as
// RegionLexical with the impact flag set.
//
// The lineage is the lexical region and the impact-ordered / anytime retrieval
// literature; this is a self-contained native implementation, no import edge, so a
// fresh clone builds.
package sparse

import (
	"math"

	"github.com/tamnd/tsumugi/codec"
)

const regionMagic = "IMP1"

const regionVersion = 1

// DefaultBlockSize is the number of postings per impact-ordered block. A smaller
// block sharpens the leading-impact bound the traversal prunes on, at the cost of
// more block headers; 128 balances the two.
const DefaultBlockSize = 128

// quantizer maps a raw learned weight to one byte and back. Learned-sparse
// weights are heavy-tailed, so the levels are spread in log space: small weights
// get as much resolution as large ones. Level 0 is reserved for no weight.
type quantizer struct {
	lnMin float32
	lnMax float32
}

func newQuantizer(wmin, wmax float64) quantizer {
	if wmin <= 0 {
		wmin = math.SmallestNonzeroFloat64
	}
	if wmax < wmin {
		wmax = wmin
	}
	return quantizer{lnMin: float32(math.Log(wmin)), lnMax: float32(math.Log(wmax))}
}

// quantize maps a positive weight to a level in [1, 255]; non-positive maps to 0.
func (q quantizer) quantize(w float64) uint8 {
	if w <= 0 {
		return 0
	}
	lw := math.Log(w)
	span := float64(q.lnMax - q.lnMin)
	if span <= 0 {
		return 255
	}
	level := math.Round(1 + 254*(lw-float64(q.lnMin))/span)
	if level < 1 {
		return 1
	}
	if level > 255 {
		return 255
	}
	return uint8(level)
}

// dequant reverses quantize, for the rare path that wants a real weight back.
func (q quantizer) dequant(level uint8) float64 {
	if level == 0 {
		return 0
	}
	span := float64(q.lnMax - q.lnMin)
	lw := float64(q.lnMin) + float64(level-1)*span/254
	return math.Exp(lw)
}

// header is the fixed prefix of an IMP1 region. The three blobs follow it in
// order: the term index, the term names, and the postings.
type header struct {
	version   uint8
	blockSize uint32
	termCount uint32
	docCount  uint32
	lnMin     float32
	lnMax     float32
	idxLen    uint64
	nameLen   uint64
	postLen   uint64
}

const headerLen = 4 + 1 + 3 + 4 + 4 + 4 + 4 + 4 + 8*3 + 4

func (h header) encode() []byte {
	b := make([]byte, 0, headerLen)
	b = append(b, regionMagic...)
	b = append(b, h.version, 0, 0, 0)
	b = codec.AppendUint32(b, h.blockSize)
	b = codec.AppendUint32(b, h.termCount)
	b = codec.AppendUint32(b, h.docCount)
	b = codec.AppendUint32(b, math.Float32bits(h.lnMin))
	b = codec.AppendUint32(b, math.Float32bits(h.lnMax))
	b = codec.AppendUint64(b, h.idxLen)
	b = codec.AppendUint64(b, h.nameLen)
	b = codec.AppendUint64(b, h.postLen)
	b = codec.AppendUint32(b, codec.CRC32C(b))
	return b
}

func decodeHeader(b []byte) (header, error) {
	if len(b) < headerLen || string(b[0:4]) != regionMagic {
		return header{}, ErrCorrupt
	}
	if codec.Uint32(b[headerLen-4:]) != codec.CRC32C(b[:headerLen-4]) {
		return header{}, ErrCorrupt
	}
	var h header
	h.version = b[4]
	if h.version != regionVersion {
		return header{}, ErrCorrupt
	}
	h.blockSize = codec.Uint32(b[8:])
	h.termCount = codec.Uint32(b[12:])
	h.docCount = codec.Uint32(b[16:])
	h.lnMin = math.Float32frombits(codec.Uint32(b[20:]))
	h.lnMax = math.Float32frombits(codec.Uint32(b[24:]))
	h.idxLen = codec.Uint64(b[28:])
	h.nameLen = codec.Uint64(b[36:])
	h.postLen = codec.Uint64(b[44:])
	return h, nil
}
