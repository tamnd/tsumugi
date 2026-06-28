// Package vector implements the .tsumugi dense retrieval region, the optional
// second candidate plane that recalls documents by meaning rather than shared
// words. It is the VEC1 region: RaBitQ one-bit codes plus per-vector scalars for
// a memory-light estimator, an int8 scalar-quantized copy for a sharp rerank, and
// an HNSW graph so the search visits a few thousand vectors instead of a few
// million. Document vectors are produced offline; this package owns the codec, the
// index, and the search, not the embedding model.
//
// The graph is built offline over the int8 dot of the scalar-quantized rotated
// vectors, an integer metric that tracks the exact dot to within a fraction of a
// percent at a fraction of the cost, so its edges connect true neighbors and the
// search needs only a narrow beam. It is the same metric the two-part search
// navigates with, so build and query agree. The int8 rows are dropped after the
// build unless the rerank copy is kept; only the links always persist. The search
// is a pipeline: walk the
// graph to gather a candidate set, navigating with the int8 copy in the two-part
// mode (or the asymmetric RaBitQ estimator over the one-bit code when a shard
// drops the int8 copy to save memory), rerank that small set sharply, and return
// the top dense results to fuse with the lexical plane by Reciprocal Rank Fusion.
// A symmetric one-bit Hamming walk was the first design, but at retrieval-grade
// recall it proved too coarse to navigate, so the one-bit code is kept for the
// estimator path and the int8 copy carries the two-part walk.
//
// Correctness here is recall, not bit-exactness: the ANN result is checked against
// a brute-force nearest neighbor scan and must recover the true neighbors with
// high probability.
//
// The lineage is RaBitQ and HNSW; this is a self-contained native implementation,
// no import edge, so a fresh clone builds.
package vector

import (
	"errors"
	"math"

	"github.com/tamnd/tsumugi/codec"
)

const regionMagic = "VEC1"

const regionVersion = 1

// Defaults are the canon settings: HNSW M of 16, M0 of 32, efConstruction 200,
// and an efSearch at the recall knee.
const (
	DefaultM              = 16
	DefaultM0             = 32
	DefaultEfConstruction = 200
	DefaultEfSearch       = 128
	DefaultRerankDepth    = 100
)

// Flag bits in the header flags word.
const (
	flagNormalized = 1 << 0
	flagHasRerank  = 1 << 1
	// flagMultibit marks the Extended-RaBitQ no-rerank form: the codes part holds
	// codeBits-wide quantized levels per dimension instead of one sign bit, and there is
	// no int8 rerank copy. The asymmetric estimator over the multi-bit code is sharp enough
	// to rank without it, the half-kilobyte path of spec 05.
	flagMultibit = 1 << 2
)

// ErrCorrupt is returned when the region bytes do not parse as a valid VEC1
// region or fail the header CRC.
var ErrCorrupt = errors.New("vector: corrupt region")

// ErrUnreachable is returned by Build when the graph cannot be made fully
// reachable from the entry point. The build repairs orphaned nodes before framing
// the region, so this is a defensive guard against a repair that fails to converge,
// not a condition a normal build hits: an orphaned node is a document invisible to
// dense search, a silent recall hole, so the build refuses to ship one.
var ErrUnreachable = errors.New("vector: graph not fully reachable from entry")

// header is the fixed prefix that fully parameterizes the reader. The three parts
// (codes, int8 rerank, links) follow it in order at the recorded lengths.
type header struct {
	version        uint8
	codeBits       uint8
	flags          uint32
	dimKept        uint32
	rdim           uint32
	count          uint32
	m              uint32
	m0             uint32
	entryPoint     uint32
	maxLayer       uint32
	efConstruction uint32
	rotationSeed   int64
	i8scale        float32
	codeStride     uint32
	rerankStride   uint32
	codesLen       uint64
	rerankLen      uint64
	linksLen       uint64
}

const headerLen = 92

func (h header) encode() []byte {
	b := make([]byte, 0, headerLen)
	b = append(b, regionMagic...)
	// byte 5 carries codeBits (1 for the one-bit code, 4 or 5 for the multi-bit code); the
	// two bytes after stay reserved. The old version stored a zero here, which decodes as
	// codeBits 0 and is normalized to 1, so the field is backward compatible.
	b = append(b, h.version, h.codeBits, 0, 0)
	b = codec.AppendUint32(b, h.flags)
	b = codec.AppendUint32(b, h.dimKept)
	b = codec.AppendUint32(b, h.rdim)
	b = codec.AppendUint32(b, h.count)
	b = codec.AppendUint32(b, h.m)
	b = codec.AppendUint32(b, h.m0)
	b = codec.AppendUint32(b, h.entryPoint)
	b = codec.AppendUint32(b, h.maxLayer)
	b = codec.AppendUint32(b, h.efConstruction)
	b = codec.AppendUint64(b, uint64(h.rotationSeed))
	b = codec.AppendUint32(b, math.Float32bits(h.i8scale))
	b = codec.AppendUint32(b, h.codeStride)
	b = codec.AppendUint32(b, h.rerankStride)
	b = codec.AppendUint64(b, h.codesLen)
	b = codec.AppendUint64(b, h.rerankLen)
	b = codec.AppendUint64(b, h.linksLen)
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
	h.codeBits = b[5]
	if h.codeBits == 0 {
		h.codeBits = 1
	}
	h.flags = codec.Uint32(b[8:])
	h.dimKept = codec.Uint32(b[12:])
	h.rdim = codec.Uint32(b[16:])
	h.count = codec.Uint32(b[20:])
	h.m = codec.Uint32(b[24:])
	h.m0 = codec.Uint32(b[28:])
	h.entryPoint = codec.Uint32(b[32:])
	h.maxLayer = codec.Uint32(b[36:])
	h.efConstruction = codec.Uint32(b[40:])
	h.rotationSeed = int64(codec.Uint64(b[44:]))
	h.i8scale = math.Float32frombits(codec.Uint32(b[52:]))
	h.codeStride = codec.Uint32(b[56:])
	h.rerankStride = codec.Uint32(b[60:])
	h.codesLen = codec.Uint64(b[64:])
	h.rerankLen = codec.Uint64(b[72:])
	h.linksLen = codec.Uint64(b[80:])
	return h, nil
}
