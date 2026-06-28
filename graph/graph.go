// Package graph implements the .tsumugi graph region: the web link graph for one
// shard, compressed in the Boldi-Vigna style to a few bits an edge, with random
// access to any node's neighbors. It stores both the forward adjacency (who a
// page links to) and the transpose (who links to a page), because the offline
// signal computation in M5, PageRank above all, iterates over in-links far more
// than out-links, and with the transpose stored that iteration is a streaming
// pass rather than a scatter.
//
// Each node's neighbor list is one self-delimiting bitstream record: the degree,
// an optional reference to a nearby node's list with a copy mask, runs of
// consecutive ids as intervals, and the rest as zeta-coded gaps. Node ids are
// dense docIDs in [0, N); the ordering that makes the gaps small is the build's
// concern, this package just encodes whatever order it is given. The records sit
// back to back with no separators, and an Elias-Fano offset index gives the bit
// position of each one for random access. The bytes are the region's GRA1 format,
// framed by the M0 container as RegionGraph.
package graph

import (
	"errors"

	"github.com/tamnd/tsumugi/codec"
)

const regionMagic = "GRA1"

const regionVersion = 2

// Params tune the adjacency coder. The defaults match the WebGraph settings that
// land near three bits an edge on a well-ordered web graph: reference within a
// small window, a bounded reference chain so a decode stays O(1), zeta with k=3
// for the residual gaps, and a minimum run of two to be worth an interval.
type Params struct {
	Window int // how far back a node may reference, default 7
	MaxRef int // longest reference chain, bounds decode work, default 3
	ZetaK  int // zeta parameter for residual gaps, default 3
	LMin   int // shortest run encoded as an interval, default 2
}

// DefaultParams returns the canonical adjacency-coder settings.
func DefaultParams() Params {
	return Params{Window: 7, MaxRef: 3, ZetaK: 3, LMin: 2}
}

// ErrCorrupt is returned when the region bytes do not parse as a valid GRA1
// region or fail the header CRC.
var ErrCorrupt = errors.New("graph: corrupt region")

// header is the fixed prefix of a graph region. The sub-blobs follow it in the
// order doc 03 fixes for the region's parts: the id table, the forward adjacency,
// the forward offsets, the transpose adjacency, and the transpose offsets. The
// header carries every blob's length so a reader slices them by walking, plus the
// node_base of the dense-to-global mapping: when the id table is absent
// (idTableLen == 0) the dense space is a contiguous run of node ids and dense
// docID d maps to global node id nodeBase+d with no table, the fast path doc 02
// describes; when the global ids do not line up with the dense order the id table
// holds the mapping and nodeBase is unused.
type header struct {
	version    uint8
	params     Params
	nodeCount  uint32
	edgeCount  uint64
	idTableLen uint64
	nodeBase   uint64
	fwdAdjLen  uint64
	fwdEFLen   uint64
	xpAdjLen   uint64
	xpEFLen    uint64
}

const headerLen = 4 + 1 + 4 + 4 + 8 + 8 + 8 + 8*4 + 4 // magic..crc, see encode

func (h header) encode() []byte {
	b := make([]byte, 0, headerLen)
	b = append(b, regionMagic...)
	b = append(b, h.version)
	b = append(b, byte(h.params.Window), byte(h.params.MaxRef), byte(h.params.ZetaK), byte(h.params.LMin))
	b = codec.AppendUint32(b, h.nodeCount)
	b = codec.AppendUint64(b, h.edgeCount)
	b = codec.AppendUint64(b, h.idTableLen)
	b = codec.AppendUint64(b, h.nodeBase)
	b = codec.AppendUint64(b, h.fwdAdjLen)
	b = codec.AppendUint64(b, h.fwdEFLen)
	b = codec.AppendUint64(b, h.xpAdjLen)
	b = codec.AppendUint64(b, h.xpEFLen)
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
	h.params = Params{Window: int(b[5]), MaxRef: int(b[6]), ZetaK: int(b[7]), LMin: int(b[8])}
	h.nodeCount = codec.Uint32(b[9:])
	h.edgeCount = codec.Uint64(b[13:])
	h.idTableLen = codec.Uint64(b[21:])
	h.nodeBase = codec.Uint64(b[29:])
	h.fwdAdjLen = codec.Uint64(b[37:])
	h.fwdEFLen = codec.Uint64(b[45:])
	h.xpAdjLen = codec.Uint64(b[53:])
	h.xpEFLen = codec.Uint64(b[61:])
	return h, nil
}
