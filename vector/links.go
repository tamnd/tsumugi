package vector

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// The HNSW links part is split the way the spec lays it out: a dense layer-0
// block every node has, then a sparse directory and blob for the few nodes that
// reach above layer 0. Layout:
//
//	[uint32 upperCount]
//	[layer-0 block: N rows of (uint32 count, m0 * uint32 neighbor IDs)]
//	[directory: upperCount * (uint32 docID, uint32 topLayer, uint32 blobOff)]
//	[blob: per upper node, per layer 1..top, (uint32 count, count * uint32 IDs)]
//
// Layer-0 is fixed width and indexed by dense docID, so a neighbor lookup is one
// multiply and a read; the upper layers are rare, so they pay only for the nodes
// that have them.

func layer0RowWidth(m0 int) int { return 4 + 4*m0 }

// serializeLinks writes the links part for a built graph.
func serializeLinks(g *hnswGraph) []byte {
	n := len(g.links)
	rowW := layer0RowWidth(g.m0)

	// Gather the upper nodes in docID order.
	var upper []int32
	for node := range g.up {
		if len(g.up[node]) > 0 {
			upper = append(upper, int32(node))
		}
	}
	sort.Slice(upper, func(i, j int) bool { return upper[i] < upper[j] })

	// Build the blob and record each upper node's offset and top layer.
	var blob []byte
	type dirEnt struct{ docID, top, off uint32 }
	dir := make([]dirEnt, 0, len(upper))
	for _, node := range upper {
		layers := g.up[node]
		top := 0
		for l := range layers {
			if l > top {
				top = l
			}
		}
		off := uint32(len(blob))
		for l := 1; l <= top; l++ {
			ns := layers[l]
			blob = codec.AppendUint32(blob, uint32(len(ns)))
			for _, x := range ns {
				blob = codec.AppendUint32(blob, uint32(x))
			}
		}
		dir = append(dir, dirEnt{docID: uint32(node), top: uint32(top), off: off})
	}

	out := make([]byte, 0, 4+n*rowW+len(dir)*12+len(blob))
	out = codec.AppendUint32(out, uint32(len(dir)))
	for node := 0; node < n; node++ {
		ns := g.links[node]
		out = codec.AppendUint32(out, uint32(len(ns)))
		for i := 0; i < g.m0; i++ {
			if i < len(ns) {
				out = codec.AppendUint32(out, uint32(ns[i]))
			} else {
				out = codec.AppendUint32(out, 0)
			}
		}
	}
	for _, e := range dir {
		out = codec.AppendUint32(out, e.docID)
		out = codec.AppendUint32(out, e.top)
		out = codec.AppendUint32(out, e.off)
	}
	out = append(out, blob...)
	return out
}

// linksReader gives read-only neighbor access over a serialized links part.
type linksReader struct {
	n        int
	m0       int
	rowW     int
	upper    []byte // directory bytes
	upCount  int
	blob     []byte
	layer0   []byte
	dirIndex map[uint32]int // docID -> directory entry index
}

func openLinks(b []byte, n, m0 int) (*linksReader, error) {
	if len(b) < 4 {
		return nil, ErrCorrupt
	}
	upCount := int(codec.Uint32(b))
	rowW := layer0RowWidth(m0)
	l0Start := 4
	l0End := l0Start + n*rowW
	dirEnd := l0End + upCount*12
	if dirEnd > len(b) {
		return nil, ErrCorrupt
	}
	lr := &linksReader{
		n:        n,
		m0:       m0,
		rowW:     rowW,
		upCount:  upCount,
		layer0:   b[l0Start:l0End],
		upper:    b[l0End:dirEnd],
		blob:     b[dirEnd:],
		dirIndex: make(map[uint32]int, upCount),
	}
	for i := 0; i < upCount; i++ {
		docID := codec.Uint32(lr.upper[i*12:])
		lr.dirIndex[docID] = i
	}
	return lr, nil
}

// neighbors0 returns a node's layer-0 neighbors.
func (lr *linksReader) neighbors0(node int32) []int32 {
	row := lr.layer0[int(node)*lr.rowW:]
	count := int(codec.Uint32(row))
	if count > lr.m0 {
		count = lr.m0
	}
	out := make([]int32, count)
	for i := 0; i < count; i++ {
		out[i] = int32(codec.Uint32(row[4+i*4:]))
	}
	return out
}

// neighborsUpper returns a node's neighbors on a layer above 0, or nil.
func (lr *linksReader) neighborsUpper(node int32, layer int) []int32 {
	idx, ok := lr.dirIndex[uint32(node)]
	if !ok {
		return nil
	}
	ent := lr.upper[idx*12:]
	top := int(codec.Uint32(ent[4:]))
	if layer > top {
		return nil
	}
	off := codec.Uint32(ent[8:])
	pos := int(off)
	// Walk layers 1..layer to reach the wanted list.
	for l := 1; l <= top; l++ {
		count := int(codec.Uint32(lr.blob[pos:]))
		pos += 4
		if l == layer {
			out := make([]int32, count)
			for i := 0; i < count; i++ {
				out[i] = int32(codec.Uint32(lr.blob[pos+i*4:]))
			}
			return out
		}
		pos += count * 4
	}
	return nil
}
