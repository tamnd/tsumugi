package graph

import "sort"

// Region is a parsed, read-only graph region. It holds the two adjacency
// bitstreams and their offset indexes; neighbor lists are decoded on demand and
// nothing is held expanded. It also holds the dense-to-global identity mapping:
// either the nodeBase of a contiguous run, or the explicit id table when the
// global ids do not line up with the dense order.
type Region struct {
	params    Params
	nodeCount int
	edgeCount uint64
	nodeBase  uint64
	idt       *idTable // nil on the contiguous fast path; dense docID d maps to nodeBase+d
	fwdAdj    []byte
	fwdEF     *ef
	xpAdj     []byte
	xpEF      *ef
}

// Open parses a graph region from its bytes.
func Open(b []byte) (*Region, error) {
	h, err := decodeHeader(b)
	if err != nil {
		return nil, err
	}
	off := headerLen
	var idt *idTable
	if h.idTableLen > 0 {
		end := off + int(h.idTableLen)
		if end > len(b) {
			return nil, ErrCorrupt
		}
		idt, err = decodeIDTable(b[off:end])
		if err != nil {
			return nil, err
		}
		off = end
	}
	end := off + int(h.fwdAdjLen)
	if end > len(b) {
		return nil, ErrCorrupt
	}
	fwdAdj := b[off:end]
	off = end
	end = off + int(h.fwdEFLen)
	if end > len(b) {
		return nil, ErrCorrupt
	}
	fwdEF, err := decodeEF(b[off:end])
	if err != nil {
		return nil, ErrCorrupt
	}
	off = end
	end = off + int(h.xpAdjLen)
	if end > len(b) {
		return nil, ErrCorrupt
	}
	xpAdj := b[off:end]
	off = end
	end = off + int(h.xpEFLen)
	if end > len(b) {
		return nil, ErrCorrupt
	}
	xpEF, err := decodeEF(b[off:end])
	if err != nil {
		return nil, ErrCorrupt
	}
	return &Region{
		params:    h.params,
		nodeCount: int(h.nodeCount),
		edgeCount: h.edgeCount,
		nodeBase:  h.nodeBase,
		idt:       idt,
		fwdAdj:    fwdAdj,
		fwdEF:     fwdEF,
		xpAdj:     xpAdj,
		xpEF:      xpEF,
	}, nil
}

// NodeCount returns N, the dense node space size.
func (g *Region) NodeCount() int { return g.nodeCount }

// Global returns the global node id of dense docID d, the corpus-stable identity a
// cross-shard edge keys by. On the contiguous fast path it is nodeBase+d; otherwise
// the id table resolves it.
func (g *Region) Global(d int) uint64 {
	if g.idt == nil {
		return g.nodeBase + uint64(d)
	}
	return g.idt.global(d)
}

// Dense resolves a global node id to its dense docID in this shard, returning false
// when the shard does not hold that node, which is how an inbound cross-shard edge
// to a node outside this shard is rejected. On the contiguous fast path it is the
// range check and one subtraction; otherwise the id table binary-searches.
func (g *Region) Dense(global uint64) (int, bool) {
	if g.idt == nil {
		if global < g.nodeBase {
			return 0, false
		}
		d := global - g.nodeBase
		if d >= uint64(g.nodeCount) {
			return 0, false
		}
		return int(d), true
	}
	return g.idt.dense(global)
}

// EdgeCount returns the number of directed edges.
func (g *Region) EdgeCount() uint64 { return g.edgeCount }

// OutNeighbors decodes the out-list (links from x) on demand.
func (g *Region) OutNeighbors(x int) []int {
	return g.decode(g.fwdAdj, g.fwdEF, g.OutNeighbors, x)
}

// InNeighbors decodes the in-list (links to x) on demand from the transpose.
func (g *Region) InNeighbors(x int) []int {
	return g.decode(g.xpAdj, g.xpEF, g.InNeighbors, x)
}

// OutDegree reads just the degree code of x's out-list, without expanding it.
func (g *Region) OutDegree(x int) int {
	return int(newBitReader(g.fwdAdj, g.fwdEF.get(x)).readGamma())
}

// InDegree reads just the degree code of x's in-list.
func (g *Region) InDegree(x int) int {
	return int(newBitReader(g.xpAdj, g.xpEF.get(x)).readGamma())
}

// decode expands one adjacency record. neighborsOf resolves a referenced node's
// list within the same plane, bounded by the reference-chain cap so the
// recursion is O(1) deep.
func (g *Region) decode(adj []byte, idx *ef, neighborsOf func(int) []int, x int) []int {
	r := newBitReader(adj, idx.get(x))
	d := int(r.readGamma())
	if d == 0 {
		return nil
	}
	ref := int(r.readGamma())
	var copied []int
	if ref > 0 {
		nb := int(r.readGamma())
		runs := make([]int, nb)
		for i := range runs {
			runs[i] = int(r.readGamma())
		}
		refList := neighborsOf(x - ref)
		copied = applyRunsInt(refList, runs)
	}
	ni := int(r.readGamma())
	out := make([]int, 0, d)
	out = append(out, copied...)
	intervalCount := 0
	prevRight := 0
	for j := 0; j < ni; j++ {
		var left int
		if j == 0 {
			left = x + int(r.readSignedGamma())
		} else {
			left = prevRight + 1 + int(r.readGamma())
		}
		length := int(r.readGamma()) + g.params.LMin
		for k := 0; k < length; k++ {
			out = append(out, left+k)
		}
		intervalCount += length
		prevRight = left + length - 1
	}
	residCount := d - len(copied) - intervalCount
	prev := 0
	for i := 0; i < residCount; i++ {
		var v int
		if i == 0 {
			v = x + int(r.readSignedZeta(g.params.ZetaK))
		} else {
			v = prev + 1 + int(r.readZeta(g.params.ZetaK))
		}
		out = append(out, v)
		prev = v
	}
	sort.Ints(out)
	return out
}

// applyRunsInt is applyRuns over the int lists the decoder works in.
func applyRunsInt(ref []int, runs []int) []int {
	var out []int
	idx := 0
	phase := true
	for _, rl := range runs {
		if phase {
			out = append(out, ref[idx:idx+rl]...)
		}
		idx += rl
		phase = !phase
	}
	return out
}
