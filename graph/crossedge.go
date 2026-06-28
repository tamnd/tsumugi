package graph

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// This file implements the graph region's fourth part, the cross-shard edge list
// (doc 03 part 4, doc 06). At web scale almost every out-edge leaves the shard: a
// 2B-doc corpus split across 100k shards has, for any one page, nearly all of its
// out-neighbors living in some other shard, so the forward adjacency, which is
// keyed by local dense docID, cannot name them. The cross-shard list holds those
// far out-edges, keyed by the corpus-stable global node id the id table maps to and
// from, so a later pass can route each one to the shard that owns its target and
// turn it into an inbound edge there.
//
// The format is doc 06's "sequence of (dense source, count, gapped global
// targets)": for each local node that has any far out-edges, its dense docID, the
// number of targets, and the sorted global ids of those targets gap-encoded with
// zeta. The dense sources are themselves sorted and Elias-Fano coded so a reader
// random-accesses a node's record, and the per-record bit offsets are a second
// Elias-Fano array, exactly the structure the forward adjacency uses for its
// offsets. The whole list is empty (xsLen == 0) for a self-contained graph with no
// cross-shard edges, which is every region built before this slice.

// crossEdge is one buffered far out-edge: a local dense source and the global node
// id of its out-neighbor in another shard.
type crossEdge struct {
	from int
	to   uint64
}

// crossRecord is one source node's far out-edges after grouping: the dense docID
// and its target global ids, sorted ascending and deduped.
type crossRecord struct {
	source  int
	targets []uint64
}

// buildCrossBlob groups buffered cross edges by source, sorts and dedupes each
// source's targets, orders the records by dense source, and encodes them. It
// returns nil when there are no cross edges, the empty-list case the header records
// as xsLen == 0.
func buildCrossBlob(edges []crossEdge, p Params) []byte {
	if len(edges) == 0 {
		return nil
	}
	bySource := make(map[int][]uint64, len(edges))
	for _, e := range edges {
		bySource[e.from] = append(bySource[e.from], e.to)
	}
	sources := make([]int, 0, len(bySource))
	for s := range bySource {
		sources = append(sources, s)
	}
	sort.Ints(sources)
	recs := make([]crossRecord, 0, len(sources))
	for _, s := range sources {
		recs = append(recs, crossRecord{source: s, targets: sortDedupU64(bySource[s])})
	}
	return encodeCrossShard(recs, p)
}

// sortDedupU64 sorts ascending and removes duplicates in place.
func sortDedupU64(s []uint64) []uint64 {
	if len(s) == 0 {
		return s
	}
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	w := 1
	for i := 1; i < len(s); i++ {
		if s[i] != s[w-1] {
			s[w] = s[i]
			w++
		}
	}
	return s[:w]
}

// encodeCrossShard codes the records into the cross-shard blob: the count, the
// Elias-Fano array of sorted dense sources, the Elias-Fano array of the per-record
// bit offsets, and the bitstream of records. Each record is the degree as gamma,
// then the sorted target globals, the first as zeta(g) and each later one as
// zeta(gap-1) over the previous, so a run of nearby globals costs a handful of bits
// each. recs must be ordered by ascending source.
func encodeCrossShard(recs []crossRecord, p Params) []byte {
	if len(recs) == 0 {
		return nil
	}
	sources := make([]uint64, len(recs))
	offsets := make([]uint64, len(recs)+1)
	w := &bitWriter{}
	for i, r := range recs {
		offsets[i] = w.bits
		sources[i] = uint64(r.source)
		w.writeGamma(uint64(len(r.targets)))
		var prev uint64
		for j, g := range r.targets {
			if j == 0 {
				w.writeZeta(g, p.ZetaK)
			} else {
				w.writeZeta(g-prev-1, p.ZetaK)
			}
			prev = g
		}
	}
	offsets[len(recs)] = w.bits
	stream := w.finish()

	srcEF := buildEF(sources).encode()
	offEF := buildEF(offsets).encode()

	b := make([]byte, 0, 16+len(srcEF)+len(offEF)+len(stream))
	b = codec.AppendUint32(b, uint32(len(recs)))
	b = codec.AppendUint32(b, uint32(len(srcEF)))
	b = append(b, srcEF...)
	b = codec.AppendUint32(b, uint32(len(offEF)))
	b = append(b, offEF...)
	b = codec.AppendUint32(b, uint32(len(stream)))
	b = append(b, stream...)
	return b
}

// crossList is the parsed cross-shard edge list: the sorted dense sources, the
// per-record bit offsets, and the bitstream, all over the same region bytes.
type crossList struct {
	srcEF  *ef
	offEF  *ef
	stream []byte
	params Params
}

// decodeCrossList parses a cross-shard blob.
func decodeCrossList(b []byte, p Params) (*crossList, error) {
	if len(b) < 4 {
		return nil, ErrCorrupt
	}
	off := 4 // count, informational; srcEF.n is authoritative
	srcEF, off, err := readEFField(b, off)
	if err != nil {
		return nil, err
	}
	offEF, off, err := readEFField(b, off)
	if err != nil {
		return nil, err
	}
	if off+4 > len(b) {
		return nil, ErrCorrupt
	}
	sBytes := int(codec.Uint32(b[off:]))
	off += 4
	if off+sBytes > len(b) {
		return nil, ErrCorrupt
	}
	stream := b[off : off+sBytes]
	if offEF.n != srcEF.n+1 {
		return nil, ErrCorrupt
	}
	return &crossList{srcEF: srcEF, offEF: offEF, stream: stream, params: p}, nil
}

// readEFField reads a uint32 length prefix and the Elias-Fano array of that length,
// returning the array and the offset past it.
func readEFField(b []byte, off int) (*ef, int, error) {
	if off+4 > len(b) {
		return nil, off, ErrCorrupt
	}
	n := int(codec.Uint32(b[off:]))
	off += 4
	if off+n > len(b) {
		return nil, off, ErrCorrupt
	}
	e, err := decodeEF(b[off : off+n])
	if err != nil {
		return nil, off, ErrCorrupt
	}
	return e, off + n, nil
}

// recordAt decodes the target globals of the record at the given sorted rank.
func (c *crossList) recordAt(rank int) []uint64 {
	r := newBitReader(c.stream, c.offEF.get(rank))
	d := int(r.readGamma())
	out := make([]uint64, 0, d)
	var prev uint64
	for j := 0; j < d; j++ {
		var g uint64
		if j == 0 {
			g = r.readZeta(c.params.ZetaK)
		} else {
			g = prev + 1 + r.readZeta(c.params.ZetaK)
		}
		out = append(out, g)
		prev = g
	}
	return out
}

// neighbors returns the far out-neighbor globals of dense docID x, nil when x has
// none. It binary-searches the sorted dense sources, so a node without cross edges
// costs one Elias-Fano predecessor lookup and no decode.
func (c *crossList) neighbors(x int) []uint64 {
	rank := sort.Search(c.srcEF.n, func(i int) bool { return c.srcEF.get(i) >= uint64(x) })
	if rank >= c.srcEF.n || c.srcEF.get(rank) != uint64(x) {
		return nil
	}
	return c.recordAt(rank)
}

// forEach walks every source record in ascending dense-source order.
func (c *crossList) forEach(fn func(source int, targets []uint64)) {
	for rank := 0; rank < c.srcEF.n; rank++ {
		fn(int(c.srcEF.get(rank)), c.recordAt(rank))
	}
}

// InboundEdge is a cross-shard edge resolved against a target shard: a source node
// named by its corpus-stable global id, in some other shard, pointing at a local
// dense docID in the shard the edge was routed into.
type InboundEdge struct {
	Source uint64 // global node id of the source node, which lives in another shard
	Target int    // dense docID of the target node in the shard this edge routes into
}

// RouteCrossEdges joins the cross-shard edge lists of a set of shards against the
// shards' id tables (doc 06's resolution join). For every shard's far out-edges it
// resolves each target global id to the shard that owns it through that shard's
// Dense lookup, and records the edge there as inbound, the source kept as its global
// id (it lives in another shard) and the target as the now-local dense docID. The
// result is indexed by shard: result[i] is the edges that point into shard i. An
// edge whose target no shard in the set holds is dropped, which is how a target in a
// shard not present in this routing pass is left for a later one.
func RouteCrossEdges(shards []*Region) [][]InboundEdge {
	inbound := make([][]InboundEdge, len(shards))
	for si, s := range shards {
		if s == nil || s.xs == nil {
			continue
		}
		s.xs.forEach(func(source int, targets []uint64) {
			srcGlobal := s.Global(source)
			for _, g := range targets {
				for ti, t := range shards {
					if ti == si || t == nil {
						continue
					}
					if d, ok := t.Dense(g); ok {
						inbound[ti] = append(inbound[ti], InboundEdge{Source: srcGlobal, Target: d})
						break
					}
				}
			}
		})
	}
	return inbound
}
