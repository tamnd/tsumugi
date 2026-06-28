package graph

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// The id table is the graph region's identity layer: the map between a shard's
// dense docIDs, the [0, N) indices every per-document array uses, and the global
// node ids, the 64-bit corpus-stable identities the link graph keys cross-shard
// edges and dedup by (doc 02's three-ID model). A graph reader that follows an
// edge into another shard, or merges results across shards, converts at this
// boundary and nowhere else.
//
// Doc 02 names two cases. When the build assigns global ids so a shard's documents
// occupy a contiguous run, dense docID d is just nodeBase+d and the inverse is one
// subtraction, no table: the region stores nodeBase in its header and no id table
// blob (idTableLen == 0). This is also the degenerate default, nodeBase == 0, where
// the global id equals the dense docID, which is what a single self-contained
// collection graph uses.
//
// When the global ids do not line up with the dense order, which is the common case
// because Recursive Graph Bisection orders dense ids against the link structure and
// not the id numbering, the region carries an explicit table. It is the sorted set
// of the shard's global node ids, Elias-Fano compressed because it is monotone
// (doc 03's "monotone uint64 array, EF compressed because it is sorted"), paired
// with the permutation that links a sorted rank back to its dense docID. The sorted
// array plus that permutation is exactly doc 02's "node-id-to-dense direction as a
// sorted Elias-Fano array, and the dense-to-node direction as its inverse": a global
// id resolves to a dense docID by a binary search over the monotone array to a rank
// and then the permutation, and a dense docID resolves to its global id by the
// inverse permutation and one array read.
type idTable struct {
	sorted      *ef      // the shard's global node ids, ascending; index is the sorted rank
	rankToDense []uint32 // rankToDense[rank] is the dense docID whose global id is sorted.get(rank)
	denseToRank []uint32 // inverse of rankToDense, built at decode for the dense->global direction
}

// computeIDTable decides the dense-to-global representation for one shard from the
// per-dense-docID global ids. It returns the header's nodeBase and the id table
// blob to frame after the header. When ids is nil, or when the ids form a
// contiguous run that dense order already follows (ids[d] == base+d for every d),
// it returns that base and a nil blob: the contiguous fast path needs no table.
// Otherwise it returns nodeBase 0 and the encoded table.
func computeIDTable(n int, ids []uint64) (nodeBase uint64, blob []byte) {
	if n == 0 || len(ids) == 0 {
		return 0, nil
	}
	if base, ok := contiguousBase(ids); ok {
		return base, nil
	}
	return 0, buildIDTable(ids).encode()
}

// contiguousBase reports whether ids is a contiguous run that the dense order walks
// in step, ids[d] == ids[0]+d for every d, and returns that base. This is the
// node_base fast path: when it holds the dense-to-global map is one add and the
// inverse one subtract, so no table is stored.
func contiguousBase(ids []uint64) (uint64, bool) {
	base := ids[0]
	for d, g := range ids {
		if g != base+uint64(d) {
			return 0, false
		}
	}
	return base, true
}

// buildIDTable constructs the explicit table from the per-dense-docID global ids:
// the sorted global ids as the Elias-Fano array, and the permutation from each
// sorted rank back to the dense docID that owns it.
func buildIDTable(ids []uint64) *idTable {
	n := len(ids)
	order := make([]uint32, n)
	for d := range order {
		order[d] = uint32(d)
	}
	sort.Slice(order, func(i, j int) bool { return ids[order[i]] < ids[order[j]] })
	sorted := make([]uint64, n)
	for rank, d := range order {
		sorted[rank] = ids[d]
	}
	t := &idTable{sorted: buildEF(sorted), rankToDense: order}
	t.buildInverse()
	return t
}

// buildInverse fills denseToRank from rankToDense, the resident map the
// dense-to-global direction reads. It is rebuilt at decode rather than stored,
// since it is a pure inverse of the permutation already on disk.
func (t *idTable) buildInverse() {
	t.denseToRank = make([]uint32, len(t.rankToDense))
	for rank, d := range t.rankToDense {
		t.denseToRank[d] = uint32(rank)
	}
}

// global returns the global node id of dense docID d: its rank, then the sorted
// array read.
func (t *idTable) global(d int) uint64 {
	return t.sorted.get(int(t.denseToRank[d]))
}

// dense resolves a global node id to its dense docID by a binary search over the
// monotone sorted array, then the rank-to-dense permutation. It returns false when
// the id is not one the shard holds, which is how an inbound cross-shard edge to a
// node this shard does not own is rejected.
func (t *idTable) dense(g uint64) (int, bool) {
	n := t.sorted.n
	rank := sort.Search(n, func(i int) bool { return t.sorted.get(i) >= g })
	if rank >= n || t.sorted.get(rank) != g {
		return 0, false
	}
	return int(t.rankToDense[rank]), true
}

// encode serializes the table: the Elias-Fano sorted array, then the rank-to-dense
// permutation as fixed uint32s. The denseToRank inverse is rebuilt at decode.
func (t *idTable) encode() []byte {
	efBytes := t.sorted.encode()
	b := make([]byte, 0, 8+len(efBytes)+len(t.rankToDense)*4)
	b = codec.AppendUint32(b, uint32(len(efBytes)))
	b = append(b, efBytes...)
	b = codec.AppendUint32(b, uint32(len(t.rankToDense)))
	for _, d := range t.rankToDense {
		b = codec.AppendUint32(b, d)
	}
	return b
}

// decodeIDTable parses an id table blob and rebuilds its denseToRank inverse.
func decodeIDTable(b []byte) (*idTable, error) {
	if len(b) < 4 {
		return nil, ErrCorrupt
	}
	efLen := int(codec.Uint32(b))
	off := 4
	if len(b) < off+efLen+4 {
		return nil, ErrCorrupt
	}
	sorted, err := decodeEF(b[off : off+efLen])
	if err != nil {
		return nil, ErrCorrupt
	}
	off += efLen
	cnt := int(codec.Uint32(b[off:]))
	off += 4
	if len(b) < off+cnt*4 {
		return nil, ErrCorrupt
	}
	rankToDense := make([]uint32, cnt)
	for i := 0; i < cnt; i++ {
		rankToDense[i] = codec.Uint32(b[off:])
		off += 4
	}
	t := &idTable{sorted: sorted, rankToDense: rankToDense}
	t.buildInverse()
	return t, nil
}
