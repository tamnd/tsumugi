package sparse

import (
	"context"
	"errors"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// rangePreemptStride bounds how often the pruned range walk polls the query context for a
// passed deadline: once every stride+1 ranges, a power of two minus one so the test is one
// bitwise and. The poll stays off the hot path because reading a cancellable context's Err
// touches its mutex; polling every range would tax the common case where the budget holds
// and the walk runs to its anytime stop. A shard preempted at the deadline stops scoring
// ranges rather than finishing a walk no one will read.
const rangePreemptStride = 1023

// ErrCorrupt is returned when the region bytes do not parse as a valid IMP1
// region or fail the header CRC.
var ErrCorrupt = errors.New("sparse: corrupt region")

// Region is a parsed, read-only impact index.
type Region struct {
	quant     quantizer
	blockSize uint32
	docCount  uint32
	names     []string // sorted, for binary search
	idx       [][4]uint32
	post      []byte
}

// idx entry layout: [nameOff, nameLen, postOff, blockCount].

// Open parses an IMP1 region.
func Open(b []byte) (*Region, error) {
	h, err := decodeHeader(b)
	if err != nil {
		return nil, err
	}
	off := headerLen
	idxEnd := off + int(h.idxLen)
	nameEnd := idxEnd + int(h.nameLen)
	postEnd := nameEnd + int(h.postLen)
	if postEnd > len(b) || int(h.termCount)*14 != int(h.idxLen) {
		return nil, ErrCorrupt
	}
	idxBytes := b[off:idxEnd]
	nameBytes := b[idxEnd:nameEnd]
	post := b[nameEnd:postEnd]

	names := make([]string, h.termCount)
	idx := make([][4]uint32, h.termCount)
	for i := 0; i < int(h.termCount); i++ {
		e := idxBytes[i*14:]
		nameOff := codec.Uint32(e)
		nameLen := uint32(codec.Uint16(e[4:]))
		if int(nameOff+nameLen) > len(nameBytes) {
			return nil, ErrCorrupt
		}
		names[i] = string(nameBytes[nameOff : nameOff+nameLen])
		idx[i] = [4]uint32{nameOff, nameLen, codec.Uint32(e[6:]), codec.Uint32(e[10:])}
	}

	return &Region{
		quant:     quantizer{lnMin: h.lnMin, lnMax: h.lnMax},
		blockSize: h.blockSize,
		docCount:  h.docCount,
		names:     names,
		idx:       idx,
		post:      post,
	}, nil
}

// DocCount returns N.
func (r *Region) DocCount() uint32 { return r.docCount }

// block is one posting block of a term. The header (id, max, count) is parsed up
// front so the BMP bound is cheap; body holds the still-encoded docID gaps and
// impact bytes, decoded only for ranges the traversal actually visits.
type block struct {
	id    uint32
	max   uint8
	count uint32
	body  []byte
}

// decode expands the lazy body into docIDs and impacts.
func (blk block) decode() (docIDs []uint32, impacts []uint8) {
	docIDs = make([]uint32, blk.count)
	pos := 0
	var prev uint32
	for k := uint32(0); k < blk.count; k++ {
		d, n := codec.Uvarint(blk.body[pos:])
		pos += n
		if k == 0 {
			prev = uint32(d)
		} else {
			prev += uint32(d)
		}
		docIDs[k] = prev
	}
	impacts = blk.body[pos : pos+int(blk.count)]
	return docIDs, impacts
}

// termBlocks parses a term's block headers without decoding the bodies, or nil if
// the term is absent.
func (r *Region) termBlocks(term string) []block {
	i := sort.SearchStrings(r.names, term)
	if i >= len(r.names) || r.names[i] != term {
		return nil
	}
	postOff := r.idx[i][2]
	blockCount := r.idx[i][3]
	b := r.post[postOff:]
	pos := 0
	blocks := make([]block, blockCount)
	for j := uint32(0); j < blockCount; j++ {
		id, n := codec.Uvarint(b[pos:])
		pos += n
		cnt, n := codec.Uvarint(b[pos:])
		pos += n
		mx := b[pos]
		pos++
		bodyLen, n := codec.Uvarint(b[pos:])
		pos += n
		blocks[j] = block{id: uint32(id), max: mx, count: uint32(cnt), body: b[pos : pos+int(bodyLen)]}
		pos += int(bodyLen)
	}
	return blocks
}

// Result is one retrieved document with its integer impact-sum score.
type Result struct {
	DocID uint32
	Score int64
}

// queryTerm pairs a query weight with a term's blocks, kept in ascending range
// order so a range can be found by binary search. Building a per-range map up
// front would cost one insert per block even for the many ranges pruning never
// visits, so the slice plus search is the cheaper structure here.
type queryTerm struct {
	weight int64
	blocks []block
}

// blockAt returns the block covering range rid, or nil if the term has none.
func (qt queryTerm) blockAt(rid uint32) *block {
	bs := qt.blocks
	i := sort.Search(len(bs), func(i int) bool { return bs[i].id >= rid })
	if i < len(bs) && bs[i].id == rid {
		return &bs[i]
	}
	return nil
}

// rangeBound is one BMP range with its upper-bound score, sorted descending so
// the traversal can stop the moment a range cannot beat the kept top-k.
type rangeBound struct {
	rid   uint32
	bound int64
}

// Search runs Block-Max Pruning and returns the top-k by summed weighted impact,
// ordered by score descending then docID ascending. Query weights are small
// integers, one for inference-free retrieval, so the scoring is exact integer
// arithmetic and the result matches SearchExhaustive bit for bit.
func (r *Region) Search(query map[string]int, k int) []Result {
	res, _ := r.SearchCtx(context.Background(), query, k)
	return res
}

// SearchCtx is Search threaded with the query's context: the pruned range walk polls the
// context on a stride and abandons mid-walk if the deadline passes, returning
// completed=false. A shard preempted at the deadline on the broker fan-out stops scoring
// ranges in flight rather than finishing a walk whose result will be discarded; the
// partial top-k it returns on an abandoned walk is meant to be discarded. Search keeps the
// un-budgeted signature, delegating with a background context that never trips.
func (r *Region) SearchCtx(ctx context.Context, query map[string]int, k int) ([]Result, bool) {
	qts := r.loadQuery(query)
	if len(qts) == 0 || k <= 0 {
		return nil, true
	}

	// Per-range upper bound: sum over query terms of weight * block-max in range.
	boundOf := map[uint32]int64{}
	for _, qt := range qts {
		for i := range qt.blocks {
			boundOf[qt.blocks[i].id] += qt.weight * int64(qt.blocks[i].max)
		}
	}
	ranges := make([]rangeBound, 0, len(boundOf))
	for id, bound := range boundOf {
		ranges = append(ranges, rangeBound{rid: id, bound: bound})
	}
	sort.Slice(ranges, func(i, j int) bool {
		if ranges[i].bound != ranges[j].bound {
			return ranges[i].bound > ranges[j].bound
		}
		return ranges[i].rid < ranges[j].rid
	})

	h := &topK{k: k}
	scratch := make([]int64, r.blockSize)
	touched := make([]uint32, 0, r.blockSize)
	for i, rb := range ranges {
		if (i&rangePreemptStride) == 0 && ctx.Err() != nil {
			return h.sorted(), false
		}
		// Anytime stop: ranges are in descending bound order, so once a range
		// cannot beat the weakest kept candidate, none after it can either.
		if h.full() && rb.bound < h.threshold() {
			break
		}
		r.scoreRange(qts, rb.rid, scratch, &touched, h)
	}
	return h.sorted(), true
}

// scoreRange sums the exact weighted impact of every document in one range and
// offers each scored doc to the heap. Docs in range rid live in the contiguous
// span [rid*B, rid*B+B), so scores accumulate into a reusable offset-indexed
// scratch buffer instead of a fresh map per range, which is what keeps the pruned
// path allocation-free in its hot loop.
func (r *Region) scoreRange(qts []queryTerm, rid uint32, scratch []int64, touched *[]uint32, h *topK) {
	base := rid * r.blockSize
	*touched = (*touched)[:0]
	for _, qt := range qts {
		blk := qt.blockAt(rid)
		if blk == nil {
			continue
		}
		docIDs, impacts := blk.decode()
		for i, d := range docIDs {
			off := d - base
			if scratch[off] == 0 {
				*touched = append(*touched, off)
			}
			scratch[off] += qt.weight * int64(impacts[i])
		}
	}
	for _, off := range *touched {
		h.offer(Result{DocID: base + off, Score: scratch[off]})
		scratch[off] = 0
	}
}

// SearchExhaustive scores every document the brute-force way, the oracle the
// pruned Search is checked against.
func (r *Region) SearchExhaustive(query map[string]int, k int) []Result {
	qts := r.loadQuery(query)
	if len(qts) == 0 || k <= 0 {
		return nil
	}
	acc := map[uint32]int64{}
	for _, qt := range qts {
		for i := range qt.blocks {
			docIDs, impacts := qt.blocks[i].decode()
			for j, d := range docIDs {
				acc[d] += qt.weight * int64(impacts[j])
			}
		}
	}
	h := &topK{k: k}
	for d, s := range acc {
		h.offer(Result{DocID: d, Score: s})
	}
	return h.sorted()
}

func (r *Region) loadQuery(query map[string]int) []queryTerm {
	var qts []queryTerm
	for term, w := range query {
		if w == 0 {
			continue
		}
		blocks := r.termBlocks(term)
		if blocks == nil {
			continue
		}
		qts = append(qts, queryTerm{weight: int64(w), blocks: blocks})
	}
	return qts
}
