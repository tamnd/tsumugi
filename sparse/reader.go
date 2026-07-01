package sparse

import (
	"context"
	"errors"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// rangePreemptStride bounds how often the anytime block walk polls the query context for a
// passed deadline: once every stride+1 blocks, a power of two minus one so the test is one
// bitwise and. The poll stays off the hot path because reading a cancellable context's Err
// touches its mutex; polling every block would tax the common case where the budget holds
// and the walk runs to its anytime stop. A shard preempted at the deadline stops scoring
// blocks rather than finishing a walk no one will read.
const rangePreemptStride = 1023

// anytimeTailDivisor sets how deep a multi-term walk goes before it treats the
// unvisited blocks as tail. The walk stops once the block at the cursor cannot add
// more than threshold/anytimeTailDivisor to a document, so a larger divisor scans
// deeper for higher recall and a smaller one stops sooner for more skip. 64 clears
// recall@k of about 0.999 against the exhaustive oracle on a learned-sparse corpus
// while still skipping the deep tail; the recall sweep in the tests is the gate.
const anytimeTailDivisor = 64

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

// block is one impact-descending posting block of a term. The header (lead, min,
// count) is parsed up front so the pruning bound is cheap; body holds the
// still-encoded docID deltas and impact bytes, decoded only for blocks the
// traversal actually visits. lead is the block's leading (max) impact, the upper
// bound the walk prunes on; min is its trailing impact.
type block struct {
	lead  uint8
	min   uint8
	count uint32
	body  []byte
}

// decode expands the lazy body into docIDs and impacts. docIDs are zig-zag signed
// deltas because impact order is not docID monotone, so a delta can be negative.
func (blk block) decode() (docIDs []uint32, impacts []uint8) {
	docIDs = make([]uint32, blk.count)
	pos := 0
	var prev int64
	for k := uint32(0); k < blk.count; k++ {
		delta, n := codec.Varint(blk.body[pos:])
		pos += n
		prev += delta
		docIDs[k] = uint32(prev)
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
		mn, n := codec.Uvarint(b[pos:])
		pos += n
		cnt, n := codec.Uvarint(b[pos:])
		pos += n
		lead := b[pos]
		pos++
		bodyLen, n := codec.Uvarint(b[pos:])
		pos += n
		blocks[j] = block{lead: lead, min: uint8(mn), count: uint32(cnt), body: b[pos : pos+int(bodyLen)]}
		pos += int(bodyLen)
	}
	return blocks
}

// Result is one retrieved document with its integer impact-sum score.
type Result struct {
	DocID uint32
	Score int64
}

// queryTerm pairs a query weight with a term's impact-descending blocks.
type queryTerm struct {
	weight int64
	blocks []block
}

// blockRef points at one query-term block with the score bound it contributes: the
// term's query weight times the block's leading (max) impact. The traversal visits
// blocks in descending bound order, so the bound of the block at the cursor is the
// most any unvisited block can add to a document from here on.
type blockRef struct {
	qi    int
	bi    int
	bound int64
}

// Search returns the top-k by summed weighted impact, ordered by score descending
// then docID ascending. It runs the impact-ordered anytime traversal with an
// un-budgeted context that never trips.
func (r *Region) Search(query map[string]int, k int) []Result {
	res, _, _ := r.searchCore(context.Background(), query, k)
	return res
}

// SearchCtx is Search threaded with the query's context: the anytime block walk polls the
// context on a stride and abandons mid-walk if the deadline passes, returning
// completed=false. A shard preempted at the deadline on the broker fan-out stops scoring
// blocks in flight rather than finishing a walk whose result will be discarded; the partial
// top-k it returns on an abandoned walk is meant to be discarded. Search keeps the
// un-budgeted signature, delegating with a background context that never trips.
func (r *Region) SearchCtx(ctx context.Context, query map[string]int, k int) ([]Result, bool) {
	res, _, completed := r.searchCore(ctx, query, k)
	return res, completed
}

// searchStats runs the traversal with a background context and also reports how many
// blocks it decoded, the pruning-effectiveness signal the skip test asserts on.
func (r *Region) searchStats(query map[string]int, k int) ([]Result, int) {
	res, examined, _ := r.searchCore(context.Background(), query, k)
	return res, examined
}

// searchCore is the impact-ordered anytime traversal. Every query-term block is a
// work item bounded by weight*leadImpact; the items are visited in descending
// bound order, each accumulating weight*impact into a per-doc score and offering
// the doc to an indexed top-k that updates the doc in place as its partial grows.
//
// The stop bound is MaxScore-style. A document appears in at most one block per
// term (postings are deduped by doc), so the most any document can still gain from
// here on is the sum over query terms of that term's best remaining block bound.
// The walk keeps that running remaining ceiling and stops once the top-k is full
// and the ceiling cannot lift a fresh document past the weakest kept score.
//
// This is doc-04's approximate anytime cutoff. It is exact for a single-term query
// (a document's whole score is one block, and the ceiling collapses to that block's
// bound) and near-exact for multi-term, where a document partially scored and then
// beaten could in principle still climb past the ceiling with contributions that
// have not landed yet; there is no docID index to complete such a partial winner,
// so the cutoff trades a bounded recall loss for the skip. It returns the results,
// the blocks decoded, and whether the walk finished before the context tripped.
func (r *Region) searchCore(ctx context.Context, query map[string]int, k int) ([]Result, int, bool) {
	qts := r.loadQuery(query)
	if len(qts) == 0 || k <= 0 {
		return nil, 0, true
	}

	var work []blockRef
	for qi := range qts {
		w := qts[qi].weight
		for bi := range qts[qi].blocks {
			work = append(work, blockRef{qi: qi, bi: bi, bound: w * int64(qts[qi].blocks[bi].lead)})
		}
	}
	sort.Slice(work, func(i, j int) bool {
		if work[i].bound != work[j].bound {
			return work[i].bound > work[j].bound
		}
		if work[i].qi != work[j].qi {
			return work[i].qi < work[j].qi
		}
		return work[i].bi < work[j].bi
	})

	// Anytime cutoff. For a single-term query a document's whole score is one
	// block, so the block at the cursor is the most an unseen document can score;
	// stopping once that bound cannot beat the weakest kept score is exact. For a
	// multi-term query a document's score sums across impact-descending blocks with
	// no docID index to complete a partially-seen winner, so the exact bound would
	// never let the walk stop early. Instead the walk keeps going until the cursor
	// bound falls below threshold/anytimeTailDivisor, at which point the unvisited
	// blocks hold only tail impact; the sweep in the tests measures the recall this
	// buys against the exhaustive oracle (about 0.999 at the chosen divisor).
	multiTerm := len(qts) > 1
	tk := newAccTopK(k)
	acc := make(map[uint32]int64)
	examined := 0
	for i := range work {
		if (i&rangePreemptStride) == 0 && ctx.Err() != nil {
			return tk.sorted(), examined, false
		}
		if tk.full() {
			lim := tk.threshold()
			if multiTerm {
				lim /= anytimeTailDivisor
			}
			if work[i].bound < lim {
				break
			}
		}
		examined++
		qt := &qts[work[i].qi]
		docIDs, impacts := qt.blocks[work[i].bi].decode()
		for j, d := range docIDs {
			acc[d] += qt.weight * int64(impacts[j])
			tk.offer(d, acc[d])
		}
	}
	return tk.sorted(), examined, true
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
	tk := newAccTopK(k)
	for d, s := range acc {
		tk.offer(d, s)
	}
	return tk.sorted()
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
