package lexical

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// sentinel marks an exhausted cursor: a docID past every real one.
const sentinel = ^uint32(0)

// cursor walks one term's docID-ordered posting list. It decodes block bodies
// lazily and skips whole blocks using the per-block last-docID, so advancing far
// reads block headers but not the bodies it passes over.
type cursor struct {
	region       *Region
	info         termInfo
	list         []byte
	blockOffsets []int
	blockLast    []uint32
	blockMax     []int32 // per-block upper bound, already scaled by this term's idf
	scaledMax    int32   // list-wide upper bound, already scaled by this term's idf

	blkIdx      int
	blkPostings []posting
	pos         int
	cur         uint32
}

// openCursor builds a cursor over a term's list, parsing the block headers and
// the block-max array up front.
func (r *Region) openCursor(info termInfo) (*cursor, error) {
	e := info.entry
	list := r.postings[e.postingsOff : e.postingsOff+e.postingsLen]
	c := &cursor{region: r, info: info, list: list}

	off := 0
	for i := uint32(0); i < e.blockCount; i++ {
		h, err := readBlockHeader(list, off)
		if err != nil {
			return nil, err
		}
		c.blockOffsets = append(c.blockOffsets, off)
		c.blockLast = append(c.blockLast, h.lastDocID)
		off = h.nextOffset
	}

	// The stored bounds are idf-free; scale them by this term's idf once, here, so the
	// hot traversal compares already-scaled integers. The idf is shard-local on the
	// single-shard path and collection-wide when the broker pushed one down, but the
	// cursor does not care which: it scales whatever idf the termInfo carries.
	bm := r.blockMax[e.blockMaxOff:]
	c.blockMax = make([]int32, e.blockCount)
	for i := uint32(0); i < e.blockCount; i++ {
		if (int(i)+1)*4 > len(bm) {
			return nil, errCorrupt
		}
		c.blockMax[i] = scaleBound(info.idf, int32(codec.Uint32(bm[int(i)*4:])))
	}
	c.scaledMax = scaleBound(info.idf, e.maxContrib)

	if e.blockCount == 0 {
		c.cur = sentinel
		return c, nil
	}
	if err := c.loadBlock(0); err != nil {
		return nil, err
	}
	c.cur = c.blkPostings[0].docID
	return c, nil
}

func (c *cursor) loadBlock(i int) error {
	var prevLast uint32
	if i > 0 {
		prevLast = c.blockLast[i-1]
	}
	h, err := readBlockHeader(c.list, c.blockOffsets[i])
	if err != nil {
		return err
	}
	ps, err := decodeBlock(h, prevLast, c.region.codec)
	if err != nil {
		return err
	}
	c.blkPostings = ps
	c.blkIdx = i
	c.pos = 0
	return nil
}

// advance moves to the next posting, crossing a block boundary if needed.
func (c *cursor) advance() {
	c.pos++
	if c.pos < len(c.blkPostings) {
		c.cur = c.blkPostings[c.pos].docID
		return
	}
	if c.blkIdx+1 >= len(c.blockOffsets) {
		c.cur = sentinel
		return
	}
	if err := c.loadBlock(c.blkIdx + 1); err != nil {
		c.cur = sentinel
		return
	}
	c.cur = c.blkPostings[c.pos].docID
}

// skipTo advances to the first docID at or beyond target, skipping whole blocks
// whose last docID is below target without decoding their bodies.
func (c *cursor) skipTo(target uint32) {
	if c.cur >= target || c.cur == sentinel {
		return
	}
	i := c.blkIdx
	for i < len(c.blockLast) && c.blockLast[i] < target {
		i++
	}
	if i >= len(c.blockOffsets) {
		c.cur = sentinel
		return
	}
	if i != c.blkIdx {
		if err := c.loadBlock(i); err != nil {
			c.cur = sentinel
			return
		}
	}
	for c.pos < len(c.blkPostings) && c.blkPostings[c.pos].docID < target {
		c.pos++
	}
	if c.pos >= len(c.blkPostings) {
		c.cur = sentinel
		return
	}
	c.cur = c.blkPostings[c.pos].docID
}

// listMax is the list-wide upper bound on this term's contribution to any
// document, the bound the WAND pivot selection needs because it must hold for a
// pivot document the cursor has not yet reached.
func (c *cursor) listMax() int32 { return c.scaledMax }

// blockIndexCovering returns the index of the first block whose last docID is at
// or beyond target, scanning forward from the cursor's current block. A target
// past every block returns len(blockLast).
func (c *cursor) blockIndexCovering(target uint32) int {
	i := c.blkIdx
	for i < len(c.blockLast) && c.blockLast[i] < target {
		i++
	}
	return i
}

// blockMaxCovering is the upper bound on this term's contribution to a document
// at target, using the block that would hold target. If the term has no block at
// or beyond target it cannot contribute, so the bound is zero.
func (c *cursor) blockMaxCovering(target uint32) int32 {
	i := c.blockIndexCovering(target)
	if i >= len(c.blockMax) {
		return 0
	}
	return c.blockMax[i]
}

// blockEndCovering is the last docID of the block that would hold target, the
// right edge of the region over which blockMaxCovering stays constant.
func (c *cursor) blockEndCovering(target uint32) uint32 {
	i := c.blockIndexCovering(target)
	if i >= len(c.blockLast) {
		return sentinel
	}
	return c.blockLast[i]
}

// termScore is the integer-domain contribution of one term to one document.
func (r *Region) termScore(termIDF float64, tf *[numFields]uint32, docID uint32) int32 {
	fl := r.fieldLen(docID)
	return quantize(contribution(termIDF, tf, &fl, &r.st, &r.params))
}

// scoreDoc sums the exact BM25F contributions of every term present at docID.
func (r *Region) scoreDoc(cursors []*cursor, docID uint32) int32 {
	var sum int32
	for _, c := range cursors {
		if c.cur == docID {
			sum += r.termScore(c.info.idf, &c.blkPostings[c.pos].fieldTF, docID)
		}
	}
	return sum
}

// blockMaxWAND runs the BlockMax-WAND traversal. It keeps a cursor per term
// sorted by docID and works in two tiers. The WAND tier picks a pivot document
// using list-wide upper bounds: every document before the pivot provably cannot
// clear the threshold, so the laggard cursors skip straight to it. The block-max
// tier then refines the pivot with per-block bounds positioned to cover the pivot
// document, which prunes the pivot itself when even its tighter bound falls short.
// It returns the top-k candidates strongest first and is the pruned path the
// oracle test checks against the exhaustive scan.
//
// Every comparison against the threshold is >= rather than >, so a document whose
// upper bound merely equals the threshold is still evaluated: it can tie on score
// and win the docID tiebreak, and the result must match the oracle on ties too.
func (r *Region) blockMaxWAND(cursors []*cursor, k int) []Candidate {
	tk := newTopK(k)
	n := len(cursors)
	for {
		sort.Slice(cursors, func(i, j int) bool { return cursors[i].cur < cursors[j].cur })
		if cursors[0].cur == sentinel {
			return tk.results()
		}

		// WAND tier: the pivot is the first cursor whose cumulative list-wide max
		// reaches the threshold. No document below cursors[pivot].cur can contain
		// any term past the pivot, so its score is bounded by the prefix sum,
		// which is below the threshold. List-wide maxes are used here because the
		// pivot document lies ahead of the laggard cursors' current blocks.
		var sum int32
		pivot := -1
		for i := 0; i < n; i++ {
			if cursors[i].cur == sentinel {
				break
			}
			sum += cursors[i].listMax()
			if sum >= tk.threshold {
				pivot = i
				break
			}
		}
		if pivot < 0 {
			return tk.results()
		}
		pivotDoc := cursors[pivot].cur

		// The cursors that can contain pivotDoc are exactly the prefix whose
		// current docID is at or before it. blockSum is the tighter block-max
		// upper bound on pivotDoc's score over that prefix.
		var prefix int
		var blockSum int32
		for prefix < n && cursors[prefix].cur != sentinel && cursors[prefix].cur <= pivotDoc {
			blockSum += cursors[prefix].blockMaxCovering(pivotDoc)
			prefix++
		}

		if blockSum < tk.threshold {
			// pivotDoc cannot clear the threshold even by its block-max bound.
			// Skip to the next document worth looking at: the end of the
			// constant-block region, but never past a cursor that starts after
			// pivotDoc and could join a later document's score.
			target := pivotDoc + 1
			end := sentinel
			for i := 0; i < prefix; i++ {
				if e := cursors[i].blockEndCovering(pivotDoc); e < end {
					end = e
				}
			}
			if end != sentinel && end+1 > target {
				target = end + 1
			}
			if prefix < n && cursors[prefix].cur != sentinel && cursors[prefix].cur < target {
				target = cursors[prefix].cur
			}
			cursors[0].skipTo(target)
			continue
		}

		if cursors[0].cur == pivotDoc {
			score := r.scoreDoc(cursors, pivotDoc)
			tk.offer(Candidate{DocID: pivotDoc, Score: score})
			for _, c := range cursors {
				if c.cur == pivotDoc {
					c.advance()
				}
			}
		} else {
			// Bring the laggard cursor up to the pivot so the next iteration can
			// evaluate it.
			cursors[0].skipTo(pivotDoc)
		}
	}
}

// exhaustive scores every document that contains any query term and keeps the
// top-k, with no pruning. It is the correctness oracle: the pruned traversal is
// right exactly when it returns this. It decodes every block of every list.
func (r *Region) exhaustive(infos []termInfo, k int) ([]Candidate, error) {
	acc := map[uint32]int32{}
	for _, info := range infos {
		err := r.eachPosting(info.entry, func(p posting) {
			acc[p.docID] += r.termScore(info.idf, &p.fieldTF, p.docID)
		})
		if err != nil {
			return nil, err
		}
	}
	tk := newTopK(k)
	for docID, score := range acc {
		tk.offer(Candidate{DocID: docID, Score: score})
	}
	return tk.results(), nil
}

// eachPosting decodes every posting of a term's list and calls fn for each.
func (r *Region) eachPosting(e termEntry, fn func(posting)) error {
	list := r.postings[e.postingsOff : e.postingsOff+e.postingsLen]
	off := 0
	var prevLast uint32
	for i := uint32(0); i < e.blockCount; i++ {
		h, err := readBlockHeader(list, off)
		if err != nil {
			return err
		}
		ps, err := decodeBlock(h, prevLast, r.codec)
		if err != nil {
			return err
		}
		for _, p := range ps {
			fn(p)
		}
		prevLast = h.lastDocID
		off = h.nextOffset
	}
	return nil
}
