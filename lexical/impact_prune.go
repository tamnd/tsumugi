package lexical

import "context"

// This is the pruned traversal the impact ordering exists for. The exhaustive impact
// scorer decodes every posting of every list; this one decodes them highest impact first
// and stops as soon as the top-k is settled, so on a query whose common terms fill the
// top-k at high impact it never touches the long low-impact tail.
//
// The scoring model is coverage weighted by static rank: a document's score is the number
// of query terms it carries times its impact, and the impact is a per-document constant
// (the quantized composite static rank), the same value in every list the document
// appears in. That constant is what makes the traversal both simple and exact. Because a
// document has one impact, all of its postings share the merge key (impact, docID), so a
// k-way merge over the lists by descending impact then ascending docID emits every posting
// of a document consecutively: the traversal groups them into one document without a
// per-document accumulator map, and a group is complete the moment the merge frontier drops
// below its impact. The frontier is monotone non-increasing, so once the top-k is full and
// the best any remaining document could reach, the number of still-active lists times the
// frontier impact, falls below the k-th score, no later document can enter and the walk
// stops. The bound is compared strictly, so a document that could only tie the threshold
// is still visited and can win the docID tie-break, which keeps the result identical to the
// exhaustive scan down to ties.

// impactCursor walks one term's impact-ordered list highest impact first. It decodes block
// bodies lazily and in order, threading each block's last docID into the next so the signed
// deltas resolve, and holds only the current block resident. It never seeks backward, so it
// needs neither a block-offset index nor a per-block skip array, only the offset of the next
// block and the previous block's last docID.
type impactCursor struct {
	region     *Region
	list       []byte
	off        int    // offset of the next block to load
	blocksLeft uint32 // blocks not yet loaded
	prevLast   uint32 // last docID of the block just loaded, the next block's delta base

	blk []impactPosting
	pos int

	curDoc    uint32
	curImpact uint8
	done      bool
}

// openImpactCursor builds a cursor over a term's impact list and loads its first block, the
// highest-impact block, so curDoc and curImpact are ready. An empty list opens done.
func (r *Region) openImpactCursor(e termEntry) (*impactCursor, error) {
	c := &impactCursor{
		region:     r,
		list:       r.postings[e.postingsOff : e.postingsOff+e.postingsLen],
		blocksLeft: e.blockCount,
	}
	if e.blockCount == 0 {
		c.done = true
		return c, nil
	}
	if err := c.loadNext(); err != nil {
		return nil, err
	}
	return c, nil
}

// loadNext decodes the next block and positions the cursor at its leading posting.
func (c *impactCursor) loadNext() error {
	h, err := readBlockHeader(c.list, c.off)
	if err != nil {
		return err
	}
	ps, err := decodeImpactBlock(h, c.prevLast)
	if err != nil {
		return err
	}
	if len(ps) == 0 {
		return errCorrupt
	}
	c.blk = ps
	c.pos = 0
	c.off = h.nextOffset
	c.prevLast = ps[len(ps)-1].docID
	c.blocksLeft--
	c.curDoc = ps[0].docID
	c.curImpact = ps[0].impact
	return nil
}

// advance moves to the next posting, crossing into the next block if the current one is
// spent and marking the cursor done when the list is exhausted.
func (c *impactCursor) advance() error {
	c.pos++
	if c.pos < len(c.blk) {
		c.curDoc = c.blk[c.pos].docID
		c.curImpact = c.blk[c.pos].impact
		return nil
	}
	if c.blocksLeft == 0 {
		c.done = true
		return nil
	}
	return c.loadNext()
}

// prunedImpact is the impact-ordered top-k traversal, the serving path with no deadline. It
// returns the same top-k the exhaustive scan does, discarding the postings-examined count
// and the completed flag prunedImpactCore also reports; without a deadline the walk always
// completes.
func (r *Region) prunedImpact(infos []termInfo, k int) ([]Candidate, error) {
	cands, _, _, err := r.prunedImpactCore(context.Background(), infos, k)
	return cands, err
}

// prunedImpactStats runs the traversal with no deadline and returns how many postings it
// examined, the number the skip test asserts is below the list length and the impl note
// quotes for the skip win.
func (r *Region) prunedImpactStats(infos []termInfo, k int) ([]Candidate, int, error) {
	cands, examined, _, err := r.prunedImpactCore(context.Background(), infos, k)
	return cands, examined, err
}

// prunedImpactCore is the traversal. It merges the query-term cursors by descending impact
// then ascending docID, groups the postings that share an (impact, docID) into one document,
// and stops once the top-k is full and no remaining document can reach the k-th score. It
// polls ctx on the same stride BlockMax-WAND uses and returns completed=false with the
// partial it gathered if the deadline passes mid-walk, so a preempted shard drops its result
// rather than serve a half-walked list; it also returns the number of postings examined.
func (r *Region) prunedImpactCore(ctx context.Context, infos []termInfo, k int) ([]Candidate, int, bool, error) {
	if !r.impact {
		return nil, 0, false, errNotImpactRegion
	}
	cursors := make([]*impactCursor, 0, len(infos))
	for _, info := range infos {
		c, err := r.openImpactCursor(info.entry)
		if err != nil {
			return nil, 0, false, err
		}
		if !c.done {
			cursors = append(cursors, c)
		}
	}

	examined := 0
	tk := newTopK(k)
	var haveGroup bool
	var gImpact uint8
	var gDoc uint32
	var gCov int32
	finalize := func() {
		if haveGroup {
			tk.offer(Candidate{DocID: gDoc, Score: gCov * int32(gImpact)})
			haveGroup = false
		}
	}

	for len(cursors) > 0 {
		if (examined&wandPreemptStride) == 0 && ctx.Err() != nil {
			return tk.results(), examined, false, nil
		}
		// The next posting to process is the one with the highest impact, breaking a tie
		// toward the smaller docID so a document's postings and the docID order the top-k
		// prefers both fall out of the merge directly.
		best := 0
		for i := 1; i < len(cursors); i++ {
			if cursors[i].curImpact > cursors[best].curImpact ||
				(cursors[i].curImpact == cursors[best].curImpact && cursors[i].curDoc < cursors[best].curDoc) {
				best = i
			}
		}
		c := cursors[best]

		// Early termination: every remaining posting has impact at most c.curImpact, and a
		// future document can draw coverage only from the still-active lists, so the best any
		// document after the current group could score is len(cursors) times c.curImpact. If
		// the top-k is full and that bound is below the k-th score, nothing further can enter.
		// The current group is exempt: a posting that extends it is not a future document, so
		// the bound is only tested when this posting opens a new document.
		newDoc := !haveGroup || gImpact != c.curImpact || gDoc != c.curDoc
		if tk.full && newDoc {
			if int32(len(cursors))*int32(c.curImpact) < tk.threshold {
				finalize()
				return tk.results(), examined, true, nil
			}
		}

		examined++
		if newDoc {
			finalize()
			haveGroup = true
			gImpact = c.curImpact
			gDoc = c.curDoc
			gCov = 1
		} else {
			gCov++
		}

		if err := c.advance(); err != nil {
			return nil, 0, false, err
		}
		if c.done {
			cursors = append(cursors[:best], cursors[best+1:]...)
		}
	}
	finalize()
	return tk.results(), examined, true, nil
}
