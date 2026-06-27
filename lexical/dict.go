package lexical

import (
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// dictBlockSize is the number of terms per front-coding block. A block starts
// with one anchor term in full, then the rest as shared-prefix-plus-suffix. The
// block-offset index holds each anchor in full so a lookup binary-searches the
// anchors and then decodes one block forward.
const dictBlockSize = 16

// termEntry is what the dictionary stores for each term: where its posting list
// is, the statistics the scorer and traversal need, and where its block-max
// array is.
type termEntry struct {
	termID      uint32
	postingsOff uint64
	postingsLen uint64
	docFreq     uint32
	blockCount  uint32
	blockMaxOff uint64
	maxContrib  int32 // list-wide max contribution, the WAND upper bound
}

func appendEntry(b []byte, e termEntry) []byte {
	b = codec.AppendUvarint(b, uint64(e.termID))
	b = codec.AppendUvarint(b, e.postingsOff)
	b = codec.AppendUvarint(b, e.postingsLen)
	b = codec.AppendUvarint(b, uint64(e.docFreq))
	b = codec.AppendUvarint(b, uint64(e.blockCount))
	b = codec.AppendUvarint(b, e.blockMaxOff)
	b = codec.AppendUvarint(b, uint64(e.maxContrib))
	return b
}

func readEntry(b []byte, off int) (termEntry, int, error) {
	var e termEntry
	read := func() (uint64, bool) {
		v, n := codec.Uvarint(b[off:])
		if n <= 0 {
			return 0, false
		}
		off += n
		return v, true
	}
	var ok bool
	var v uint64
	if v, ok = read(); !ok {
		return e, off, errCorrupt
	}
	e.termID = uint32(v)
	if e.postingsOff, ok = read(); !ok {
		return e, off, errCorrupt
	}
	if e.postingsLen, ok = read(); !ok {
		return e, off, errCorrupt
	}
	if v, ok = read(); !ok {
		return e, off, errCorrupt
	}
	e.docFreq = uint32(v)
	if v, ok = read(); !ok {
		return e, off, errCorrupt
	}
	e.blockCount = uint32(v)
	if e.blockMaxOff, ok = read(); !ok {
		return e, off, errCorrupt
	}
	if v, ok = read(); !ok {
		return e, off, errCorrupt
	}
	e.maxContrib = int32(v)
	return e, off, nil
}

// encodeDict front-codes a sorted term list with parallel entries into the
// dictionary bytes: a block-offset index followed by the front-coded blocks.
// terms must be sorted ascending and entries[i] is term[i]'s entry.
func encodeDict(terms []string, entries []termEntry) []byte {
	type anchor struct {
		off    uint64
		termID uint32
		bytes  string
	}
	var blocks []byte
	var anchors []anchor

	for i := 0; i < len(terms); i += dictBlockSize {
		end := i + dictBlockSize
		if end > len(terms) {
			end = len(terms)
		}
		anchors = append(anchors, anchor{off: uint64(len(blocks)), termID: entries[i].termID, bytes: terms[i]})
		// anchor term in full
		blocks = codec.AppendUvarint(blocks, uint64(len(terms[i])))
		blocks = append(blocks, terms[i]...)
		blocks = appendEntry(blocks, entries[i])
		prev := terms[i]
		for j := i + 1; j < end; j++ {
			shared := commonPrefix(prev, terms[j])
			suffix := terms[j][shared:]
			blocks = codec.AppendUvarint(blocks, uint64(shared))
			blocks = codec.AppendUvarint(blocks, uint64(len(suffix)))
			blocks = append(blocks, suffix...)
			blocks = appendEntry(blocks, entries[j])
			prev = terms[j]
		}
	}

	var index []byte
	index = codec.AppendUvarint(index, uint64(len(anchors)))
	for _, a := range anchors {
		index = codec.AppendUvarint(index, a.off)
		index = codec.AppendUvarint(index, uint64(a.termID))
		index = codec.AppendUvarint(index, uint64(len(a.bytes)))
		index = append(index, a.bytes...)
	}

	out := codec.AppendUvarint(nil, uint64(len(index)))
	out = append(out, index...)
	out = append(out, blocks...)
	return out
}

func commonPrefix(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// dict is the parsed dictionary: the anchor index for binary search and the
// front-coded blocks for the forward decode.
type dict struct {
	anchorOffsets []uint64
	anchorTermIDs []uint32
	anchorTerms   []string
	blocks        []byte
}

func decodeDict(b []byte) (*dict, error) {
	indexLen, n := codec.Uvarint(b)
	if n <= 0 || uint64(len(b)-n) < indexLen {
		return nil, errCorrupt
	}
	idx := b[n : n+int(indexLen)]
	d := &dict{blocks: b[n+int(indexLen):]}

	count, m := codec.Uvarint(idx)
	if m <= 0 {
		return nil, errCorrupt
	}
	idx = idx[m:]
	d.anchorOffsets = make([]uint64, count)
	d.anchorTermIDs = make([]uint32, count)
	d.anchorTerms = make([]string, count)
	for i := uint64(0); i < count; i++ {
		off, k := codec.Uvarint(idx)
		if k <= 0 {
			return nil, errCorrupt
		}
		idx = idx[k:]
		tid, k := codec.Uvarint(idx)
		if k <= 0 {
			return nil, errCorrupt
		}
		idx = idx[k:]
		tl, k := codec.Uvarint(idx)
		if k <= 0 || uint64(len(idx)-k) < tl {
			return nil, errCorrupt
		}
		idx = idx[k:]
		d.anchorOffsets[i] = off
		d.anchorTermIDs[i] = uint32(tid)
		d.anchorTerms[i] = string(idx[:tl])
		idx = idx[tl:]
	}
	return d, nil
}

// lookup finds a term's entry, or false if the term is absent. It binary
// searches the anchors for the block whose range covers the term, then decodes
// that block forward comparing as it goes.
func (d *dict) lookup(term string) (termEntry, bool) {
	// largest anchor <= term
	bi := sort.Search(len(d.anchorTerms), func(i int) bool {
		return d.anchorTerms[i] > term
	}) - 1
	if bi < 0 {
		return termEntry{}, false
	}
	return d.scanBlock(bi, term)
}

// scanBlock decodes block bi forward looking for term.
func (d *dict) scanBlock(bi int, term string) (termEntry, bool) {
	off := int(d.anchorOffsets[bi])
	end := len(d.blocks)
	if bi+1 < len(d.anchorOffsets) {
		end = int(d.anchorOffsets[bi+1])
	}
	// anchor
	al, n := codec.Uvarint(d.blocks[off:])
	if n <= 0 {
		return termEntry{}, false
	}
	off += n
	cur := string(d.blocks[off : off+int(al)])
	off += int(al)
	e, off2, err := readEntry(d.blocks, off)
	if err != nil {
		return termEntry{}, false
	}
	off = off2
	if cur == term {
		return e, true
	}
	if cur > term {
		return termEntry{}, false
	}
	prev := cur
	for off < end {
		shared, n := codec.Uvarint(d.blocks[off:])
		if n <= 0 {
			return termEntry{}, false
		}
		off += n
		sl, n := codec.Uvarint(d.blocks[off:])
		if n <= 0 {
			return termEntry{}, false
		}
		off += n
		suffix := d.blocks[off : off+int(sl)]
		off += int(sl)
		cur = prev[:shared] + string(suffix)
		e, off2, err := readEntry(d.blocks, off)
		if err != nil {
			return termEntry{}, false
		}
		off = off2
		if cur == term {
			return e, true
		}
		if cur > term {
			return termEntry{}, false
		}
		prev = cur
	}
	return termEntry{}, false
}

// term reconstructs the term string for a termID, the reverse lookup tooling and
// query explanation use. It binary searches the anchor termIDs for the block,
// then decodes forward to the target position.
func (d *dict) term(termID uint32) (string, bool) {
	bi := sort.Search(len(d.anchorTermIDs), func(i int) bool {
		return d.anchorTermIDs[i] > termID
	}) - 1
	if bi < 0 {
		return "", false
	}
	off := int(d.anchorOffsets[bi])
	end := len(d.blocks)
	if bi+1 < len(d.anchorOffsets) {
		end = int(d.anchorOffsets[bi+1])
	}
	al, n := codec.Uvarint(d.blocks[off:])
	if n <= 0 {
		return "", false
	}
	off += n
	cur := string(d.blocks[off : off+int(al)])
	off += int(al)
	e, off2, err := readEntry(d.blocks, off)
	if err != nil {
		return "", false
	}
	off = off2
	if e.termID == termID {
		return cur, true
	}
	prev := cur
	for off < end {
		shared, n := codec.Uvarint(d.blocks[off:])
		if n <= 0 {
			return "", false
		}
		off += n
		sl, n := codec.Uvarint(d.blocks[off:])
		if n <= 0 {
			return "", false
		}
		off += n
		suffix := d.blocks[off : off+int(sl)]
		off += int(sl)
		cur = prev[:shared] + string(suffix)
		e, off2, err := readEntry(d.blocks, off)
		if err != nil {
			return "", false
		}
		off = off2
		if e.termID == termID {
			return cur, true
		}
		prev = cur
	}
	return "", false
}

// forEach calls fn for every term in the dictionary in sorted order, with its document
// frequency. It walks each front-coding block from its anchor, reconstructing terms
// from the shared-prefix-plus-suffix coding, the same decode the block scan does but
// over the whole dictionary rather than to a single target. The spell-corrector
// dictionary build and the reverse-lookup tooling use it.
func (d *dict) forEach(fn func(term string, docFreq uint32)) {
	for bi := 0; bi < len(d.anchorOffsets); bi++ {
		off := int(d.anchorOffsets[bi])
		end := len(d.blocks)
		if bi+1 < len(d.anchorOffsets) {
			end = int(d.anchorOffsets[bi+1])
		}
		al, n := codec.Uvarint(d.blocks[off:])
		if n <= 0 {
			return
		}
		off += n
		cur := string(d.blocks[off : off+int(al)])
		off += int(al)
		e, off2, err := readEntry(d.blocks, off)
		if err != nil {
			return
		}
		off = off2
		fn(cur, e.docFreq)
		prev := cur
		for off < end {
			shared, n := codec.Uvarint(d.blocks[off:])
			if n <= 0 {
				return
			}
			off += n
			sl, n := codec.Uvarint(d.blocks[off:])
			if n <= 0 {
				return
			}
			off += n
			suffix := d.blocks[off : off+int(sl)]
			off += int(sl)
			cur = prev[:shared] + string(suffix)
			e, off2, err := readEntry(d.blocks, off)
			if err != nil {
				return
			}
			off = off2
			fn(cur, e.docFreq)
			prev = cur
		}
	}
}
