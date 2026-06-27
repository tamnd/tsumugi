package lexical

import "github.com/tamnd/tsumugi/codec"

// blockSize is the number of postings per block. The codec, the block-max
// metadata, and the skip structure all operate at this granularity. 128 is the
// value the block codec is tuned for.
const blockSize = 128

// posting is one entry in a docID-ordered list: a document and the term's
// frequency in each of its fields.
type posting struct {
	docID   uint32
	fieldTF [numFields]uint32
}

// encodeBlock writes one docID-ordered block. The block header is the highest
// docID in the block (the skip pointer) and the byte lengths of the two streams,
// so a reader can step to the next block from the header alone without decoding
// either stream. DocIDs are delta-coded against prevLast, the last docID of the
// previous block, so the first gap of a block is small too. The payload of a
// posting is a field mask plus one frequency varint per set field.
func encodeBlock(out []byte, ps []posting, prevLast uint32) []byte {
	last := ps[len(ps)-1].docID

	var docs []byte
	prev := prevLast
	for _, p := range ps {
		docs = codec.AppendUvarint(docs, uint64(p.docID-prev))
		prev = p.docID
	}

	var pay []byte
	for i := range ps {
		var mask uint8
		for f := 0; f < numFields; f++ {
			if ps[i].fieldTF[f] > 0 {
				mask |= 1 << uint8(f)
			}
		}
		pay = append(pay, mask)
		for f := 0; f < numFields; f++ {
			if ps[i].fieldTF[f] > 0 {
				pay = codec.AppendUvarint(pay, uint64(ps[i].fieldTF[f]))
			}
		}
	}

	out = codec.AppendUvarint(out, uint64(last))
	out = codec.AppendUvarint(out, uint64(len(docs)))
	out = codec.AppendUvarint(out, uint64(len(pay)))
	out = append(out, docs...)
	out = append(out, pay...)
	return out
}

// blockHeader is the parsed three-varint head of a block plus the slices of its
// two streams and the offset where the next block begins.
type blockHeader struct {
	lastDocID  uint32
	docs       []byte
	payload    []byte
	nextOffset int
}

// readBlockHeader parses the block starting at b[off:], returning the streams and
// the offset of the following block.
func readBlockHeader(b []byte, off int) (blockHeader, error) {
	var h blockHeader
	last, n := codec.Uvarint(b[off:])
	if n <= 0 {
		return h, errCorrupt
	}
	off += n
	dl, n := codec.Uvarint(b[off:])
	if n <= 0 {
		return h, errCorrupt
	}
	off += n
	pl, n := codec.Uvarint(b[off:])
	if n <= 0 {
		return h, errCorrupt
	}
	off += n
	if off+int(dl)+int(pl) > len(b) {
		return h, errCorrupt
	}
	h.lastDocID = uint32(last)
	h.docs = b[off : off+int(dl)]
	h.payload = b[off+int(dl) : off+int(dl)+int(pl)]
	h.nextOffset = off + int(dl) + int(pl)
	return h, nil
}

// decodeBlock decodes a block into postings, given the last docID of the
// previous block so the first gap resolves to an absolute docID.
func decodeBlock(h blockHeader, prevLast uint32) ([]posting, error) {
	// First pass over the docID stream recovers the docIDs and the count.
	var docIDs []uint32
	prev := prevLast
	for off := 0; off < len(h.docs); {
		gap, n := codec.Uvarint(h.docs[off:])
		if n <= 0 {
			return nil, errCorrupt
		}
		off += n
		prev += uint32(gap)
		docIDs = append(docIDs, prev)
	}
	ps := make([]posting, len(docIDs))
	off := 0
	for i := range docIDs {
		ps[i].docID = docIDs[i]
		if off >= len(h.payload) {
			return nil, errCorrupt
		}
		mask := h.payload[off]
		off++
		for f := 0; f < numFields; f++ {
			if mask&(1<<uint8(f)) != 0 {
				tf, n := codec.Uvarint(h.payload[off:])
				if n <= 0 {
					return nil, errCorrupt
				}
				off += n
				ps[i].fieldTF[f] = uint32(tf)
			}
		}
	}
	return ps, nil
}
