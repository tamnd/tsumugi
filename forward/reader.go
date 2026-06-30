package forward

import (
	"errors"

	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi/codec"
)

// ErrCorrupt is returned when the region bytes do not parse as a valid FWD1
// region or fail their header CRC.
var ErrCorrupt = errors.New("forward: corrupt region")

// Region is a parsed, read-only forward store. A value from a CodecNone column
// aliases the region bytes and must not be mutated; a value from a compressed
// column is decoded into a fresh slice the caller owns. A Region that opened a
// compressed column holds a zstd decoder and must be closed to release it.
type Region struct {
	b      []byte
	cols   []colDesc
	colIdx map[string]int
	rows   uint32
	dec    *zstd.Decoder // nil when every column is CodecNone
}

// Open parses a forward region from its bytes.
func Open(b []byte) (*Region, error) {
	if len(b) < 16 || string(b[0:4]) != regionMagic {
		return nil, ErrCorrupt
	}
	if codec.Uint32(b[4:]) != regionVersion {
		return nil, ErrCorrupt
	}
	colCount := int(codec.Uint32(b[8:]))
	rows := codec.Uint32(b[12:])

	off := 16
	cols := make([]colDesc, 0, colCount)
	for i := 0; i < colCount; i++ {
		c, next, ok := readColDesc(b, off)
		if !ok {
			return nil, ErrCorrupt
		}
		cols = append(cols, c)
		off = next
	}
	if off+4 > len(b) {
		return nil, ErrCorrupt
	}
	if codec.Uint32(b[off:]) != codec.CRC32C(b[:off]) {
		return nil, ErrCorrupt
	}

	// Every column's data block and dictionary must fall inside the region, and a
	// compressed column registers its dictionary so frames decode against it. The
	// decoder runs single-goroutine: a shard may hold tens of thousands of these,
	// so the per-region cost stays one lightweight decoder, not a goroutine fan-out.
	opts := []zstd.DOption{zstd.WithDecoderConcurrency(1)}
	compressed := false
	for ci, c := range cols {
		if c.dataOff+c.dataLen > uint64(len(b)) {
			return nil, ErrCorrupt
		}
		// The offset index needs at least (rows+1) uint32s.
		if c.dataLen < uint64(rows+1)*4 {
			return nil, ErrCorrupt
		}
		if c.dictLen > 0 {
			if c.dictOff+c.dictLen > uint64(len(b)) {
				return nil, ErrCorrupt
			}
			opts = append(opts, zstd.WithDecoderDictRaw(dictID(ci), b[c.dictOff:c.dictOff+c.dictLen]))
		}
		if c.Codec != CodecNone {
			compressed = true
		}
	}

	idx := make(map[string]int, len(cols))
	for i, c := range cols {
		idx[c.Name] = i
	}
	r := &Region{b: b, cols: cols, colIdx: idx, rows: rows}
	if compressed {
		dec, err := zstd.NewReader(nil, opts...)
		if err != nil {
			return nil, err
		}
		r.dec = dec
	}
	return r, nil
}

// Close releases the region's zstd decoder. It is safe to call on a Region with
// no compressed columns and safe to call more than once.
func (r *Region) Close() {
	if r != nil && r.dec != nil {
		r.dec.Close()
		r.dec = nil
	}
}

// Schema returns the column descriptors in storage order.
func (r *Region) Schema() []Column {
	out := make([]Column, len(r.cols))
	for i, c := range r.cols {
		out[i] = c.Column
	}
	return out
}

// DocCount returns N, the number of dense docID rows.
func (r *Region) DocCount() uint32 { return r.rows }

// Column returns one column's value for one document. The bool is false for an
// unknown column or an out-of-range docID. An empty value (a document with no
// value set for the column) returns an empty, non-nil slice and true.
func (r *Region) Column(name string, docID uint32) ([]byte, bool) {
	ci, ok := r.colIdx[name]
	if !ok || docID >= r.rows {
		return nil, false
	}
	return r.colAt(ci, docID), true
}

// ColumnInto is Column with a caller-supplied scratch buffer the decode reuses, so
// a caller reading one column across many documents holds a single decode buffer
// instead of allocating a fresh value per call. This is the L2 rerank path: an
// extractor reads the body of every survivor of a query, and reusing the buffer
// turns hundreds of per-survivor body allocations into one.
//
// It returns the value, the buffer to keep for the next call, and the Column bool.
// The usage is to keep that second return as the scratch for the next read:
//
//	val, scratch, ok = r.ColumnInto(col, doc, scratch)
//
// The kept buffer is always caller-owned, never a region alias: a decoded value
// decodes into the (possibly grown) scratch and the kept buffer is that scratch, so
// reuse carries the capacity forward; a raw or empty value aliases the region and
// the kept buffer is the unchanged scratch, so the caller never retains a region
// alias and then decodes into it, which would write into the read-only region. The
// returned value is valid only until the next call that reuses the same scratch.
func (r *Region) ColumnInto(name string, docID uint32, scratch []byte) (value, keep []byte, ok bool) {
	ci, ok := r.colIdx[name]
	if !ok || docID >= r.rows {
		return nil, scratch, false
	}
	value, keep = r.colAtInto(ci, docID, scratch)
	return value, keep, true
}

// ColumnPrefixInto is ColumnInto for a reader that needs only a leading window of a
// large value, the L2 body scan that reads at most a fixed rune cap. minBytes is the
// least number of decoded bytes the caller needs (the rune cap times the max bytes a
// rune takes, so the window always holds enough complete runes); the read decodes
// only the blocks that window spans rather than the whole value. It applies only to a
// CodecZstdDictBlocked column; for every other codec the window is not expressible in
// the stored layout, so it falls back to the full ColumnInto decode and minBytes is
// ignored, which keeps the caller uniform across codecs and across an old region whose
// body column predates the blocked codec. The value, kept buffer, and bool follow the
// same contract as ColumnInto: the kept buffer is always caller-owned, and the value
// is valid only until the next call that reuses the scratch.
func (r *Region) ColumnPrefixInto(name string, docID uint32, minBytes int, scratch []byte) (value, keep []byte, ok bool) {
	ci, ok := r.colIdx[name]
	if !ok || docID >= r.rows {
		return nil, scratch, false
	}
	value, keep = r.colPrefixInto(ci, docID, minBytes, scratch)
	return value, keep, true
}

// colAt slices a value from a column data block by docID. It decodes into a fresh
// slice; colAtInto is the buffer-reusing form the L2 path calls.
func (r *Region) colAt(ci int, docID uint32) []byte {
	v, _ := r.colAtInto(ci, docID, nil)
	return v
}

// colAtInto slices a value from a column data block by docID, reading the two
// bracketing offsets from the block's offset index, decoding a compressed value
// into scratch[:0]. It returns the value and the buffer the caller should keep as
// scratch for the next call. A compressed value decodes into scratch, which grows
// as needed, and the kept buffer is that grown buffer so reuse carries the capacity
// forward. A CodecNone value, and an empty value of any codec (start == end), alias
// the region without a decode; the kept buffer is then the unchanged scratch, never
// the region alias, so a caller that keeps it and decodes into it next time never
// writes into the read-only region. A frame that fails to decode, which the
// container CRC over the region should already have caught, reads back empty rather
// than panicking.
func (r *Region) colAtInto(ci int, docID uint32, scratch []byte) (value, keep []byte) {
	c := r.cols[ci]
	block := r.b[c.dataOff : c.dataOff+c.dataLen]
	start := codec.Uint32(block[int(docID)*4:])
	end := codec.Uint32(block[int(docID+1)*4:])
	base := (int(r.rows) + 1) * 4
	frame := block[base+int(start) : base+int(end)]
	if c.Codec == CodecNone || len(frame) == 0 {
		return frame, scratch
	}
	if c.Codec == CodecZstdDictBlocked {
		return r.decodeBlocked(frame, -1, scratch)
	}
	out, err := r.dec.DecodeAll(frame, scratch[:0])
	if err != nil {
		return nil, scratch
	}
	return out, out
}

// colPrefixInto decodes a leading window of a value, the windowed form of colAtInto.
// For a CodecZstdDictBlocked column it decodes only the blocks needed to reach
// minBytes decoded bytes (or every block, for a value shorter than the window),
// concatenating them into scratch exactly as the full decode would, so the returned
// prefix is byte-for-byte the start of the full value. For any other codec the window
// is not expressible, so it returns the full value through colAtInto. A minBytes below
// zero means decode the whole value.
func (r *Region) colPrefixInto(ci int, docID uint32, minBytes int, scratch []byte) (value, keep []byte) {
	c := r.cols[ci]
	if c.Codec != CodecZstdDictBlocked {
		return r.colAtInto(ci, docID, scratch)
	}
	block := r.b[c.dataOff : c.dataOff+c.dataLen]
	start := codec.Uint32(block[int(docID)*4:])
	end := codec.Uint32(block[int(docID+1)*4:])
	base := (int(r.rows) + 1) * 4
	frame := block[base+int(start) : base+int(end)]
	if len(frame) == 0 {
		return frame, scratch
	}
	return r.decodeBlocked(frame, minBytes, scratch)
}

// decodeBlocked decodes a CodecZstdDictBlocked value: a uint32 block count, one
// uint32 cumulative end offset per block, then the block frames. It decodes blocks in
// order, appending each into out so the concatenation reproduces the value, and stops
// once out holds minBytes decoded bytes; a negative minBytes decodes every block. The
// kept buffer is out, the (grown) scratch, always caller-owned. A malformed header or
// a frame that fails to decode, which the container CRC should already have caught,
// reads back empty rather than panicking, matching colAtInto's defensive decode.
func (r *Region) decodeBlocked(frame []byte, minBytes int, scratch []byte) (value, keep []byte) {
	if len(frame) < 4 {
		return nil, scratch
	}
	nblocks := int(codec.Uint32(frame))
	framesBase := 4 + nblocks*4
	if framesBase > len(frame) {
		return nil, scratch
	}
	out := scratch[:0]
	var prev uint32
	for k := 0; k < nblocks; k++ {
		end := codec.Uint32(frame[4+k*4:])
		if int(prev) > int(end) || framesBase+int(end) > len(frame) {
			return nil, scratch
		}
		bf := frame[framesBase+int(prev) : framesBase+int(end)]
		prev = end
		var err error
		out, err = r.dec.DecodeAll(bf, out)
		if err != nil {
			return nil, scratch
		}
		if minBytes >= 0 && len(out) >= minBytes {
			break
		}
	}
	return out, out
}

// Row returns every column value for one document, keyed by column name. It is
// the result-display path; a query that only needs the snippet should call
// Column to avoid materializing the body.
func (r *Region) Row(docID uint32) (map[string][]byte, bool) {
	if docID >= r.rows {
		return nil, false
	}
	out := make(map[string][]byte, len(r.cols))
	for ci, c := range r.cols {
		out[c.Name] = r.colAt(ci, docID)
	}
	return out, true
}
