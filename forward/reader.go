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

// colAt slices a value from a column data block by docID, reading the two
// bracketing offsets from the block's offset index. A CodecNone value aliases the
// region; a compressed value is one zstd frame decoded into a fresh slice. An
// empty value (start == end) reads back empty without a decode, so an unset cell
// costs nothing. A frame that fails to decode, which the container CRC over the
// region should already have caught, reads back empty rather than panicking.
func (r *Region) colAt(ci int, docID uint32) []byte {
	c := r.cols[ci]
	block := r.b[c.dataOff : c.dataOff+c.dataLen]
	start := codec.Uint32(block[int(docID)*4:])
	end := codec.Uint32(block[int(docID+1)*4:])
	base := (int(r.rows) + 1) * 4
	frame := block[base+int(start) : base+int(end)]
	if c.Codec == CodecNone || len(frame) == 0 {
		return frame
	}
	out, err := r.dec.DecodeAll(frame, nil)
	if err != nil {
		return nil
	}
	return out
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
