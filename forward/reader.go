package forward

import (
	"errors"

	"github.com/tamnd/tsumugi/codec"
)

// ErrCorrupt is returned when the region bytes do not parse as a valid FWD1
// region or fail their header CRC.
var ErrCorrupt = errors.New("forward: corrupt region")

// Region is a parsed, read-only forward store. It holds sub-slices of the region
// bytes and copies nothing: a value returned by Column aliases the region, so a
// caller must not mutate it.
type Region struct {
	b      []byte
	cols   []colDesc
	colIdx map[string]int
	rows   uint32
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

	// Every column's data block must fall inside the region.
	for _, c := range cols {
		if c.dataOff+c.dataLen > uint64(len(b)) {
			return nil, ErrCorrupt
		}
		// The offset index needs at least (rows+1) uint32s.
		if c.dataLen < uint64(rows+1)*4 {
			return nil, ErrCorrupt
		}
	}

	idx := make(map[string]int, len(cols))
	for i, c := range cols {
		idx[c.Name] = i
	}
	return &Region{b: b, cols: cols, colIdx: idx, rows: rows}, nil
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
// bracketing offsets from the block's offset index.
func (r *Region) colAt(ci int, docID uint32) []byte {
	c := r.cols[ci]
	block := r.b[c.dataOff : c.dataOff+c.dataLen]
	start := codec.Uint32(block[int(docID)*4:])
	end := codec.Uint32(block[int(docID+1)*4:])
	base := (int(r.rows) + 1) * 4
	return block[base+int(start) : base+int(end)]
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
