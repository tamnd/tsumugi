package feature

import (
	"errors"
	"math"

	"github.com/tamnd/tsumugi/codec"
)

// ErrCorrupt is returned when the region bytes do not parse as a valid FEA1
// region or fail the header CRC.
var ErrCorrupt = errors.New("feature: corrupt region")

// Region is a parsed, read-only feature matrix. The row bytes alias the region,
// so a value read out is dequantized into a fresh float and the underlying bytes
// are never mutated.
type Region struct {
	rows          []byte
	stride        int
	rowCount      uint32
	schemaVersion uint16
	cols          []colLayout
	colIdx        map[FeatureID]int
}

// Open parses a feature region from its bytes.
func Open(b []byte) (*Region, error) {
	if len(b) < 20 || string(b[0:4]) != regionMagic {
		return nil, ErrCorrupt
	}
	if codec.Uint32(b[4:]) != regionVersion {
		return nil, ErrCorrupt
	}
	colCount := int(codec.Uint16(b[8:]))
	stride := int(codec.Uint16(b[10:]))
	rowCount := codec.Uint32(b[12:])
	schemaVersion := codec.Uint16(b[16:])

	headerLen := 20 + colCount*8 + colCount*14 + 4
	if len(b) < headerLen {
		return nil, ErrCorrupt
	}
	if codec.Uint32(b[headerLen-4:]) != codec.CRC32C(b[:headerLen-4]) {
		return nil, ErrCorrupt
	}

	cols := make([]colLayout, colCount)
	off := 20
	for i := 0; i < colCount; i++ {
		cols[i].ID = FeatureID(b[off])
		cols[i].Width = b[off+1]
		cols[i].Quant = Quant(b[off+2])
		cols[i].offset = codec.Uint16(b[off+4:])
		off += 8
	}
	for i := 0; i < colCount; i++ {
		// The dequant block repeats id and quant for self-description; the layout
		// already has them, so only the params are taken here.
		cols[i].p0 = math.Float32frombits(codec.Uint32(b[off+2:]))
		cols[i].p1 = math.Float32frombits(codec.Uint32(b[off+6:]))
		cols[i].p2 = math.Float32frombits(codec.Uint32(b[off+10:]))
		off += 14
	}

	rows := b[headerLen:]
	if len(rows) < int(rowCount)*stride {
		return nil, ErrCorrupt
	}

	idx := make(map[FeatureID]int, colCount)
	for i, c := range cols {
		idx[c.ID] = i
	}
	return &Region{
		rows:          rows,
		stride:        stride,
		rowCount:      rowCount,
		schemaVersion: schemaVersion,
		cols:          cols,
		colIdx:        idx,
	}, nil
}

// DocCount returns N, the number of rows.
func (r *Region) DocCount() uint32 { return r.rowCount }

// Stride returns the bytes per row.
func (r *Region) Stride() int { return r.stride }

// SchemaVersion returns the feature schema version the region was built against.
func (r *Region) SchemaVersion() uint16 { return r.schemaVersion }

// Columns returns the region's column layout in row order: which signal each column
// holds, its width, and its quantization. It drops the byte offset and dequant
// params, which are derived, so the result is comparable across regions and against
// DefaultSchema.
func (r *Region) Columns() []Column {
	cols := make([]Column, len(r.cols))
	for i, c := range r.cols {
		cols[i] = c.Column
	}
	return cols
}

// SchemaHash is the SchemaHash of the region's own column layout, the fingerprint a
// loader compares against DefaultSchemaHash to refuse a shard whose feature matrix
// does not match the schema this build scores against.
func (r *Region) SchemaHash() uint64 { return SchemaHash(r.Columns()) }

// Row returns the raw stride bytes for a document. It aliases the region and must
// not be mutated.
func (r *Region) Row(docID uint32) ([]byte, bool) {
	if docID >= r.rowCount {
		return nil, false
	}
	base := int(docID) * r.stride
	return r.rows[base : base+r.stride], true
}

// dequant turns a column's stored level back into a real value.
func dequant(c colLayout, q uint32) float64 {
	switch c.Quant {
	case QuantLog:
		return math.Exp(float64(c.p0)+float64(q)*float64(c.p1)) - float64(c.p2)
	case QuantSigned:
		return (float64(q) - math.Round(maxLevel(c.Width)/2)) * float64(c.p0)
	default:
		return float64(c.p0) + float64(q)*float64(c.p1)
	}
}

// Value returns the dequantized value of one feature for one document.
func (r *Region) Value(docID uint32, id FeatureID) (float64, bool) {
	ci, ok := r.colIdx[id]
	if !ok || docID >= r.rowCount {
		return 0, false
	}
	c := r.cols[ci]
	row := r.rows[int(docID)*r.stride:]
	q := loadQuant(row, c.offset, c.Width)
	return dequant(c, q), true
}

// L1Weights binds a feature id to its weight in the L1 linear scorer.
type L1Weights struct {
	IDs     []FeatureID
	Weights []float64
}

// L1Scan is the L1 linear scoring kernel: for each candidate it sums
// weight*value over a small subset of columns and adds the carried retrieval
// score. The row-major fixed-width layout makes the inner loop pure address
// arithmetic and a few multiply-adds per candidate, no indirection and no trees.
// It returns one score per candidate, aligned with cands.
func (r *Region) L1Scan(cands []uint32, w L1Weights, carried []float64) []float64 {
	// Resolve the columns once, outside the hot loop.
	type bound struct {
		c colLayout
		w float64
	}
	bs := make([]bound, 0, len(w.IDs))
	for i, id := range w.IDs {
		ci, ok := r.colIdx[id]
		if !ok {
			continue
		}
		bs = append(bs, bound{c: r.cols[ci], w: w.Weights[i]})
	}

	out := make([]float64, len(cands))
	for i, d := range cands {
		var s float64
		if i < len(carried) {
			s = carried[i]
		}
		if d < r.rowCount {
			row := r.rows[int(d)*r.stride:]
			for _, b := range bs {
				q := loadQuant(row, b.c.offset, b.c.Width)
				s += b.w * dequant(b.c, q)
			}
		}
		out[i] = s
	}
	return out
}
