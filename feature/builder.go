package feature

import (
	"math"

	"github.com/tamnd/tsumugi/codec"
)

// Builder accumulates per-document signal values and encodes them into a feature
// region. It is a two-pass build: the first pass collects raw values and the
// per-column range, the second pass derives the dequant params and writes the
// quantized rows. Holding the raw values in memory is fine because a shard is the
// unit one file holds and a build runs over one shard.
type Builder struct {
	cols          []Column
	colIdx        map[FeatureID]int
	stride        uint16
	offsets       []uint16
	schemaVersion uint16

	raw     map[uint32][]float64 // docID -> per-column raw value
	maxDoc  uint32
	hasDocs bool
}

// NewBuilder starts a builder over a column schema. Columns are packed into a row
// in declared order; the stride is the sum of the column widths.
func NewBuilder(cols []Column, schemaVersion uint16) *Builder {
	idx := make(map[FeatureID]int, len(cols))
	offsets := make([]uint16, len(cols))
	var off uint16
	for i, c := range cols {
		idx[c.ID] = i
		offsets[i] = off
		off += uint16(c.Width)
	}
	return &Builder{
		cols:          cols,
		colIdx:        idx,
		stride:        off,
		offsets:       offsets,
		schemaVersion: schemaVersion,
		raw:           map[uint32][]float64{},
	}
}

// Set records one signal value for one document. An unknown feature id is a
// no-op. Unset values default to zero, which quantizes to the column minimum.
func (b *Builder) Set(docID uint32, id FeatureID, value float64) {
	ci, ok := b.colIdx[id]
	if !ok {
		return
	}
	if !b.hasDocs || docID > b.maxDoc {
		b.maxDoc = docID
		b.hasDocs = true
	}
	row := b.raw[docID]
	if row == nil {
		row = make([]float64, len(b.cols))
		b.raw[docID] = row
	}
	row[ci] = value
}

func (b *Builder) docCount() uint32 {
	if !b.hasDocs {
		return 0
	}
	return b.maxDoc + 1
}

// quantParams derives the dequant params for one column from its observed range.
func quantParams(c Column, min, max float64) (p0, p1, p2 float32) {
	lvl := maxLevel(c.Width)
	switch c.Quant {
	case QuantLog:
		lmin := math.Log(min + epsLog)
		lmax := math.Log(max + epsLog)
		lscale := (lmax - lmin) / lvl
		if lscale == 0 {
			lscale = 1
		}
		return float32(lmin), float32(lscale), float32(epsLog)
	case QuantSigned:
		// Symmetric around zero: the scale spans the larger magnitude end.
		r := math.Max(math.Abs(min), math.Abs(max))
		scale := 2 * r / lvl
		if scale == 0 {
			scale = 1
		}
		return float32(scale), 0, 0
	default: // QuantLinear
		scale := (max - min) / lvl
		if scale == 0 {
			scale = 1
		}
		return float32(min), float32(scale), 0
	}
}

// quantize maps a raw value to its stored level using a column's params.
func quantize(c Column, p0, p1, p2 float32, v float64) uint32 {
	lvl := maxLevel(c.Width)
	var q float64
	switch c.Quant {
	case QuantLog:
		q = math.Round((math.Log(v+float64(p2)) - float64(p0)) / float64(p1))
	case QuantSigned:
		q = math.Round(v/float64(p0)) + math.Round(lvl/2)
	default:
		q = math.Round((v - float64(p0)) / float64(p1))
	}
	if q < 0 {
		q = 0
	}
	if q > lvl {
		q = lvl
	}
	return uint32(q)
}

// Build encodes the accumulated rows into a feature region byte run.
func (b *Builder) Build() []byte {
	n := b.docCount()

	// First pass: per-column min and max over the dense space, zeros included.
	mins := make([]float64, len(b.cols))
	maxs := make([]float64, len(b.cols))
	for ci := range b.cols {
		mins[ci] = math.Inf(1)
		maxs[ci] = math.Inf(-1)
	}
	for d := uint32(0); d < n; d++ {
		row := b.raw[d]
		for ci := range b.cols {
			v := 0.0
			if row != nil {
				v = row[ci]
			}
			if v < mins[ci] {
				mins[ci] = v
			}
			if v > maxs[ci] {
				maxs[ci] = v
			}
		}
	}

	// Derive dequant params per column.
	lay := make([]colLayout, len(b.cols))
	for ci, c := range b.cols {
		mn, mx := mins[ci], maxs[ci]
		if math.IsInf(mn, 1) { // no documents
			mn, mx = 0, 0
		}
		p0, p1, p2 := quantParams(c, mn, mx)
		lay[ci] = colLayout{Column: c, offset: b.offsets[ci], p0: p0, p1: p1, p2: p2}
	}

	// Second pass: quantize each row into fixed-width bytes.
	rows := make([]byte, int(n)*int(b.stride))
	for d := uint32(0); d < n; d++ {
		row := b.raw[d]
		base := int(d) * int(b.stride)
		for ci, c := range b.cols {
			v := 0.0
			if row != nil {
				v = row[ci]
			}
			q := quantize(c, lay[ci].p0, lay[ci].p1, lay[ci].p2, v)
			off := base + int(lay[ci].offset)
			if c.Width == 2 {
				codec.PutUint16(rows[off:], uint16(q))
			} else {
				rows[off] = byte(q)
			}
		}
	}

	// Header: fixed prefix, the column table, the dequant block, header CRC.
	head := make([]byte, 0, 64+len(b.cols)*22)
	head = append(head, regionMagic...)
	head = codec.AppendUint32(head, regionVersion)
	head = codec.AppendUint16(head, uint16(len(b.cols)))
	head = codec.AppendUint16(head, b.stride)
	head = codec.AppendUint32(head, n)
	head = codec.AppendUint16(head, b.schemaVersion)
	head = codec.AppendUint16(head, 0) // flags
	for ci := range lay {
		head = append(head, byte(lay[ci].ID), lay[ci].Width, byte(lay[ci].Quant), 0)
		head = codec.AppendUint16(head, lay[ci].offset)
		head = codec.AppendUint16(head, 0) // reserved
	}
	for ci := range lay {
		head = append(head, byte(lay[ci].ID), byte(lay[ci].Quant))
		head = codec.AppendUint32(head, math.Float32bits(lay[ci].p0))
		head = codec.AppendUint32(head, math.Float32bits(lay[ci].p1))
		head = codec.AppendUint32(head, math.Float32bits(lay[ci].p2))
	}
	head = codec.AppendUint32(head, codec.CRC32C(head))

	out := make([]byte, 0, len(head)+len(rows))
	out = append(out, head...)
	out = append(out, rows...)
	return out
}
