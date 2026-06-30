package forward

import (
	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi/codec"
)

// Builder accumulates document rows into a forward region. Columns are declared
// up front; values arrive per document in any docID order and the build packs
// them into the dense docID space [0, N), filling absent values with empty bytes.
type Builder struct {
	cols    []Column
	colIdx  map[string]int
	rows    map[uint32][][]byte // docID -> per-column value, nil where unset
	maxDoc  uint32
	hasDocs bool
}

// NewBuilder starts a builder over a fixed column schema. The column order is the
// order values are stored and reported in.
func NewBuilder(cols []Column) *Builder {
	idx := make(map[string]int, len(cols))
	for i, c := range cols {
		idx[c.Name] = i
	}
	return &Builder{cols: cols, colIdx: idx, rows: map[uint32][][]byte{}}
}

// Set records one column value for one document. An unknown column name is a
// no-op, which keeps a caller that derives columns dynamically from panicking on
// a column it forgot to declare. The value bytes are retained, not copied, so a
// caller must not mutate them after the call.
func (b *Builder) Set(docID uint32, col string, value []byte) {
	ci, ok := b.colIdx[col]
	if !ok {
		return
	}
	if !b.hasDocs || docID > b.maxDoc {
		b.maxDoc = docID
		b.hasDocs = true
	}
	row := b.rows[docID]
	if row == nil {
		row = make([][]byte, len(b.cols))
		b.rows[docID] = row
	}
	row[ci] = value
}

// docCount returns N, the dense docID space size.
func (b *Builder) docCount() uint32 {
	if !b.hasDocs {
		return 0
	}
	return b.maxDoc + 1
}

// Build encodes the accumulated rows into a forward region byte run. Each column
// becomes an offset index over [0, N] followed by the packed value bytes, so a
// reader reaches any value with two offset reads and a slice. A column declared
// with a zstd codec packs each value as an independent frame instead of raw bytes,
// and a CodecZstdDict column derives a shared dictionary from its values and
// appends it to the region for the reader to register.
func (b *Builder) Build() []byte {
	n := b.docCount()

	// Pack each column: its data block (offset index plus packed values or frames)
	// and, for a dictionary column, the dictionary bytes its frames reference.
	blocks := make([][]byte, len(b.cols))
	dicts := make([][]byte, len(b.cols))
	for ci := range b.cols {
		switch b.cols[ci].Codec {
		case CodecZstd, CodecZstdDict:
			blocks[ci], dicts[ci] = b.buildCompressed(ci, n)
		default:
			blocks[ci] = b.buildRaw(ci, n)
		}
	}

	descs := make([]colDesc, len(b.cols))
	for ci := range b.cols {
		descs[ci] = colDesc{Column: b.cols[ci]}
	}

	// First lay out the header to learn where data begins. The header length
	// depends only on the descriptors, which have fixed-size tails plus names.
	headerLen := 4 + 4 + 4 + 4 // magic, version, colCount, rowCount
	for ci := range b.cols {
		headerLen += 4 + len(b.cols[ci].Name) + descTail
	}
	headerLen += 4 // header_crc

	// Data blocks come first in column order, then the dictionary blocks, so a
	// reader that never decodes still touches only the small offset indices.
	off := uint64(headerLen)
	for ci := range blocks {
		descs[ci].dataOff = off
		descs[ci].dataLen = uint64(len(blocks[ci]))
		off += descs[ci].dataLen
	}
	for ci := range dicts {
		if len(dicts[ci]) > 0 {
			descs[ci].dictOff = off
			descs[ci].dictLen = uint64(len(dicts[ci]))
			off += descs[ci].dictLen
		}
	}

	head := make([]byte, 0, headerLen)
	head = append(head, regionMagic...)
	head = codec.AppendUint32(head, regionVersion)
	head = codec.AppendUint32(head, uint32(len(b.cols)))
	head = codec.AppendUint32(head, n)
	for ci := range descs {
		head = appendColDesc(head, descs[ci])
	}
	head = codec.AppendUint32(head, codec.CRC32C(head))

	out := make([]byte, 0, int(off))
	out = append(out, head...)
	for ci := range blocks {
		out = append(out, blocks[ci]...)
	}
	for ci := range dicts {
		if len(dicts[ci]) > 0 {
			out = append(out, dicts[ci]...)
		}
	}
	return out
}

// buildRaw packs one column as (N+1) uint32 offsets followed by the raw value
// bytes, the uncompressed layout a reader aliases without a copy.
func (b *Builder) buildRaw(ci int, n uint32) []byte {
	offsets := make([]byte, (int(n)+1)*4)
	var values []byte
	var cur uint32
	for d := uint32(0); d < n; d++ {
		codec.PutUint32(offsets[int(d)*4:], cur)
		if row := b.rows[d]; row != nil && row[ci] != nil {
			values = append(values, row[ci]...)
			cur += uint32(len(row[ci]))
		}
	}
	codec.PutUint32(offsets[int(n)*4:], cur)
	blk := make([]byte, 0, len(offsets)+len(values))
	blk = append(blk, offsets...)
	blk = append(blk, values...)
	return blk
}

// buildCompressed packs one column with each value stored as an independent zstd
// frame. A CodecZstdDict column first derives a shared dictionary from its
// non-empty values and encodes every frame against it; the derived dictionary is
// returned so Build can append it to the region. An empty value stays a
// zero-length frame so the empty-value path reads back without a decode.
func (b *Builder) buildCompressed(ci int, n uint32) (data, dict []byte) {
	if b.cols[ci].Codec == CodecZstdDict {
		var samples [][]byte
		for d := uint32(0); d < n; d++ {
			if row := b.rows[d]; row != nil && len(row[ci]) > 0 {
				samples = append(samples, row[ci])
			}
		}
		dict = deriveDict(samples, maxDictBytes)
	}

	opts := []zstd.EOption{zstd.WithEncoderLevel(encoderLevel)}
	if len(dict) > 0 {
		opts = append(opts, zstd.WithEncoderDictRaw(dictID(ci), dict))
	} else {
		dict = nil // a CodecZstd column, or a dictionary column with no values
	}
	enc, _ := zstd.NewWriter(nil, opts...)
	defer func() { _ = enc.Close() }()

	offsets := make([]byte, (int(n)+1)*4)
	var values []byte
	var scratch []byte // reused frame buffer; values copies out of it each row
	var cur uint32
	for d := uint32(0); d < n; d++ {
		codec.PutUint32(offsets[int(d)*4:], cur)
		if row := b.rows[d]; row != nil && len(row[ci]) > 0 {
			scratch = enc.EncodeAll(row[ci], scratch[:0])
			values = append(values, scratch...)
			cur += uint32(len(scratch))
		}
	}
	codec.PutUint32(offsets[int(n)*4:], cur)
	blk := make([]byte, 0, len(offsets)+len(values))
	blk = append(blk, offsets...)
	blk = append(blk, values...)
	return blk, dict
}
