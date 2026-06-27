package forward

import "github.com/tamnd/tsumugi/codec"

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
// reader reaches any value with two offset reads and a slice.
func (b *Builder) Build() []byte {
	n := b.docCount()

	// Pack each column's data block: (N+1) uint32 offsets, then value bytes.
	type block struct {
		bytes []byte
	}
	blocks := make([]block, len(b.cols))
	for ci := range b.cols {
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
		blocks[ci].bytes = blk
	}

	// Header is fixed up front, then column descriptors with their data offsets,
	// then the data blocks in column order.
	descs := make([]colDesc, len(b.cols))
	for ci := range b.cols {
		descs[ci] = colDesc{Column: b.cols[ci]}
	}

	// First lay out the header to learn where data begins. The header length
	// depends only on the descriptors, which have fixed-size tails plus names.
	headerLen := 4 + 4 + 4 + 4 // magic, version, colCount, rowCount
	for ci := range b.cols {
		headerLen += 4 + len(b.cols[ci].Name) + 2 + 4 + 8 + 8
	}
	headerLen += 4 // header_crc

	off := uint64(headerLen)
	for ci := range blocks {
		descs[ci].dataOff = off
		descs[ci].dataLen = uint64(len(blocks[ci].bytes))
		off += descs[ci].dataLen
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
		out = append(out, blocks[ci].bytes...)
	}
	return out
}
