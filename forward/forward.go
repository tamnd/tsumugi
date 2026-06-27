// Package forward implements the .tsumugi forward region: the stored fields a
// result row displays and the document content the build derived features from.
// It is a columnar store indexed by dense docID, one value per document per
// column, with O(1) random access to any column of any document. The column and
// blob discipline is modelled on tatami's columnar store; the bytes here are the
// forward region's own FWD1 format, framed by the M0 container as RegionForward.
//
// M2 keeps the layout deliberately plain: each column is an offset index over the
// dense docID space followed by the packed value bytes, so a snippet fetch reads
// one small column and never drags the body through cache. Region-level zstd in
// the container does the compression, so columns are stored uncompressed here and
// the per-column codec field records intent for a later shared-dictionary pass.
package forward

import "github.com/tamnd/tsumugi/codec"

// regionMagic marks the forward region, distinct from the container's TSM1 and
// the lexical region's LEX1 so a tool dumping raw bytes knows what it holds.
const regionMagic = "FWD1"

const regionVersion = 1

// ColType is a column's logical type. The store keeps every value as raw bytes;
// the type tells a reader how to interpret them and a tool how to print them.
type ColType uint8

const (
	ColString ColType = 1 // utf-8 text: url, title, snippet, body
	ColInt    ColType = 2 // varint-encoded integer: http status, timestamp
	ColFloat  ColType = 3 // ieee-754 bits: rarely used in the forward store
	ColBytes  ColType = 4 // opaque bytes: content hash
)

// Column codec ids mirror the container's: 0 none, 1 zstd, 2 zstd with a shared
// dictionary. M2 stores columns uncompressed and lets the region-level codec
// compress the whole region, so Codec is intent recorded in the schema rather
// than applied per column yet.
const (
	CodecNone     uint8 = 0
	CodecZstd     uint8 = 1
	CodecZstdDict uint8 = 2
)

// FlagBlob marks a large column, like the body, that a reader can skip when it
// only wants the small display fields. Every column has O(1) random access here,
// so the flag is a hint about access cost, not a different layout.
const FlagBlob uint16 = 0x0001

// Column is one column's schema: its name, logical type, intended codec, and
// flags. The same descriptors are what the container's footer schema section
// records, so a reader can list the columns without opening the region body.
type Column struct {
	Name  string
	Type  ColType
	Codec uint8
	Flags uint16
}

// colDesc is a column descriptor plus where its data block sits in the region.
type colDesc struct {
	Column
	dataOff uint64
	dataLen uint64
}

func appendColDesc(b []byte, c colDesc) []byte {
	b = codec.AppendUint32(b, uint32(len(c.Name)))
	b = append(b, c.Name...)
	b = append(b, byte(c.Type), c.Codec)
	b = codec.AppendUint32(b, uint32(c.Flags))
	b = codec.AppendUint64(b, c.dataOff)
	b = codec.AppendUint64(b, c.dataLen)
	return b
}

func readColDesc(b []byte, off int) (colDesc, int, bool) {
	var c colDesc
	if off+4 > len(b) {
		return c, off, false
	}
	nl := int(codec.Uint32(b[off:]))
	off += 4
	if off+nl+2+4+16 > len(b) {
		return c, off, false
	}
	c.Name = string(b[off : off+nl])
	off += nl
	c.Type = ColType(b[off])
	c.Codec = b[off+1]
	off += 2
	c.Flags = uint16(codec.Uint32(b[off:]))
	off += 4
	c.dataOff = codec.Uint64(b[off:])
	off += 8
	c.dataLen = codec.Uint64(b[off:])
	off += 8
	return c, off, true
}
