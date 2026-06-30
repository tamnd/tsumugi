// Package forward implements the .tsumugi forward region: the stored fields a
// result row displays and the document content the build derived features from.
// It is a columnar store indexed by dense docID, one value per document per
// column, with O(1) random access to any column of any document. The column and
// blob discipline is modelled on tatami's columnar store; the bytes here are the
// forward region's own FWD1 format, framed by the M0 container as RegionForward.
//
// The layout is an offset index over the dense docID space followed by the packed
// value bytes, so a snippet fetch reads one small column and never drags the body
// through cache. A column declared with a zstd codec is compressed per value: each
// value is an independent zstd frame, the offset index brackets the frame, and a
// CodecZstdDict column carries its own shared dictionary inside the region so a
// reader inflates only the one value it reaches. The region itself is stored
// uncompressed in the container (CodecNone) so opening a shard does not inflate
// every body, only the values a query touches do.
package forward

import (
	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi/codec"
)

// regionMagic marks the forward region, distinct from the container's TSM1 and
// the lexical region's LEX1 so a tool dumping raw bytes knows what it holds.
const regionMagic = "FWD1"

// regionVersion 2 adds the per-column dictionary descriptor and per-value frames;
// a version 1 region had neither, so the bump refuses to misread an old layout.
const regionVersion = 2

// maxDictBytes caps a column's embedded shared dictionary. A few kilobytes of
// representative values is enough context for the small text columns to compress
// against, and the cap keeps the per-shard dictionary cost bounded at scale.
const maxDictBytes = 16 << 10

// encoderLevel matches the container's whole-region setting so a per-value frame
// is no larger than the same bytes would have been under region-level zstd.
const encoderLevel = zstd.SpeedBetterCompression

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
// dictionary. A CodecNone column stores raw value bytes a reader aliases without
// a copy; a CodecZstd column stores each value as an independent zstd frame; a
// CodecZstdDict column does the same against a shared dictionary embedded in the
// region, so many small similar values share one context.
//
// CodecZstdDictBlocked is CodecZstdDict for large values that are read by a leading
// window rather than whole: the value is split into fixed-size uncompressed blocks,
// each an independent dictionary frame, behind a small per-value sub-index. A reader
// that wants the whole value decodes every block and the result is byte-for-byte the
// value CodecZstdDict would have stored; a reader that wants only a leading window
// (the L2 body scan, which reads at most a fixed rune cap) decodes only the blocks
// that window spans, so a long body costs the decode of its first blocks rather than
// the whole frame. The lever the throughput-versus-latency curve needs: the p99 tail
// is the body-decompression cost of the longest documents, and the window bounds it.
const (
	CodecNone            uint8 = 0
	CodecZstd            uint8 = 1
	CodecZstdDict        uint8 = 2
	CodecZstdDictBlocked uint8 = 3
)

// bodyBlockBytes is the uncompressed size of one block in a CodecZstdDictBlocked
// column. A value at or under this size is a single block, identical in cost to a
// CodecZstdDict frame; a larger value splits into this many bytes per block so a
// windowed read decodes ceil(window/bodyBlockBytes) blocks instead of the whole
// value. It trades a little compression ratio (each block loses cross-block context,
// softened by the shared dictionary) for a bounded windowed decode, so it is sized to
// the body scan window: a few blocks cover the cap, and only documents longer than
// the cap pay for blocks they never decode under the full-value path.
const bodyBlockBytes = 16 << 10

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

// colDesc is a column descriptor plus where its data block and, for a dictionary
// column, its embedded shared dictionary sit in the region. dictLen is zero for a
// column with no dictionary, including every CodecNone and CodecZstd column.
type colDesc struct {
	Column
	dataOff uint64
	dataLen uint64
	dictOff uint64
	dictLen uint64
}

// descTail is the fixed-size part of a column descriptor after its name: type,
// codec, flags, then the data and dictionary offset and length pairs.
const descTail = 1 + 1 + 4 + 8 + 8 + 8 + 8

func appendColDesc(b []byte, c colDesc) []byte {
	b = codec.AppendUint32(b, uint32(len(c.Name)))
	b = append(b, c.Name...)
	b = append(b, byte(c.Type), c.Codec)
	b = codec.AppendUint32(b, uint32(c.Flags))
	b = codec.AppendUint64(b, c.dataOff)
	b = codec.AppendUint64(b, c.dataLen)
	b = codec.AppendUint64(b, c.dictOff)
	b = codec.AppendUint64(b, c.dictLen)
	return b
}

func readColDesc(b []byte, off int) (colDesc, int, bool) {
	var c colDesc
	if off+4 > len(b) {
		return c, off, false
	}
	nl := int(codec.Uint32(b[off:]))
	off += 4
	if off+nl+descTail > len(b) {
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
	c.dictOff = codec.Uint64(b[off:])
	off += 8
	c.dictLen = codec.Uint64(b[off:])
	off += 8
	return c, off, true
}

// deriveDict builds a raw-content dictionary from a column's non-empty values, the
// shared context its per-value frames compress against. It walks the samples on an
// even stride so the dictionary spans the whole column rather than only its first
// rows, and stops at maxBytes. It returns nil when there is nothing to derive,
// which the builder reads as "store these frames without a dictionary".
func deriveDict(samples [][]byte, maxBytes int) []byte {
	if maxBytes <= 0 || len(samples) == 0 {
		return nil
	}
	step := len(samples) / 1024
	if step < 1 {
		step = 1
	}
	out := make([]byte, 0, maxBytes)
	for i := 0; i < len(samples) && len(out) < maxBytes; i += step {
		v := samples[i]
		if len(out)+len(v) > maxBytes {
			v = v[:maxBytes-len(out)]
		}
		out = append(out, v...)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// dictID maps a column index to the raw-dictionary id its frames are encoded
// under and the reader registers the dictionary against. The id must be non-zero
// (zstd reserves 0 for "no dictionary"), so it is the index plus one.
func dictID(ci int) uint32 { return uint32(ci) + 1 }
