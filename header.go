package tsumugi

import (
	"encoding/binary"

	"github.com/tamnd/tsumugi/codec"
)

// Header is the fixed 64-byte head of a shard, little-endian throughout. The hot
// fields a reader needs first sit at the front. The layout matches spec 2067
// doc 03:
//
//	0  magic[4]          "TSM1"
//	4  version_major u16
//	6  version_minor u16
//	8  flags u64          capability bits
//	16 doc_count u32      N, the dense docID space size
//	20 region_count u32   number of region descriptors in the footer
//	24 footer_offset u64
//	32 footer_length u64
//	40 node_base u64      global node id of dense docID 0 when contiguous, else 0
//	48 build_epoch u64    build timestamp seconds, passed in, never read from a clock
//	56 header_crc u32     CRC32C of bytes [0:56]
//	60 reserved u32       zero
type Header struct {
	VersionMajor uint16
	VersionMinor uint16
	Flags        uint64
	DocCount     uint32
	RegionCount  uint32
	FooterOffset uint64
	FooterLength uint64
	NodeBase     uint64
	BuildEpoch   uint64
}

// encode writes the header into a fresh 64-byte slice, stamping the magic and
// the header CRC. The CRC covers bytes [0:56], everything up to the CRC field,
// so a flipped bit anywhere in the header is caught before any offset in it is
// trusted.
func (h *Header) encode() []byte {
	b := make([]byte, HeaderSize)
	copy(b[0:4], Magic)
	binary.LittleEndian.PutUint16(b[4:6], h.VersionMajor)
	binary.LittleEndian.PutUint16(b[6:8], h.VersionMinor)
	binary.LittleEndian.PutUint64(b[8:16], h.Flags)
	binary.LittleEndian.PutUint32(b[16:20], h.DocCount)
	binary.LittleEndian.PutUint32(b[20:24], h.RegionCount)
	binary.LittleEndian.PutUint64(b[24:32], h.FooterOffset)
	binary.LittleEndian.PutUint64(b[32:40], h.FooterLength)
	binary.LittleEndian.PutUint64(b[40:48], h.NodeBase)
	binary.LittleEndian.PutUint64(b[48:56], h.BuildEpoch)
	binary.LittleEndian.PutUint32(b[56:60], codec.CRC32C(b[0:56]))
	// bytes [60:64] stay zero (reserved)
	return b
}

// decodeHeader parses and validates the 64-byte header at the front of a shard.
func decodeHeader(b []byte) (Header, error) {
	var h Header
	if len(b) < HeaderSize {
		return h, ErrShortFile
	}
	if string(b[0:4]) != Magic {
		return h, ErrBadMagic
	}
	want := binary.LittleEndian.Uint32(b[56:60])
	if got := codec.CRC32C(b[0:56]); got != want {
		return h, ErrHeaderCRC
	}
	h.VersionMajor = binary.LittleEndian.Uint16(b[4:6])
	h.VersionMinor = binary.LittleEndian.Uint16(b[6:8])
	if h.VersionMajor != VersionMajor {
		return h, ErrBadVersion
	}
	h.Flags = binary.LittleEndian.Uint64(b[8:16])
	h.DocCount = binary.LittleEndian.Uint32(b[16:20])
	h.RegionCount = binary.LittleEndian.Uint32(b[20:24])
	h.FooterOffset = binary.LittleEndian.Uint64(b[24:32])
	h.FooterLength = binary.LittleEndian.Uint64(b[32:40])
	h.NodeBase = binary.LittleEndian.Uint64(b[40:48])
	h.BuildEpoch = binary.LittleEndian.Uint64(b[48:56])
	return h, nil
}

// Has reports whether a capability flag bit is set.
func (h Header) Has(flag uint64) bool { return h.Flags&flag != 0 }
