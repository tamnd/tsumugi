// Package codec holds the low-level encoding primitives every tsumugi region is
// built from: the CRC32C checksum that guards each region and the footer, and
// the variable-length integer helpers the postings, graph, and footer share.
//
// These mirror the primitives kv and tatami settled on, kept here so the rest
// of tsumugi has one place to reach for a checksum or a varint and never
// reinvents either.
package codec

import (
	"encoding/binary"
	"hash/crc32"

	"github.com/cespare/xxhash/v2"
)

// castagnoli is the CRC32C polynomial table. CRC32C is the same checksum kv and
// tatami use; on amd64 and arm64 the standard library lowers it to the hardware
// CRC instruction, so guarding every region costs almost nothing.
var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// CRC32C returns the Castagnoli CRC32 of b.
func CRC32C(b []byte) uint32 {
	return crc32.Checksum(b, castagnoli)
}

// AppendUvarint appends the unsigned varint encoding of x to b and returns the
// extended slice.
func AppendUvarint(b []byte, x uint64) []byte {
	return binary.AppendUvarint(b, x)
}

// AppendVarint appends the signed (zig-zag) varint encoding of x to b.
func AppendVarint(b []byte, x int64) []byte {
	return binary.AppendVarint(b, x)
}

// Uvarint decodes an unsigned varint from b, returning the value and the number
// of bytes consumed. A non-positive count signals a truncated or overflowing
// encoding, matching encoding/binary.Uvarint.
func Uvarint(b []byte) (uint64, int) {
	return binary.Uvarint(b)
}

// Varint decodes a signed (zig-zag) varint from b.
func Varint(b []byte) (int64, int) {
	return binary.Varint(b)
}

// AppendUint64 appends x in little-endian to b.
func AppendUint64(b []byte, x uint64) []byte {
	return binary.LittleEndian.AppendUint64(b, x)
}

// AppendUint32 appends x in little-endian to b.
func AppendUint32(b []byte, x uint32) []byte {
	return binary.LittleEndian.AppendUint32(b, x)
}

// AppendUint16 appends x in little-endian to b.
func AppendUint16(b []byte, x uint16) []byte {
	return binary.LittleEndian.AppendUint16(b, x)
}

// PutUint16 writes x in little-endian into the front of b.
func PutUint16(b []byte, x uint16) { binary.LittleEndian.PutUint16(b, x) }

// Uint16 reads a little-endian uint16 from the front of b.
func Uint16(b []byte) uint16 { return binary.LittleEndian.Uint16(b) }

// PutUint32 writes x in little-endian into the front of b.
func PutUint32(b []byte, x uint32) { binary.LittleEndian.PutUint32(b, x) }

// Uint64 reads a little-endian uint64 from the front of b.
func Uint64(b []byte) uint64 { return binary.LittleEndian.Uint64(b) }

// Uint32 reads a little-endian uint32 from the front of b.
func Uint32(b []byte) uint32 { return binary.LittleEndian.Uint32(b) }

// XXHash64 returns the 64-bit xxHash of b, the same hash kv and tatami use for
// membership filters and routing.
func XXHash64(b []byte) uint64 { return xxhash.Sum64(b) }

// XXHash64Pair returns two independent 64-bit hashes of b for the
// Kirsch-Mitzenmacher double-hashing a bloom filter needs: the second hash is
// the xxHash of the first mixed with a salt, which decorrelates the two cheaply.
func XXHash64Pair(b []byte) (uint64, uint64) {
	h1 := xxhash.Sum64(b)
	var salt [8]byte
	binary.LittleEndian.PutUint64(salt[:], h1)
	h2 := xxhash.Sum64(salt[:])
	if h2 == 0 {
		h2 = 1 // keep the stride nonzero so the probes spread
	}
	return h1, h2
}
