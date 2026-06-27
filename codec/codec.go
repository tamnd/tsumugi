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
