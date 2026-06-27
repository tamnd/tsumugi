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
	"math"

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

// Float16bits converts a float32 to its IEEE 754 half-precision (binary16) bit
// pattern, rounding the mantissa to nearest even. It is the codec for the small
// per-vector scalars a region stores at half width: a value that only needs three
// or four significant digits (a calibration scalar, a unit norm) costs two bytes
// instead of four. Overflow saturates to infinity, subnormals round toward zero,
// and NaN is preserved as a quiet NaN.
func Float16bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	if (b>>23)&0xff == 0xff { // Inf or NaN
		if mant != 0 {
			return sign | 0x7e00 // quiet NaN
		}
		return sign | 0x7c00 // Inf
	}
	if exp >= 0x1f { // overflow to Inf
		return sign | 0x7c00
	}
	if exp <= 0 { // subnormal or zero
		if exp < -10 { // too small to represent, round to signed zero
			return sign
		}
		mant |= 0x800000 // restore the implicit leading one
		shift := uint32(14 - exp)
		half := mant >> shift
		rem := mant & ((1 << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || (rem == halfway && half&1 == 1) {
			half++
		}
		return sign | uint16(half)
	}
	// normal: 5-bit exponent, 10-bit mantissa, round to nearest even
	half := sign | uint16(exp<<10) | uint16(mant>>13)
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && (mant>>13)&1 == 1) {
		half++ // a mantissa carry ripples into the exponent field correctly
	}
	return half
}

// Float16frombits converts an IEEE 754 half-precision bit pattern back to float32,
// the exact inverse of Float16bits for every finite value it produced.
func Float16frombits(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: shift the mantissa up until the implicit one appears
		e := uint32(0)
		for mant&0x400 == 0 {
			mant <<= 1
			e++
		}
		mant &= 0x3ff
		exp32 := uint32(127 - 15 - e + 1)
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	case 0x1f:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // Inf
		}
		return math.Float32frombits(sign | 0x7fc00000) // NaN
	default:
		exp32 := exp - 15 + 127
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	}
}

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
