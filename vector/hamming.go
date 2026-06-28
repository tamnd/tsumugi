package vector

import (
	"math/bits"

	"github.com/tamnd/tsumugi/codec"
)

// The symmetric one-bit Hamming distance is the first of spec doc 05's three distance
// modes ("Three distance modes, by stage", line 556): query code against document code,
// the number of disagreeing sign bits, which is monotone in the angular distance after
// the random rotation. It is the cheapest mode, a popcount over a few uint64 words, the
// distance the spec uses for the bulk of the graph walk and for the build, where both
// sides are one bit and a sharper estimate is not yet worth the arithmetic (line 235,
// 402, 560). The asymmetric estimator (code.go) and the int8 dot (kernels.go) are the
// other two modes, reserved for scoring the small candidate set.

// queryBits is the query side of the symmetric path: the rotated query's sign bits,
// packed into uint64 words the same way encodeOneBit packs a document's. Quantizing the
// query to one bit is what makes the walk distance a pure popcount instead of the
// asymmetric estimator's masked sum.
type queryBits struct {
	bits []uint64
}

// encodeQueryBits one-bit-quantizes the rotated query for the symmetric Hamming walk
// (spec line 532, "q_code = one_bit_quantize(q_rot)"). The packing matches encodeOneBit
// exactly so a word-wise XOR lines the query's signs up against a document's.
func encodeQueryBits(qRot []float32) queryBits {
	words := (len(qRot) + 63) / 64
	bs := make([]uint64, words)
	for i, x := range qRot {
		if x >= 0 {
			bs[i>>6] |= 1 << uint(i&63)
		}
	}
	return queryBits{bits: bs}
}

// hammingWords is the symmetric distance between two packed one-bit codes: the popcount
// of their word-wise XOR, the count of disagreeing signs. Smaller is nearer, so it drops
// straight into the walk's smaller-is-nearer convention with no negation.
func hammingWords(a, b []uint64) int {
	var d int
	for i := range a {
		d += bits.OnesCount64(a[i] ^ b[i])
	}
	return d
}

// hammingBytes is hammingWords read straight from a document's one-bit code in the mapped
// region, so the walk never lifts the code onto the heap. rowBits is the words*8-byte sign
// block of little-endian uint64 words, q the query's packed sign bits. It returns the same
// count as hammingWords over the same signs, one word at a time.
func hammingBytes(rowBits []byte, q []uint64) int {
	var d int
	for w := range q {
		word := codec.Uint64(rowBits[w*8:])
		d += bits.OnesCount64(word ^ q[w])
	}
	return d
}
