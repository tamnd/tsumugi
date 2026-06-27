package vector

import "math"

// oneBitCode is a document's RaBitQ one-bit code: the rotated sign bits packed
// into uint64 words, plus the two per-vector scalars the asymmetric estimator
// needs. scalar is <o_rot_unit, o_bar>, how much of the true direction the sign
// code captured; norm is the original L2 norm, so the estimate scales back to an
// un-normalized inner product.
type oneBitCode struct {
	scalar float32
	norm   float32
	bits   []uint64
}

// encodeOneBit quantizes a rotated vector to a one-bit code. o_bar is the unit
// vector whose coordinates are all +/-1/sqrt(rdim) with the signs of o_rot, the
// hypercube corner nearest the rotated direction. The stored scalar is the inner
// product of the true unit direction with o_bar, which works out to the rotated
// vector's L1 norm over its L2 norm over sqrt(rdim).
func encodeOneBit(oRot []float32) oneBitCode {
	rdim := len(oRot)
	words := (rdim + 63) / 64
	bitset := make([]uint64, words)
	var l1, l2sq float64
	for i, x := range oRot {
		if x >= 0 {
			bitset[i>>6] |= 1 << uint(i&63)
		}
		l1 += math.Abs(float64(x))
		l2sq += float64(x) * float64(x)
	}
	norm := math.Sqrt(l2sq)
	scalar := 0.0
	if norm > 0 {
		scalar = l1 / (norm * math.Sqrt(float64(rdim)))
	}
	return oneBitCode{scalar: float32(scalar), norm: float32(norm), bits: bitset}
}

// queryCode is the query side of the asymmetric estimator: the rotated query
// coordinates kept in full precision, dotted against a document's one-bit sign
// pattern. The query is never quantized, which is what makes the estimate sharper
// than a symmetric code-to-code comparison.
type queryCode struct {
	signed []float32
}

func encodeQuery(qRot []float32) queryCode {
	return queryCode{signed: qRot}
}

// estimate is the asymmetric RaBitQ inner-product estimate of <q, o> from the
// full-precision rotated query and the document's one-bit code. It forms
// <q_rot, o_bar> as the sum of query coordinates with the code's sign pattern,
// divides by the code scalar to correct for the direction the code lost, and
// scales by the stored norm. It is sharper than the symmetric Hamming because the
// query keeps full precision, and it is the score the no-rerank path would use.
func (c oneBitCode) estimate(q queryCode) float64 {
	rdim := len(q.signed)
	invSqrt := 1 / math.Sqrt(float64(rdim))
	var dot float64
	for i, x := range q.signed {
		if c.bits[i>>6]&(1<<uint(i&63)) != 0 {
			dot += float64(x)
		} else {
			dot -= float64(x)
		}
	}
	dot *= invSqrt
	if c.scalar == 0 {
		return 0
	}
	return float64(c.norm) * dot / float64(c.scalar)
}

// int8Quant maps a rotated vector to int8 codes against a shard-global scale. For
// a normalized embedding all vectors share a range, so one scale across the shard
// is enough and the rerank dot stays close to the full-precision dot.
type int8Quant struct {
	scale float32 // multiply an int8 level by this to get back a coordinate
}

func newInt8Quant(scale float32) int8Quant { return int8Quant{scale: scale} }

func (q int8Quant) encode(oRot []float32) []int8 {
	out := make([]int8, len(oRot))
	inv := float32(0)
	if q.scale > 0 {
		inv = 1 / q.scale
	}
	for i, x := range oRot {
		v := math.Round(float64(x * inv))
		if v > 127 {
			v = 127
		}
		if v < -128 {
			v = -128
		}
		out[i] = int8(v)
	}
	return out
}

// encodeQueryI8 quantizes the rotated query the same way for the int8 rerank dot.
func (q int8Quant) encodeQuery(qRot []float32) []int8 { return q.encode(qRot) }
