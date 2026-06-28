package vector

import "math"

// multiCode is a document's Extended-RaBitQ multi-bit code: each rotated dimension
// quantized to one of 2^bits levels instead of a single sign bit, plus the same two
// per-vector scalars the one-bit code carries. At four or five bits the code is sharp
// enough to rank without an int8 rerank pass, the no-rerank half-kilobyte path (spec
// 05 "The Extended-RaBitQ no-rerank knob"). The one-bit code (code.go) is the bits==1
// special case of this construction, and the estimator here reduces to that one exactly.
//
// levels holds the unsigned per-dimension codes in [0, 2^bits-1]; the dequantized
// signed level a code stands for is level - mid, where mid = (2^bits-1)/2, so the levels
// straddle zero the way the one-bit code's +/- signs do. scalar is <u, c>, the inner
// product of the unit rotated vector u with the signed-level vector c, the calibration
// the asymmetric estimator divides out. norm is the original L2 norm, to scale the
// estimate back to an un-normalized inner product.
type multiCode struct {
	bits   int
	scalar float32
	norm   float32
	levels []uint8
}

// mid returns the center of the symmetric level grid for a bits-wide code: 0.5 for one
// bit (levels {0,1} straddle 0 at +/-0.5), 7.5 for four bits, 15.5 for five.
func midLevel(bits int) float64 {
	n := 1 << uint(bits)
	return float64(n-1) / 2
}

// stepPerSigma is the Max-Lloyd optimal uniform-quantizer step, in units of the source
// standard deviation, for a unit-variance Gaussian at the given number of levels. After
// the random rotation a unit direction's coordinates are very close to Gaussian, so the
// step that minimizes the quantization error (and therefore the estimator's variance) is
// the classic table below. A measured per-bit sweep on the clustered corpus lands on
// exactly these values. Bit widths outside the table fall back to the sqrt(2 ln n)
// overload heuristic, which is within a few percent of optimal.
func stepPerSigma(bits int) float64 {
	switch bits {
	case 1:
		return 1.5956
	case 2:
		return 0.9957
	case 3:
		return 0.5860
	case 4:
		return 0.3352
	case 5:
		return 0.1881
	case 6:
		return 0.1041
	case 7:
		return 0.0569
	case 8:
		return 0.0308
	}
	n := float64(int(1) << uint(bits))
	return 2 * math.Sqrt(2*math.Log(n)) / (n - 1)
}

// encodeMulti quantizes a rotated vector to a bits-wide Extended-RaBitQ code. It works
// on the unit direction u = oRot/norm so the grid is scale-free, picks a per-vector step
// that maps the largest-magnitude coordinate to the extreme level (no clipping), and
// records scalar = <u, c> so the estimator is unbiased: as bits grow, c approaches u and
// the estimate approaches the true inner product.
func encodeMulti(oRot []float32, bits int) multiCode {
	rdim := len(oRot)
	var l2sq float64
	for _, x := range oRot {
		l2sq += float64(x) * float64(x)
	}
	norm := math.Sqrt(l2sq)
	levels := make([]uint8, rdim)
	if norm == 0 {
		return multiCode{bits: bits, scalar: 0, norm: 0, levels: levels}
	}

	mid := midLevel(bits)
	nlev := 1 << uint(bits)
	maxLevel := float64(nlev - 1)
	// After the random rotation the unit direction u = oRot/norm has coordinates close to
	// N(0, 1/rdim), so sigma = 1/sqrt(rdim). The step that minimizes the quantization error
	// (and therefore the estimator's variance) is the Max-Lloyd optimal uniform step for a
	// Gaussian, sigma times stepPerSigma(bits). This beats spreading the grid to the single
	// largest coordinate, which wastes resolution on outliers. For bits==1 delta is
	// irrelevant (the level is just the sign), so this reduces to the one-bit code exactly.
	sigma := 1 / math.Sqrt(float64(rdim))
	delta := stepPerSigma(bits) * sigma
	if delta == 0 {
		delta = 1
	}

	var scalar float64
	for i, x := range oRot {
		u := float64(x) / norm
		lvl := math.Round(u/delta + mid)
		if lvl < 0 {
			lvl = 0
		}
		if lvl > maxLevel {
			lvl = maxLevel
		}
		levels[i] = uint8(lvl)
		scalar += u * (lvl - mid) // <u, c>
	}
	return multiCode{bits: bits, scalar: float32(scalar), norm: float32(norm), levels: levels}
}

// estimateMulti is the asymmetric RaBitQ inner-product estimate of <q, o> from the
// full-precision rotated query and the document's multi-bit code: norm * <q_rot, c> /
// <u, c>, where c is the signed-level vector the code unpacks to. It is the no-rerank
// mode's final score (spec 05 "the second is the asymmetric RaBitQ estimator").
func (c multiCode) estimate(qRot []float32) float64 {
	if c.scalar == 0 {
		return 0
	}
	mid := midLevel(c.bits)
	var dot float64
	for i, lvl := range c.levels {
		dot += float64(qRot[i]) * (float64(lvl) - mid)
	}
	return float64(c.norm) * dot / float64(c.scalar)
}

// packLevels packs the per-dimension level codes into a bit stream, bits per level,
// least-significant-bit first, into ceil(rdim*bits/8) bytes. One bit is the existing
// word-packed sign layout's information but written through the same packer so the
// multi-bit path has one encoder.
func packLevels(levels []uint8, bits int) []byte {
	out := make([]byte, (len(levels)*bits+7)/8)
	pos := 0
	for _, lvl := range levels {
		v := uint(lvl)
		for b := 0; b < bits; b++ {
			if v&(1<<uint(b)) != 0 {
				out[pos>>3] |= 1 << uint(pos&7)
			}
			pos++
		}
	}
	return out
}

// unpackLevel reads the bits-wide level at dimension i from a packed bit stream.
func unpackLevel(packed []byte, i, bits int) uint8 {
	pos := i * bits
	var v uint
	for b := 0; b < bits; b++ {
		if packed[pos>>3]&(1<<uint(pos&7)) != 0 {
			v |= 1 << uint(b)
		}
		pos++
	}
	return uint8(v)
}

// estimateMultiBytes is estimate read straight from a packed multi-bit code in the
// mapped region, so the no-rerank scoring path never lifts the code onto the heap. It
// computes the same value as multiCode.estimate over the same levels.
func estimateMultiBytes(packed []byte, bits int, scalar, norm float32, qRot []float32) float64 {
	if scalar == 0 {
		return 0
	}
	mid := midLevel(bits)
	var dot float64
	for i := range qRot {
		dot += float64(qRot[i]) * (float64(unpackLevel(packed, i, bits)) - mid)
	}
	return float64(norm) * dot / float64(scalar)
}
