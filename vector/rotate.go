package vector

import (
	"math"
	"math/rand"
)

// rotator is the RaBitQ random orthogonal transform: a seeded sign flip followed
// by a normalized fast Walsh-Hadamard transform. Both factors are orthogonal, so
// the product is an isometry that preserves inner products and distances while
// spreading a vector's energy evenly across coordinates, which is what makes the
// per-coordinate one-bit quantization error uniform and analyzable. The build
// stores only the seed; the reader and the broker regenerate the identical
// transform from it, so document and query always meet in the same rotated frame.
type rotator struct {
	rdim  int       // rotated dimension, a power of two >= the kept dimension
	signs []float32 // per-coordinate +1/-1 from the seed
	scale float32   // 1/sqrt(rdim), the WHT normalization
}

// newRotator builds the transform for a kept dimension, padding up to the next
// power of two so the Hadamard transform applies cleanly.
func newRotator(dimKept int, seed int64) *rotator {
	rdim := 1
	for rdim < dimKept || rdim < 64 {
		rdim <<= 1
	}
	rng := rand.New(rand.NewSource(seed))
	signs := make([]float32, rdim)
	for i := range signs {
		if rng.Intn(2) == 0 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}
	return &rotator{rdim: rdim, signs: signs, scale: float32(1 / math.Sqrt(float64(rdim)))}
}

// rotate maps a kept-dimension vector into the rotated frame, returning a fresh
// rdim-length slice. The input is zero-padded to rdim, sign-flipped, run through
// the in-place Walsh-Hadamard transform, then scaled to keep the transform
// orthonormal.
func (r *rotator) rotate(v []float32) []float32 {
	out := make([]float32, r.rdim)
	for i := 0; i < len(v) && i < r.rdim; i++ {
		out[i] = v[i] * r.signs[i]
	}
	fwht(out)
	for i := range out {
		out[i] *= r.scale
	}
	return out
}

// fwht is the in-place fast Walsh-Hadamard transform on a power-of-two slice.
func fwht(a []float32) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j] = x + y
				a[j+h] = x - y
			}
		}
	}
}
