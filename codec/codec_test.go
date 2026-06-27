package codec

import (
	"math"
	"testing"
)

// TestFloat16RoundTrip checks Float16frombits inverts Float16bits and that the
// half-precision codec keeps the relative error within the 2^-11 a binary16
// mantissa allows, the bound the per-vector scalar storage relies on.
func TestFloat16RoundTrip(t *testing.T) {
	// Exact values a half can represent round-trip bit for bit.
	exact := []float32{0, 1, -1, 0.5, -0.5, 2, 0.25, 1.5, 65504, -65504}
	for _, f := range exact {
		got := Float16frombits(Float16bits(f))
		if got != f {
			t.Errorf("exact %v round-tripped to %v", f, got)
		}
	}

	// Values in the (0, 1] band the per-vector scalars live in: the relative error
	// must stay under one half-ULP, ~2^-11.
	for i := 1; i <= 1000; i++ {
		f := float32(i) / 1000
		got := Float16frombits(Float16bits(f))
		rel := math.Abs(float64(got-f)) / float64(f)
		if rel > 1.0/2048 {
			t.Errorf("%v round-tripped to %v, relative error %g over 2^-11", f, got, rel)
		}
	}
}

// TestFloat16Specials checks the saturating and non-finite edges: overflow to
// infinity, signed zero, and NaN preserved as NaN.
func TestFloat16Specials(t *testing.T) {
	if got := Float16frombits(Float16bits(1e30)); !math.IsInf(float64(got), 1) {
		t.Errorf("overflow did not saturate to +Inf, got %v", got)
	}
	if got := Float16frombits(Float16bits(-1e30)); !math.IsInf(float64(got), -1) {
		t.Errorf("overflow did not saturate to -Inf, got %v", got)
	}
	if got := Float16frombits(Float16bits(float32(math.Inf(1)))); !math.IsInf(float64(got), 1) {
		t.Errorf("Inf did not round-trip, got %v", got)
	}
	if got := Float16frombits(Float16bits(float32(math.NaN()))); !math.IsNaN(float64(got)) {
		t.Errorf("NaN did not round-trip, got %v", got)
	}
	neg := Float16frombits(Float16bits(float32(math.Copysign(0, -1))))
	if math.Signbit(float64(neg)) != true || neg != 0 {
		t.Errorf("negative zero did not round-trip, got %v", neg)
	}
}
