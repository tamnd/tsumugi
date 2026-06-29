package feature

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// TestRoundTripLinear checks a linear column dequantizes close to its input
// across the range, within one quantization step.
func TestRoundTripLinear(t *testing.T) {
	b := NewBuilder([]Column{{FeatContentQuality, 1, QuantLinear}}, 1)
	const n = 256
	in := make([]float64, n)
	for d := 0; d < n; d++ {
		in[d] = float64(d) / float64(n) // 0..1
		b.Set(uint32(d), FeatContentQuality, in[d])
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	step := 1.0 / 255.0
	for d := 0; d < n; d++ {
		got, ok := r.Value(uint32(d), FeatContentQuality)
		if !ok {
			t.Fatalf("value %d missing", d)
		}
		if math.Abs(got-in[d]) > step {
			t.Fatalf("d=%d got=%.5f want=%.5f step=%.5f", d, got, in[d], step)
		}
	}
}

// TestRoundTripLog checks a log column keeps multiplicative error small over a
// heavy-tailed range, the regime PageRank and the lengths live in.
func TestRoundTripLog(t *testing.T) {
	b := NewBuilder([]Column{{FeatPageRank, 1, QuantLog}}, 1)
	const n = 1000
	rng := rand.New(rand.NewSource(1))
	in := make([]float64, n)
	for d := 0; d < n; d++ {
		// values spanning six orders of magnitude
		in[d] = math.Pow(10, rng.Float64()*6)
		b.Set(uint32(d), FeatPageRank, in[d])
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for d := 0; d < n; d++ {
		got, _ := r.Value(uint32(d), FeatPageRank)
		ratio := got / in[d]
		// one 8-bit log bucket over six decades is about exp(ln(10)*6/255) ~ 5.6%
		if ratio < 0.90 || ratio > 1.10 {
			t.Fatalf("d=%d got=%.3g want=%.3g ratio=%.3f", d, got, in[d], ratio)
		}
	}
}

// TestRoundTripCategorical checks a categorical column returns each stored code
// exactly, with no scaling smear: an id in must read the same id out. This is the
// property the linear and log schemes cannot give a code set, where 7 and 8 are
// different languages, not a value 7/255 apart.
func TestRoundTripCategorical(t *testing.T) {
	b := NewBuilder([]Column{{FeatLanguage, 1, QuantCategorical}}, 1)
	const n = 256
	for d := 0; d < n; d++ {
		b.Set(uint32(d), FeatLanguage, float64(d)) // every byte value, 0..255
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for d := 0; d < n; d++ {
		got, ok := r.Value(uint32(d), FeatLanguage)
		if !ok {
			t.Fatalf("value %d missing", d)
		}
		if got != float64(d) {
			t.Fatalf("d=%d got=%v want=%d, categorical id was not preserved exactly", d, got, d)
		}
	}
	// An unset document reads the zero id, the unknown language, not a fabricated one.
	b2 := NewBuilder([]Column{{FeatLanguage, 1, QuantCategorical}}, 1)
	b2.Set(5, FeatLanguage, 9)
	r2, err := Open(b2.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if got, _ := r2.Value(0, FeatLanguage); got != 0 {
		t.Fatalf("unset doc language id = %v, want 0 (unknown)", got)
	}
	if got, _ := r2.Value(5, FeatLanguage); got != 9 {
		t.Fatalf("set doc language id = %v, want 9", got)
	}
}

// TestCategoricalColumnInDefaultSchema pins that the language column rides the default
// schema as a categorical column, so a build round-trips a real language id rather than
// the old latin-ratio stand-in, and an out-of-range id folds to the byte ceiling rather
// than wrapping into another language's id.
func TestCategoricalColumnInDefaultSchema(t *testing.T) {
	var lang Column
	found := false
	for _, c := range DefaultSchema() {
		if c.ID == FeatLanguage {
			lang, found = c, true
		}
	}
	if !found {
		t.Fatal("FeatLanguage missing from the default schema")
	}
	if lang.Quant != QuantCategorical {
		t.Fatalf("FeatLanguage quant = %d, want categorical %d", lang.Quant, QuantCategorical)
	}
	b := NewBuilder(DefaultSchema(), SchemaVersion)
	b.Set(0, FeatLanguage, 3)
	b.Set(1, FeatLanguage, 300) // past a one-byte column
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if got, _ := r.Value(0, FeatLanguage); got != 3 {
		t.Fatalf("language id = %v, want 3", got)
	}
	if got, _ := r.Value(1, FeatLanguage); got != 255 {
		t.Fatalf("over-range id = %v, want the byte ceiling 255", got)
	}
}

// TestRankCorrelation is the build's own acceptance check for a column: the
// Spearman rank correlation between the full-precision values and their
// dequantized forms must be very high, since ranking only cares about order.
func TestRankCorrelation(t *testing.T) {
	b := NewBuilder([]Column{{FeatInDegree, 2, QuantLog}}, 1)
	const n = 5000
	rng := rand.New(rand.NewSource(2))
	in := make([]float64, n)
	for d := 0; d < n; d++ {
		in[d] = math.Floor(math.Pow(10, rng.Float64()*5)) // 1..100000 in-links
		b.Set(uint32(d), FeatInDegree, in[d])
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	got := make([]float64, n)
	for d := 0; d < n; d++ {
		got[d], _ = r.Value(uint32(d), FeatInDegree)
	}
	rho := spearman(in, got)
	if rho < 0.999 {
		t.Fatalf("two-byte log in-degree Spearman %.5f below 0.999", rho)
	}
}

// TestL1Scan checks the linear scorer sums weight*value plus the carried score,
// matching a direct computation.
func TestL1Scan(t *testing.T) {
	b := NewBuilder(DefaultSchema(), 1)
	const n = 100
	rng := rand.New(rand.NewSource(3))
	for d := 0; d < n; d++ {
		b.Set(uint32(d), FeatStaticRank, rng.Float64())
		b.Set(uint32(d), FeatPageRank, math.Pow(10, rng.Float64()*4))
		b.Set(uint32(d), FeatContentQuality, rng.Float64())
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	cands := []uint32{3, 17, 42, 99}
	carried := []float64{1, 2, 3, 4}
	w := L1Weights{
		IDs:     []FeatureID{FeatStaticRank, FeatPageRank, FeatContentQuality},
		Weights: []float64{2.0, 0.5, 1.5},
	}
	out := r.L1Scan(cands, w, carried)
	for i, d := range cands {
		want := carried[i]
		for j, id := range w.IDs {
			v, _ := r.Value(d, id)
			want += w.Weights[j] * v
		}
		if math.Abs(out[i]-want) > 1e-9 {
			t.Fatalf("cand %d: L1Scan=%.6f direct=%.6f", d, out[i], want)
		}
	}
}

// TestDefaultSchemaLayout confirms the default schema packs without overlap and
// the stride is the sum of column widths.
func TestDefaultSchemaLayout(t *testing.T) {
	cols := DefaultSchema()
	r, err := Open(NewBuilder(cols, 7).Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	var sum int
	for _, c := range cols {
		sum += int(c.Width)
	}
	if r.Stride() != sum {
		t.Fatalf("stride %d want %d", r.Stride(), sum)
	}
	if r.SchemaVersion() != 7 {
		t.Fatalf("schema version %d want 7", r.SchemaVersion())
	}
}

// TestCorruptionRejected flips a header byte and truncates the rows.
func TestCorruptionRejected(t *testing.T) {
	b := NewBuilder(DefaultSchema(), 1)
	b.Set(0, FeatStaticRank, 0.5)
	good := b.Build()

	bad := append([]byte(nil), good...)
	bad[12] ^= 0xff // row count, inside header CRC
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	// Drop a row's worth of bytes off the end.
	if _, err := Open(good[:len(good)-1]); err == nil {
		t.Fatal("truncated rows accepted")
	}
}

// spearman computes the Spearman rank correlation of two equal-length series.
func spearman(a, b []float64) float64 {
	ra := ranks(a)
	rb := ranks(b)
	return pearson(ra, rb)
}

func ranks(x []float64) []float64 {
	type pair struct {
		v float64
		i int
	}
	ps := make([]pair, len(x))
	for i, v := range x {
		ps[i] = pair{v, i}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].v < ps[j].v })
	r := make([]float64, len(x))
	// average ranks for ties
	for i := 0; i < len(ps); {
		j := i
		for j < len(ps) && ps[j].v == ps[i].v {
			j++
		}
		avg := float64(i+j-1) / 2.0
		for k := i; k < j; k++ {
			r[ps[k].i] = avg
		}
		i = j
	}
	return r
}

func pearson(a, b []float64) float64 {
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma /= float64(len(a))
	mb /= float64(len(b))
	var num, da, db float64
	for i := range a {
		x := a[i] - ma
		y := b[i] - mb
		num += x * y
		da += x * x
		db += y * y
	}
	if da == 0 || db == 0 {
		return 1
	}
	return num / math.Sqrt(da*db)
}
