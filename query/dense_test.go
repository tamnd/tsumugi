package query_test

import (
	"testing"

	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/query"
)

// fakeEncoder returns a fixed vector for any terms, so the ApplyDense wiring is tested
// without a real table.
type fakeEncoder struct {
	vec []float32
}

func (f fakeEncoder) Encode([]string) []float32 { return f.vec }

// TestApplyDenseFillsVec checks that the encoder's vector is packed into DenseVec, decodes
// back to the same vector, and rewrites the cache key so a dense query caches distinctly
// from the bare one.
func TestApplyDenseFillsVec(t *testing.T) {
	pq := query.Parse("web search", def, query.SoftOR)
	before := pq.NormKey
	want := []float32{0.5, -0.5, 0.5, -0.5}
	pq.ApplyDense(fakeEncoder{vec: want})

	if pq.DenseVec == nil {
		t.Fatal("DenseVec not filled")
	}
	got := query.DecodeDenseVec(pq.DenseVec)
	if len(got) != len(want) {
		t.Fatalf("decoded len %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("coord %d: got %v, want %v", i, got[i], want[i])
		}
	}
	if pq.NormKey == before {
		t.Error("cache key was not recomputed after dense encoding")
	}
}

// TestApplyDenseZeroIsNoSignal checks that an all-zero vector, the encoder's no-signal
// return, is left unstored so the query stays dense-off in the cache key.
func TestApplyDenseZeroIsNoSignal(t *testing.T) {
	pq := query.Parse("rust", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyDense(fakeEncoder{vec: []float32{0, 0, 0, 0}})
	if pq.DenseVec != nil {
		t.Error("zero vector was stored as a dense signal")
	}
	if pq.NormKey != before {
		t.Error("zero vector changed the cache key")
	}
}

// TestApplyDenseNilIsNoop checks the broker can call ApplyDense with no encoder configured
// and nothing changes, the dense-plane-off path.
func TestApplyDenseNilIsNoop(t *testing.T) {
	pq := query.Parse("rust borrow checker", def, query.SoftOR)
	before := pq.NormKey
	pq.ApplyDense(nil)
	if pq.NormKey != before {
		t.Error("nil encoder changed the cache key")
	}
	if pq.DenseVec != nil {
		t.Error("nil encoder filled DenseVec")
	}
}

// TestDenseVecRoundTrip checks the codec is lossless over the float32 values the encoder
// produces, including negatives and fractional values.
func TestDenseVecRoundTrip(t *testing.T) {
	want := []float32{0, 1, -1, 0.125, -0.333, 1e-7, 123456.5}
	got := query.DecodeDenseVec(query.EncodeDenseVec(want))
	if len(got) != len(want) {
		t.Fatalf("len %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("coord %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// TestDecodeDenseVecMalformed checks a byte slice whose length is not a multiple of four
// decodes to nil rather than a truncated vector.
func TestDecodeDenseVecMalformed(t *testing.T) {
	for _, b := range [][]byte{nil, {}, {1}, {1, 2, 3}, {1, 2, 3, 4, 5}} {
		if v := query.DecodeDenseVec(b); v != nil {
			t.Errorf("DecodeDenseVec(%v) = %v, want nil", b, v)
		}
	}
}

// TestApplyDenseWithStaticEncoder wires the real static encoder into the query pipeline
// and proves two queries that share terms encode to nearer vectors than two that share
// none, end to end through Parse and ApplyDense.
func TestApplyDenseWithStaticEncoder(t *testing.T) {
	enc := dense.NewStatic(dense.NewHashTable(256, 6, 1))

	base := query.Parse("rust memory safety", def, query.SoftOR)
	overlap := query.Parse("rust memory model", def, query.SoftOR)
	unrelated := query.Parse("italian pasta recipe", def, query.SoftOR)
	for _, pq := range []*query.ParsedQuery{base, overlap, unrelated} {
		pq.ApplyDense(enc)
		if pq.DenseVec == nil {
			t.Fatal("static encoder produced no dense vector for a real query")
		}
	}

	bv := query.DecodeDenseVec(base.DenseVec)
	ov := query.DecodeDenseVec(overlap.DenseVec)
	uv := query.DecodeDenseVec(unrelated.DenseVec)
	near := cosine(bv, ov)
	far := cosine(bv, uv)
	if near <= far {
		t.Errorf("overlap cosine %v not above unrelated %v", near, far)
	}
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}
