package dense

import (
	"math"
	"testing"
)

// dot is the cosine of two unit vectors, the similarity the dense plane ranks by. The
// encoder normalizes its output, so for normalized vectors the dot is the cosine directly.
func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

// TestHashTableUnitVectors pins that every token's table vector is a unit vector with
// exactly nonzero set coordinates, the random-indexing signature the encoder pools.
func TestHashTableUnitVectors(t *testing.T) {
	tbl := NewHashTable(64, 4, 1)
	for _, tok := range []string{"rust", "search", "engine", "東京", "tokyo"} {
		v := tbl.Lookup(tok)
		if len(v) != 64 {
			t.Fatalf("%q: dim %d, want 64", tok, len(v))
		}
		set := 0
		for _, x := range v {
			if x != 0 {
				set++
			}
		}
		if set != 4 {
			t.Errorf("%q: %d nonzero coords, want 4", tok, set)
		}
		if n := norm(v); math.Abs(n-1) > 1e-6 {
			t.Errorf("%q: norm %v, want 1", tok, n)
		}
	}
}

// TestHashTableDeterministic pins that a token always hashes to the same vector, the
// property that makes the build side and the query side agree when they share a table.
func TestHashTableDeterministic(t *testing.T) {
	a := NewHashTable(48, 3, 7)
	b := NewHashTable(48, 3, 7)
	v1 := a.Lookup("relevance")
	v2 := b.Lookup("relevance")
	for i := range v1 {
		if v1[i] != v2[i] {
			t.Fatalf("coord %d: %v vs %v, table not deterministic", i, v1[i], v2[i])
		}
	}
}

// TestHashTableSeedSeparates pins that a different seed yields a different space, so a
// deployment can pin a private table.
func TestHashTableSeedSeparates(t *testing.T) {
	a := NewHashTable(64, 4, 1).Lookup("query")
	b := NewHashTable(64, 4, 2).Lookup("query")
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds produced identical vectors")
	}
}

// TestEncodeMeanPoolNormalized pins the static encoder's contract: its output is a unit
// vector, the L2-normalized mean of its terms' vectors.
func TestEncodeMeanPoolNormalized(t *testing.T) {
	enc := NewStatic(NewHashTable(64, 4, 1))
	v := enc.Encode([]string{"web", "search", "engine"})
	if enc.Dim() != 64 {
		t.Fatalf("dim %d, want 64", enc.Dim())
	}
	if n := norm(v); math.Abs(n-1) > 1e-6 {
		t.Errorf("encoded norm %v, want 1", n)
	}
}

// TestEncodeEmptyIsZero pins that a query with no known terms encodes to the zero vector,
// the dense plane's no-signal value rather than a spurious neighbor.
func TestEncodeEmptyIsZero(t *testing.T) {
	enc := NewStatic(NewHashTable(32, 4, 1))
	for _, terms := range [][]string{nil, {}, {""}} {
		v := enc.Encode(terms)
		for _, x := range v {
			if x != 0 {
				t.Fatalf("terms %v encoded to nonzero %v", terms, v)
			}
		}
	}
}

// TestEncodeOverlapRanksHigher pins the bag-of-words property the dense plane relies on:
// two queries that share terms are nearer than two that share none. This is the whole
// reason the static encoder is useful, so it is the load-bearing test.
func TestEncodeOverlapRanksHigher(t *testing.T) {
	enc := NewStatic(NewHashTable(256, 6, 1))
	base := enc.Encode([]string{"rust", "memory", "safety"})
	overlap := enc.Encode([]string{"rust", "memory", "model"})
	unrelated := enc.Encode([]string{"italian", "pasta", "recipe"})
	near := dot(base, overlap)
	far := dot(base, unrelated)
	if near <= far {
		t.Errorf("overlap similarity %v not above unrelated %v", near, far)
	}
}

// TestEncodeDeterministic pins that the same terms encode to the same vector across calls,
// the order-independent build-equals-query property the dense plane needs.
func TestEncodeDeterministic(t *testing.T) {
	enc := NewStatic(NewHashTable(128, 5, 3))
	a := enc.Encode([]string{"alpha", "beta", "gamma"})
	b := enc.Encode([]string{"alpha", "beta", "gamma"})
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("coord %d differs across calls", i)
		}
	}
}

// TestEncodeOrderInvariant pins that mean-pooling is order-free: the same bag of terms in
// any order encodes identically, matching the bag-of-words model with no term interaction.
func TestEncodeOrderInvariant(t *testing.T) {
	enc := NewStatic(NewHashTable(128, 5, 3))
	a := enc.Encode([]string{"one", "two", "three"})
	b := enc.Encode([]string{"three", "one", "two"})
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-6 {
			t.Fatalf("coord %d order-dependent: %v vs %v", i, a[i], b[i])
		}
	}
}

// TestEncodeSkipsUnknown pins that unknown terms are skipped, not pooled as zeros: a query
// of two known terms plus one the table omits encodes like the two known terms alone. With
// HashTable every nonempty token is known, so a Vocabulary table drives this.
func TestEncodeSkipsUnknown(t *testing.T) {
	vocab := NewVocabulary(4, map[string][]float32{
		"a": {1, 0, 0, 0},
		"b": {0, 1, 0, 0},
	})
	enc := NewStatic(vocab)
	withUnknown := enc.Encode([]string{"a", "b", "missing"})
	known := enc.Encode([]string{"a", "b"})
	for i := range known {
		if math.Abs(float64(withUnknown[i]-known[i])) > 1e-6 {
			t.Fatalf("coord %d: unknown term changed pooling %v vs %v", i, withUnknown[i], known[i])
		}
	}
}

// TestVocabularyRejectsWrongDim pins that a vocabulary drops vectors that are not the
// declared dimension, so Lookup's every-vector-is-dim-long contract holds.
func TestVocabularyRejectsWrongDim(t *testing.T) {
	vocab := NewVocabulary(3, map[string][]float32{
		"ok":  {1, 0, 0},
		"bad": {1, 0},
	})
	if vocab.Lookup("bad") != nil {
		t.Error("wrong-dim vector was not dropped")
	}
	if vocab.Lookup("ok") == nil {
		t.Error("correct-dim vector was dropped")
	}
	if got := vocab.Tokens(); len(got) != 1 || got[0] != "ok" {
		t.Errorf("Tokens() = %v, want [ok]", got)
	}
}

// TestNilEncoderSafe pins that a nil encoder and a nil-table encoder are safe no-ops, the
// dense-plane-off path.
func TestNilEncoderSafe(t *testing.T) {
	var e *StaticEncoder
	if e.Dim() != 0 || e.Encode([]string{"x"}) != nil {
		t.Error("nil encoder not a safe no-op")
	}
	e2 := NewStatic(nil)
	if e2.Dim() != 0 || e2.Encode([]string{"x"}) != nil {
		t.Error("nil-table encoder not a safe no-op")
	}
}

func BenchmarkEncode(b *testing.B) {
	enc := NewStatic(NewHashTable(256, 6, 1))
	terms := []string{"web", "scale", "search", "engine", "ranking"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Encode(terms)
	}
}
