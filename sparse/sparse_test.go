package sparse

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// buildRegion makes an impact index over a random corpus: a vocabulary of terms,
// each appearing in a random subset of documents with a heavy-tailed weight.
func buildRegion(t *testing.T, docs, vocab int, blockSize uint32, seed int64) *Region {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	b := NewBuilder(uint32(docs)).WithBlockSize(blockSize)
	for term := 0; term < vocab; term++ {
		name := termName(term)
		df := 1 + rng.Intn(docs/2+1)
		for i := 0; i < df; i++ {
			d := uint32(rng.Intn(docs))
			w := math.Pow(10, rng.Float64()*3) // 1..1000, heavy tailed
			b.Add(name, d, w)
		}
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return r
}

func termName(i int) string {
	return string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('a'+(i/676)%26))
}

// TestPrunedMatchesExhaustive is the M6 gate: Block-Max Pruning must return
// exactly what an exhaustive scan returns, ties included, over many random
// queries and k values. Integer query weights make the scoring exact, so the two
// results are bit-for-bit equal, not merely close.
func TestPrunedMatchesExhaustive(t *testing.T) {
	r := buildRegion(t, 4000, 300, 64, 1)
	rng := rand.New(rand.NewSource(2))
	for q := 0; q < 500; q++ {
		nTerms := 1 + rng.Intn(4)
		query := map[string]int{}
		for i := 0; i < nTerms; i++ {
			query[termName(rng.Intn(300))] = 1 + rng.Intn(3)
		}
		for _, k := range []int{1, 5, 10, 100} {
			got := r.Search(query, k)
			want := r.SearchExhaustive(query, k)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("query %v k=%d:\n pruned=%v\n  exact=%v", query, k, got, want)
			}
		}
	}
}

// TestBlockMaxUpperBound checks the stored block-max is a true upper bound: no
// posting in a block exceeds the block's recorded max.
func TestBlockMaxUpperBound(t *testing.T) {
	r := buildRegion(t, 2000, 100, 32, 3)
	for _, name := range r.names {
		for _, blk := range r.termBlocks(name) {
			_, impacts := blk.decode()
			for _, imp := range impacts {
				if imp > blk.max {
					t.Fatalf("term %s block %d: impact %d above max %d", name, blk.id, imp, blk.max)
				}
			}
		}
	}
}

// TestQuantizerRoundTrip checks the log quantizer keeps order and small
// multiplicative error over a heavy-tailed range.
func TestQuantizerRoundTrip(t *testing.T) {
	q := newQuantizer(0.01, 1000)
	prev := -1.0
	for _, w := range []float64{0.01, 0.1, 1, 10, 100, 1000} {
		lvl := q.quantize(w)
		got := q.dequant(lvl)
		ratio := got / w
		if ratio < 0.95 || ratio > 1.05 {
			t.Fatalf("w=%g dequant=%g ratio=%.3f", w, got, ratio)
		}
		if float64(lvl) <= prev {
			t.Fatalf("levels not monotone at w=%g", w)
		}
		prev = float64(lvl)
	}
	if q.quantize(0) != 0 || q.quantize(-1) != 0 {
		t.Fatal("non-positive weight must quantize to level 0")
	}
}

// TestMissingTermAndEmptyQuery covers absent terms and a query that hits nothing.
func TestMissingTermAndEmptyQuery(t *testing.T) {
	r := buildRegion(t, 500, 50, 32, 4)
	if got := r.Search(map[string]int{"zzzzz": 1}, 10); len(got) != 0 {
		t.Fatalf("absent term returned %d results", len(got))
	}
	if got := r.Search(map[string]int{}, 10); got != nil {
		t.Fatal("empty query returned results")
	}
}

// TestCorruptionRejected flips a header byte and truncates the region.
func TestCorruptionRejected(t *testing.T) {
	b := NewBuilder(100)
	b.Add("alpha", 1, 5)
	b.Add("alpha", 50, 9)
	good := b.Build()

	bad := append([]byte(nil), good...)
	bad[12] ^= 0xff // term count, inside header CRC
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	if _, err := Open(good[:headerLen+2]); err == nil {
		t.Fatal("truncated region accepted")
	}
}
