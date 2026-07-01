package sparse

import (
	"math"
	"math/rand"
	"testing"
)

// benchRegion builds a corpus shaped like a real impact index: a Zipfian
// vocabulary where a few terms are very common and most are rare, each posting
// carrying a heavy-tailed learned weight.
func benchRegion(b *testing.B, docs, vocab int) (*Region, []map[string]int) {
	b.Helper()
	rng := rand.New(rand.NewSource(7))
	bld := NewBuilder(uint32(docs))
	for term := 0; term < vocab; term++ {
		// Rank-skewed document frequency: term 0 hits ~half the corpus, the tail
		// only a handful.
		df := 1 + int(float64(docs)/2/float64(term+1))
		name := termName(term)
		// A term's strong documents cluster in a contiguous docID band, the way a
		// real index assigns nearby docIDs to related pages. The impact ordering
		// re-sorts postings by weight regardless, but the band gives a term a few
		// genuinely strong docs among many weak ones, the heavy-tailed weight skew
		// the anytime cutoff prunes on; a flat weight would top every block out near
		// the ceiling and leave nothing to skip.
		homeStart := uint32(rng.Intn(docs))
		homeLen := uint32(docs / 50)
		for i := 0; i < df; i++ {
			d := uint32(rng.Intn(docs))
			var w float64
			if d-homeStart < homeLen {
				w = math.Pow(10, 2+rng.Float64()) // strong: 100..1000
			} else {
				w = math.Pow(10, rng.Float64()) // weak: 1..10
			}
			bld.Add(name, d, w)
		}
	}
	r, err := Open(bld.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	queries := make([]map[string]int, 256)
	for i := range queries {
		q := map[string]int{}
		for t := 0; t < 3; t++ {
			q[termName(rng.Intn(vocab))] = 1 + rng.Intn(3)
		}
		queries[i] = q
	}
	return r, queries
}

func BenchmarkSearchAnytime(b *testing.B) {
	r, queries := benchRegion(b, 1_000_000, 50_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Search(queries[i%len(queries)], 10)
	}
}

func BenchmarkSearchExhaustive(b *testing.B) {
	r, queries := benchRegion(b, 1_000_000, 50_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.SearchExhaustive(queries[i%len(queries)], 10)
	}
}
