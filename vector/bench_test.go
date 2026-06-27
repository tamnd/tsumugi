package vector

import (
	"math/rand"
	"testing"
)

// benchRegion builds a clustered corpus of the given size and dimension and opens
// it once, so the benchmark times only the search.
func benchRegion(b *testing.B, n, dim, clusters int) (*Region, [][]float32) {
	b.Helper()
	corpus := clusteredCorpus(n, dim, clusters, 42)
	bd := NewBuilder(dim)
	for _, v := range corpus {
		bd.Add(v)
	}
	r, err := Open(bd.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	return r, corpus
}

// BenchmarkSearch times the two-part dense search at the canon ef. The sub-10ms
// shard latency budget is the bar; a single dense plane should sit well under it.
func BenchmarkSearch(b *testing.B) {
	r, _ := benchRegion(b, 100_000, 128, 200)
	rng := rand.New(rand.NewSource(7))
	queries := make([][]float32, 256)
	for i := range queries {
		queries[i] = normalize(randVec(rng, 128))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Search(queries[i%len(queries)], 10, DefaultEfSearch, DefaultRerankDepth)
	}
}

// BenchmarkSearchNoRerank times the memory-light estimator path on the same
// corpus, the mode a shard runs when it drops the int8 copy.
func BenchmarkSearchNoRerank(b *testing.B) {
	corpus := clusteredCorpus(100_000, 128, 200, 42)
	bd := NewBuilder(128).WithRerank(false)
	for _, v := range corpus {
		bd.Add(v)
	}
	r, err := Open(bd.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	rng := rand.New(rand.NewSource(7))
	queries := make([][]float32, 256)
	for i := range queries {
		queries[i] = normalize(randVec(rng, 128))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Search(queries[i%len(queries)], 10, DefaultEfSearch, DefaultRerankDepth)
	}
}
