package lexical

import "testing"

// BenchmarkSearch measures BlockMax-WAND query latency on a mid-size shard, the
// number the M1 impl note quotes against the sub-10ms shard budget. The corpus
// and query set are deterministic so the number is comparable across runs.
func BenchmarkSearch(b *testing.B) {
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	r, err := Open(bld.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	queries := genQueries(2, 1000, 2000, 4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if _, err := r.Search(q, DefaultK); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}

// BenchmarkSearchSmallK measures the pruned path at a small k, where the rising
// threshold lets BlockMax-WAND skip most of the posting lists. This is the regime
// WAND is built for and where it pulls ahead of the exhaustive scan.
func BenchmarkSearchSmallK(b *testing.B) {
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	r, err := Open(bld.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	queries := genQueries(2, 1000, 2000, 4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if _, err := r.Search(q, 10); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}

// BenchmarkSearchVsExhaustive runs the same queries through the no-pruning scan
// so the impl note can quote the speedup BlockMax-WAND buys.
func BenchmarkSearchExhaustive(b *testing.B) {
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	r, err := Open(bld.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	queries := genQueries(2, 1000, 2000, 4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if _, err := r.SearchExhaustive(q, DefaultK); err != nil {
			b.Fatalf("exhaustive: %v", err)
		}
	}
}
