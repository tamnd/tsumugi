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

// BenchmarkBuildImpact measures the impact-ordered build against the docID-ordered build
// over the same corpus, so the impl note can quote what impact ordering costs at build
// time: the same term inversion plus a per-list impact sort in place of the docID sort.
func BenchmarkBuildImpact(b *testing.B) {
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bld.BuildImpact(impactFor)
	}
}

// BenchmarkBuildBM25 is the docID-ordered build baseline BenchmarkBuildImpact is read
// against, the same corpus and builder.
func BenchmarkBuildBM25(b *testing.B) {
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bld.Build()
	}
}

// BenchmarkSearchImpact measures the impact scorer's query latency on the mid-size shard,
// the number the impl note reads against the sub-10ms shard budget. This slice serves it
// from the exhaustive scan; the pruned traversal the next slice adds is benchmarked against
// this baseline.
func BenchmarkSearchImpact(b *testing.B) {
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	r, err := Open(bld.BuildImpact(impactFor))
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	queries := genQueries(2, 1000, 2000, 4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if _, err := r.SearchImpact(q, DefaultK); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}
