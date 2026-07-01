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

// benchImpactShard builds the mid-size impact-ordered shard the impact search benchmarks
// share, so the pruned and exhaustive numbers are read off one corpus.
func benchImpactShard(b *testing.B) (*Region, []string) {
	b.Helper()
	docs := genCorpus(1, 50000, 2000)
	bld := NewBuilder(DefaultParams())
	for i, d := range docs {
		bld.AddDoc(uint32(i), d)
	}
	r, err := Open(bld.BuildImpact(impactFor))
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	return r, genQueries(2, 1000, 2000, 4)
}

// BenchmarkSearchImpact measures the impact scorer's query latency on the mid-size shard,
// the number the impl note reads against the sub-10ms shard budget. It now serves from the
// pruned early-termination traversal; BenchmarkSearchImpactExhaustive is the full-scan
// baseline the skip win is read against.
func BenchmarkSearchImpact(b *testing.B) {
	r, queries := benchImpactShard(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if _, err := r.SearchImpact(q, DefaultK); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}

// BenchmarkSearchImpactSmallK measures the pruned traversal at k=10, where the top-k settles
// early and the skip discards the most work, the query shape the impact ordering exists for.
func BenchmarkSearchImpactSmallK(b *testing.B) {
	r, queries := benchImpactShard(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		if _, err := r.SearchImpact(q, 10); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}

// BenchmarkSearchImpactExhaustive is the full-scan baseline: the exhaustive scorer decodes
// every posting of every list. The gap to BenchmarkSearchImpact at the same k is the pruned
// traversal's skip win.
func BenchmarkSearchImpactExhaustive(b *testing.B) {
	r, queries := benchImpactShard(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		infos := r.termInfos(Analyze(q), nil)
		if _, err := r.exhaustiveImpact(infos, DefaultK); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}
