package rank

import (
	"math/rand"
	"testing"
)

// BenchmarkL2Rerank times the QuickScorer model over the canon L2 set: 200
// candidates against a 500-tree, 32-leaf ensemble over 40 features. The L2 budget
// is roughly 3 to 6 milliseconds for the whole set, so the per-document cost must
// stay in the low microseconds.
func BenchmarkL2Rerank(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const numFeatures, nTrees, leaves, nDocs = 40, 500, 32, 200
	trees := make([]*treeNode, nTrees)
	for i := range trees {
		trees[i] = randomTree(rng, numFeatures, leaves)
	}
	m := Compile(trees, numFeatures)
	docs := make([][]float64, nDocs)
	for i := range docs {
		docs[i] = randomDoc(rng, numFeatures)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sink float64
		for _, doc := range docs {
			sink += m.Score(doc)
		}
		_ = sink
	}
}

// BenchmarkCascadeRerank times the rerank through the cascade, the path that
// borrows one leaf bitvector per query and reuses it across all survivors rather
// than per document. It is the allocation gate for doc 11's pooled buffers: a
// whole query's rerank must allocate only the result slice, no per-document
// scratch, so the per-op alloc count stays flat as the survivor set grows.
func BenchmarkCascadeRerank(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const numFeatures, nTrees, leaves, nDocs = 40, 500, 32, 200
	trees := make([]*treeNode, nTrees)
	for i := range trees {
		trees[i] = randomTree(rng, numFeatures, leaves)
	}
	m := Compile(trees, numFeatures)
	rows := make([][]float64, nDocs+1)
	cands := make([]Candidate, nDocs)
	for i := 0; i < nDocs; i++ {
		rows[i+1] = randomDoc(rng, numFeatures)
		cands[i] = Candidate{DocID: uint32(i + 1)}
	}
	feat := func(id uint32) []float64 { return rows[id] }
	c := NewCascade(&Linear{Cols: []int{0}, Weights: []float64{1}, RetrievalWeight: 1}, m)
	rerankRows := make([][]float64, len(cands))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, cd := range cands {
			rerankRows[j] = feat(cd.DocID)
		}
		_ = c.ScoreRows(cands, rerankRows, nDocs)
	}
}

// BenchmarkNaiveRerank is the cost baseline the QuickScorer form has to beat: the
// same 200 documents over the same ensemble with the naive root-to-leaf walk.
func BenchmarkNaiveRerank(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const numFeatures, nTrees, leaves, nDocs = 40, 500, 32, 200
	trees := make([]*treeNode, nTrees)
	for i := range trees {
		trees[i] = randomTree(rng, numFeatures, leaves)
	}
	docs := make([][]float64, nDocs)
	for i := range docs {
		docs[i] = randomDoc(rng, numFeatures)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sink float64
		for _, doc := range docs {
			sink += naiveScore(doc, trees)
		}
		_ = sink
	}
}
