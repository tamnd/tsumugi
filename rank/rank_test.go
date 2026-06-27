package rank

import (
	"math/rand"
	"testing"
)

// randomTree builds a random binary decision tree of exactly targetLeaves leaves
// over numFeatures features, the kind of tree LightGBM grows, so the oracle
// exercises real branch structure rather than a single shape. It starts from a
// single leaf and repeatedly turns a random current leaf into a split with two
// fresh leaves, which keeps the leaf count exact and the shape irregular.
func randomTree(rng *rand.Rand, numFeatures, targetLeaves int) *treeNode {
	root := newLeaf(rng.NormFloat64())
	leaves := []*treeNode{root}
	for len(leaves) < targetLeaves {
		i := rng.Intn(len(leaves))
		n := leaves[i]
		left := newLeaf(rng.NormFloat64())
		right := newLeaf(rng.NormFloat64())
		n.leaf = false
		n.value = 0
		n.feature = rng.Intn(numFeatures)
		n.thresh = rng.Float64()
		n.left = left
		n.right = right
		// Replace the split node in the leaf pool with its two new leaves.
		leaves[i] = left
		leaves = append(leaves, right)
	}
	return root
}

func randomDoc(rng *rand.Rand, numFeatures int) []float64 {
	doc := make([]float64, numFeatures)
	for i := range doc {
		doc[i] = rng.Float64()
	}
	return doc
}

// TestQuickScorerMatchesNaive is the M8 gate: the compiled bitvector evaluator must
// produce exactly the score of the naive root-to-leaf walk, the oracle, for every
// document. A one-off in the leaf-mask bit convention is invisible until a document
// exits the wrong leaf, so the only honest check is bit-for-bit equality over many
// random forests and many random documents.
func TestQuickScorerMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const numFeatures = 30
	for forest := 0; forest < 200; forest++ {
		nTrees := 1 + rng.Intn(80)
		maxLeaves := 2 + rng.Intn(MaxLeaves-1)
		trees := make([]*treeNode, nTrees)
		for i := range trees {
			trees[i] = randomTree(rng, numFeatures, maxLeaves)
		}
		m := Compile(trees, numFeatures)
		for q := 0; q < 50; q++ {
			doc := randomDoc(rng, numFeatures)
			want := naiveScore(doc, trees)
			got := m.Score(doc)
			if got != want {
				t.Fatalf("forest %d doc %d: quickscorer %v, naive %v", forest, q, got, want)
			}
		}
	}
}

// TestWorkedTree walks the worked example from the spec: three internal nodes, four
// leaves, the document that must exit leaf 3, checked against the naive walk.
func TestWorkedTree(t *testing.T) {
	// leaves 0,1 under L (f2<=3), leaves 2,3 under Q (f7<=1), root R (f5<=10).
	l := newSplit(2, 3, newLeaf(10), newLeaf(11))
	q := newSplit(7, 1, newLeaf(12), newLeaf(13))
	r := newSplit(5, 10, l, q)
	m := Compile([]*treeNode{r}, 8)
	doc := make([]float64, 8)
	doc[5], doc[2], doc[7] = 20, 1, 5 // f5>10 right, f7>1 right -> leaf 3 value 13
	if got := m.Score(doc); got != 13 {
		t.Fatalf("worked tree score = %v, want 13 (leaf 3)", got)
	}
	if nv := naiveScore(doc, []*treeNode{r}); nv != 13 {
		t.Fatalf("naive worked tree = %v, want 13", nv)
	}
}

// TestMaxLeavesPanic checks a tree over the leaf bound is rejected at compile.
func TestMaxLeavesPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on tree over MaxLeaves")
		}
	}()
	// Build a left-deep tree with MaxLeaves+1 leaves.
	root := newLeaf(0.0)
	for i := 0; i < MaxLeaves; i++ {
		root = newSplit(0, 0.5, root, newLeaf(float64(i)))
	}
	Compile([]*treeNode{root}, 1)
}

// TestLinearCut checks the linear cut keeps the highest-scoring candidates and
// honors the retrieval weight.
func TestLinearCut(t *testing.T) {
	rows := map[uint32][]float64{
		1: {1.0, 0.0},
		2: {0.0, 1.0},
		3: {0.5, 0.5},
	}
	feat := func(id uint32) []float64 { return rows[id] }
	cands := []Candidate{{1, 0.0}, {2, 0.0}, {3, 1.0}}
	l := &Linear{Cols: []int{0, 1}, Weights: []float64{1, 1}, RetrievalWeight: 2}
	// scores: 1 -> 1, 2 -> 1, 3 -> 2*1 + 1 = 3. keep 2 drops doc with lowest.
	got := l.Cut(cands, feat, 2)
	if len(got) != 2 || got[0].DocID != 3 {
		t.Fatalf("linear cut = %v, want doc 3 first and length 2", got)
	}
}

// TestRRF checks fusion lifts a document both planes rank and is order independent.
func TestRRF(t *testing.T) {
	lexical := []uint32{1, 2, 3, 4}
	dense := []uint32{3, 5, 1, 6}
	out := RRF(60, lexical, dense)
	if out[0].DocID != 1 && out[0].DocID != 3 {
		t.Fatalf("expected 1 or 3 first, got %v", out)
	}
	if out[1].DocID != 1 && out[1].DocID != 3 {
		t.Fatalf("expected 1 and 3 in the top two, got %v", out)
	}
	// Order independence: swapping the lists gives the same fused order.
	swapped := RRF(60, dense, lexical)
	for i := range out {
		if out[i].DocID != swapped[i].DocID {
			t.Fatalf("RRF not order independent at %d: %v vs %v", i, out, swapped)
		}
	}
}

// TestCascadeEndToEnd plants a document that both planes rank well and that the
// model scores highest, and checks it heads the final order through fuse, cut, and
// rerank.
func TestCascadeEndToEnd(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const numFeatures = 10
	// A single-feature stump: feature 0 high means high score.
	stump := newSplit(0, 0.5, newLeaf(0.0), newLeaf(1.0))
	m := Compile([]*treeNode{stump}, numFeatures)

	rows := map[uint32][]float64{}
	for id := uint32(1); id <= 50; id++ {
		rows[id] = randomDoc(rng, numFeatures)
		rows[id][0] = 0.0 // low model feature
	}
	rows[42][0] = 1.0 // the planted winner: above the stump threshold
	feat := func(id uint32) []float64 { return rows[id] }

	lexical := make([]uint32, 0, 50)
	for id := uint32(1); id <= 50; id++ {
		lexical = append(lexical, id)
	}
	dense := []uint32{42, 7, 13}

	l1 := &Linear{Cols: []int{0}, Weights: []float64{1}, RetrievalWeight: 1}
	c := NewCascade(l1, m)
	got := c.Rank(lexical, dense, feat, 10)
	if len(got) == 0 || got[0].DocID != 42 {
		t.Fatalf("cascade top = %v, want doc 42 first", got)
	}
}
