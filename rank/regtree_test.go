package rank

import (
	"math"
	"testing"
)

// countLeaves counts the leaves of a fitted tree, the quantity the leaf-wise growth
// bound limits.
func countLeaves(n *treeNode) int {
	if n.leaf {
		return 1
	}
	return countLeaves(n.left) + countLeaves(n.right)
}

// maxDepthOf returns the depth of the deepest leaf, zero for a bare leaf. Leaf-wise
// growth produces unbalanced trees, so this can exceed what a depth-wise fit of the
// same leaf count would.
func maxDepthOf(n *treeNode) int {
	if n.leaf {
		return 0
	}
	l := maxDepthOf(n.left)
	r := maxDepthOf(n.right)
	if l > r {
		return l + 1
	}
	return r + 1
}

// TestLeafWiseRespectsLeafBudget checks that leaf-wise growth never exceeds the
// configured leaf budget and that the budget, not a depth bound, is what limits the
// tree. It fits a single tree directly on strong gradients so growth wants to keep
// splitting.
func TestLeafWiseRespectsLeafBudget(t *testing.T) {
	r := lcg(31)
	const n, nf = 800, 6
	features := make([][]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		row := make([]float64, nf)
		for f := range row {
			row[f] = r.next()
		}
		features[i] = row
		// Gradient correlated with feature 0 so there is real signal to split on.
		grad[i] = row[0] - 0.5 + (r.next()-0.5)*0.2
		hess[i] = 1
		idx[i] = i
	}
	b := newBinned(features, 255)

	for _, budget := range []int{4, 8, 16, 31} {
		tree := fitTree(b, grad, hess, idx, fitParams{
			maxLeaves:      budget,
			minSamplesLeaf: 5,
			lambdaL2:       1,
			learningRate:   0.1,
		})
		leaves := countLeaves(tree)
		if leaves > budget {
			t.Fatalf("budget %d: tree has %d leaves, over budget", budget, leaves)
		}
		if leaves < 2 {
			t.Fatalf("budget %d: tree did not grow, only %d leaves", budget, leaves)
		}
	}
}

// TestLeafBudgetClampedToQuickScorer checks that a leaf budget above the QuickScorer
// one-word ceiling is clamped, so a trained tree can never carry more leaves than the
// inference bitvector can address.
func TestLeafBudgetClampedToQuickScorer(t *testing.T) {
	r := lcg(37)
	const n, nf = 4000, 8
	features := make([][]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		row := make([]float64, nf)
		for f := range row {
			row[f] = r.next()
		}
		features[i] = row
		grad[i] = r.next() - 0.5
		hess[i] = 1
		idx[i] = i
	}
	b := newBinned(features, 255)
	tree := fitTree(b, grad, hess, idx, fitParams{
		maxLeaves:      1000, // far over the ceiling
		minSamplesLeaf: 2,
		lambdaL2:       1,
		learningRate:   0.1,
	})
	if leaves := countLeaves(tree); leaves > MaxLeaves {
		t.Fatalf("tree has %d leaves, over the QuickScorer ceiling %d", leaves, MaxLeaves)
	}
}

// TestTrainDeterministic checks that two Train runs on the same data and params
// produce a bit-identical model, the property that makes a trained artifact
// reproducible and its serving behavior stable across rebuilds.
func TestTrainDeterministic(t *testing.T) {
	d := makeRankData(5, 60, 12, 8)
	p := DefaultParams()
	p.Rounds = 60

	a := Train(d, p).Compile()
	b := Train(d, p).Compile()

	for i := 0; i < len(d.Features); i++ {
		sa := a.Score(d.Features[i])
		sb := b.Score(d.Features[i])
		if math.Float64bits(sa) != math.Float64bits(sb) {
			t.Fatalf("row %d: nondeterministic score %v vs %v", i, sa, sb)
		}
	}
}

// TestLeafWiseGrowsUnbalanced checks the shape the spec calls for: leaf-wise growth
// spends its leaf budget where the gain is, so on data with signal concentrated in a
// region the tree grows deeper than a balanced tree of the same leaf count would. A
// tree with L leaves grown depth-wise would be depth ceil(log2 L); leaf-wise is free
// to go deeper.
func TestLeafWiseGrowsUnbalanced(t *testing.T) {
	r := lcg(41)
	const n, nf = 1200, 4
	features := make([][]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		row := make([]float64, nf)
		for f := range row {
			row[f] = r.next()
		}
		features[i] = row
		// Signal is concentrated in the upper region of feature 0: a fine staircase
		// above 0.5 that rewards splitting again and again there, and flat noise below
		// it with no gain. Leaf-wise growth keeps drilling into the upper region, so
		// the tree grows a long chain on that side rather than balancing.
		if row[0] > 0.5 {
			grad[i] = math.Floor((row[0]-0.5)*40) - 10
		} else {
			grad[i] = (r.next() - 0.5) * 0.001
		}
		hess[i] = 1
		idx[i] = i
	}
	b := newBinned(features, 255)
	const budget = 16
	tree := fitTree(b, grad, hess, idx, fitParams{
		maxLeaves:      budget,
		minSamplesLeaf: 5,
		lambdaL2:       1,
		learningRate:   0.1,
	})
	leaves := countLeaves(tree)
	depth := maxDepthOf(tree)
	balanced := int(math.Ceil(math.Log2(float64(leaves))))
	if depth <= balanced {
		t.Fatalf("leaf-wise tree not unbalanced: %d leaves at depth %d, balanced would be %d",
			leaves, depth, balanced)
	}
}
