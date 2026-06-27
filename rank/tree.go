// Package rank implements the tsumugi ranking cascade, the stage that turns a
// retrieved candidate set into a final order. It is the L1 linear cut, the L2
// LambdaMART rerank with a QuickScorer-style bitvector evaluator, and the
// Reciprocal Rank Fusion that merges the lexical and dense retrieval planes. The
// model is trained offline; this package owns the inference and the cascade
// orchestration, not the training.
//
// The cascade is a telescoping funnel: fuse the retrieval planes, cut the fused
// set cheaply with a linear score, then rerank the survivors precisely with the
// tree model. Each stage runs over fewer documents than the last at a higher cost
// per document, so the expensive model only ever sees a couple hundred candidates.
//
// Correctness here is bit-exactness: the fast QuickScorer evaluator is checked
// against the naive root-to-leaf tree walk as an oracle, the same discipline the
// lexical and sparse regions use, because a one-off in the leaf-mask bit
// convention is invisible until a document exits the wrong leaf.
//
// The lineage is LambdaMART and QuickScorer (Lucchese et al., SIGIR 2015); this is
// a self-contained native implementation.
package rank

// treeNode is one node of a decision tree in pointer form, the shape the naive
// walk and the compiler both read. An internal node tests feature against thresh
// and branches; a leaf carries a value. Leaves are numbered left to right when the
// tree is compiled, which is the numbering the QuickScorer masks rely on.
type treeNode struct {
	feature     int
	thresh      float64
	left, right *treeNode
	leaf        bool
	value       float64
}

// newLeaf returns a leaf node with the given output value.
func newLeaf(value float64) *treeNode { return &treeNode{leaf: true, value: value} }

// newSplit returns an internal node testing feature against thresh, going left
// when the document value is at or below the threshold and right when it is above.
func newSplit(feature int, thresh float64, left, right *treeNode) *treeNode {
	return &treeNode{feature: feature, thresh: thresh, left: left, right: right}
}

// walk is the naive root-to-leaf evaluation: at each internal node compare the
// document's feature value against the threshold, branch, and repeat until a leaf.
// It is correct and slow, the cost baseline and the oracle the fast evaluator is
// checked against.
func (t *treeNode) walk(doc []float64) float64 {
	n := t
	for !n.leaf {
		if doc[n.feature] <= n.thresh {
			n = n.left
		} else {
			n = n.right
		}
	}
	return n.value
}

// naiveScore sums the naive walk over an ensemble, the oracle for the QuickScorer
// evaluator. The trees are summed in order, the same order the evaluator sums, so
// the two scores are identical float values when the evaluator is correct.
func naiveScore(doc []float64, trees []*treeNode) float64 {
	var score float64
	for _, t := range trees {
		score += t.walk(doc)
	}
	return score
}
