package rank

import (
	"math/bits"
	"sort"
)

// MaxLeaves is the leaf bound a tree must stay within for the single-word
// QuickScorer form: one bit per leaf in a uint64. LightGBM is configured to grow
// trees at or under this bound for exactly this reason, which ties a training
// hyperparameter to the inference data structure.
const MaxLeaves = 64

// nodeEntry is one internal node in the feature-major layout: the threshold it
// tests, the tree it belongs to, and the leaf mask applied when the test is false
// (the document value is above the threshold, so the walk goes right and the
// node's left-subtree leaves are ruled out). The mask has a zero in each ruled-out
// leaf position and a one elsewhere.
type nodeEntry struct {
	threshold float64
	treeID    int32
	mask      uint64
}

// Model is the compiled QuickScorer ensemble, laid out by feature rather than by
// tree so the evaluator streams contiguous memory in a predictable scan. nodes[f]
// holds every internal node across all trees that splits on feature f, sorted by
// threshold ascending so the false-test nodes are a contiguous prefix the scan
// stops at. leafValue[t] holds tree t's leaf outputs in left-to-right order, and
// allOnes[t] is the initial all-leaves-live bitvector for tree t.
type Model struct {
	nodes       [][]nodeEntry
	leafValue   [][]float64
	allOnes     []uint64
	numTrees    int
	numFeatures int

	// schemaVersion and schemaHash carry the feature-matrix schema the model was
	// trained against, propagated from the ensemble by Compile. Zero means the model
	// was built without a stamp, the free Compile path that names no schema.
	schemaVersion uint16
	schemaHash    uint64
}

// NumTrees returns the ensemble size.
func (m *Model) NumTrees() int { return m.numTrees }

// NumFeatures returns the feature-vector width the model was trained on.
func (m *Model) NumFeatures() int { return m.numFeatures }

// SchemaVersion returns the feature-matrix schema version the model was trained
// against, or zero if the model carries no stamp.
func (m *Model) SchemaVersion() uint16 { return m.schemaVersion }

// SchemaHash returns the fingerprint of the feature-matrix schema the model was
// trained against, or zero if the model carries no stamp.
func (m *Model) SchemaHash() uint64 { return m.schemaHash }

// Compile restructures a pointer-form ensemble into the feature-major QuickScorer
// layout. It numbers each tree's leaves left to right, builds each internal node's
// leaf mask from the leaf range of its left subtree, and sorts each feature's
// nodes by threshold. It panics if a tree exceeds MaxLeaves or names a feature
// outside numFeatures, since both are build-time invariants the trainer guarantees.
func Compile(trees []*treeNode, numFeatures int) *Model {
	m := &Model{
		nodes:       make([][]nodeEntry, numFeatures),
		leafValue:   make([][]float64, len(trees)),
		allOnes:     make([]uint64, len(trees)),
		numTrees:    len(trees),
		numFeatures: numFeatures,
	}
	for t, root := range trees {
		var leaves []float64
		var walk func(n *treeNode) (lo, hi int)
		// walk assigns leaf indices left to right and returns the contiguous leaf
		// range [lo, hi) a subtree covers, which is what each split's mask needs.
		walk = func(n *treeNode) (int, int) {
			if n.leaf {
				idx := len(leaves)
				leaves = append(leaves, n.value)
				return idx, idx + 1
			}
			llo, lhi := walk(n.left)
			rlo, rhi := walk(n.right)
			_ = rlo
			// A false test at n rules out n's left-subtree leaves [llo, lhi).
			mask := ^uint64(0)
			for i := llo; i < lhi; i++ {
				mask &^= uint64(1) << uint(i)
			}
			if n.feature < 0 || n.feature >= numFeatures {
				panic("rank: tree splits on feature outside numFeatures")
			}
			m.nodes[n.feature] = append(m.nodes[n.feature], nodeEntry{
				threshold: n.thresh,
				treeID:    int32(t),
				mask:      mask,
			})
			return llo, rhi
		}
		walk(root)
		if len(leaves) > MaxLeaves {
			panic("rank: tree exceeds MaxLeaves")
		}
		m.leafValue[t] = leaves
		if len(leaves) == MaxLeaves {
			m.allOnes[t] = ^uint64(0)
		} else {
			m.allOnes[t] = (uint64(1) << uint(len(leaves))) - 1
		}
	}
	for f := range m.nodes {
		nd := m.nodes[f]
		sort.Slice(nd, func(i, j int) bool { return nd[i].threshold < nd[j].threshold })
	}
	return m
}

// Score evaluates the ensemble for one document. It starts every tree with all
// leaves live, sweeps each feature's nodes in threshold order applying the masks
// of the false-test nodes (stopping at the first true test, since the rest are
// true too), then reads each tree's exit leaf as the lowest surviving bit and sums
// the leaf values in tree order. The trees are summed in the same order as the
// naive walk, so the result is bit-identical to it.
func (m *Model) Score(doc []float64) float64 {
	v := make([]uint64, m.numTrees)
	copy(v, m.allOnes)
	for f := range m.nodes {
		x := doc[f]
		for _, nd := range m.nodes[f] {
			if x <= nd.threshold {
				break
			}
			v[nd.treeID] &= nd.mask
		}
	}
	var score float64
	for t := range v {
		leaf := bits.TrailingZeros64(v[t])
		score += m.leafValue[t][leaf]
	}
	return score
}
