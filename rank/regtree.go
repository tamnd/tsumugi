package rank

import "container/heap"

// fitParams controls a single regression tree fit. The tree grows leaf-wise: the
// leaf whose best split gains the most is split first, and growth stops when the
// tree reaches maxLeaves leaves, the LightGBM shape the ranking spec pins
// (09-ranking, "LightGBM grows trees leaf-wise rather than depth-wise, so a tree is
// bounded by a maximum number of leaves rather than a maximum depth", num_leaves
// <= 64). maxDepth, when positive, is an extra guard that stops any single branch
// from growing past a depth even before the leaf budget is spent. minSamplesLeaf
// guards against splitting on noise, and lambdaL2 regularizes the Newton leaf
// values.
type fitParams struct {
	maxLeaves      int
	maxDepth       int
	minSamplesLeaf int
	lambdaL2       float64
	learningRate   float64
}

// fitTree grows one regression tree on the gradient-boosting targets: grad is the
// per-sample LambdaMART gradient and hess its second-order weight. It grows the tree
// leaf-wise over the given sample indices, finding each leaf's best split from
// per-feature histograms over the shared binning b, using the XGBoost split gain and
// Newton leaf values scaled by the learning rate, so the returned tree is the shrunk
// contribution to the ensemble. The tree stores raw-value thresholds reconstructed
// from the bin bounds, so the served QuickScorer path reads it unchanged.
func fitTree(b *binned, grad, hess []float64, idx []int, p fitParams) *treeNode {
	tb := &treeBuilder{
		b:    b,
		grad: grad,
		hess: hess,
		p:    p,
	}
	maxBins := 0
	for _, nb := range b.nbins {
		if nb > maxBins {
			maxBins = nb
		}
	}
	tb.hg = make([]float64, maxBins)
	tb.hh = make([]float64, maxBins)
	tb.hc = make([]int, maxBins)
	return tb.grow(idx)
}

type treeBuilder struct {
	b    *binned
	grad []float64
	hess []float64
	p    fitParams

	// scratch per-feature histograms, reused across every candidate split so a fit
	// allocates no per-node histogram storage. hg and hh accumulate the gradient and
	// hessian sums per bin, hc the sample count per bin.
	hg []float64
	hh []float64
	hc []int

	seq int // monotonic tie-breaker so leaf-wise growth is deterministic
}

// leafValue is the Newton step for a set of samples: the negative sum of gradients
// over the sum of hessians plus the L2 term, scaled by the learning rate. The
// gradient is defined so a positive value raises a document's score, so the leaf
// adds that directly.
func (b *treeBuilder) leafValue(idx []int) float64 {
	var g, h float64
	for _, i := range idx {
		g += b.grad[i]
		h += b.hess[i]
	}
	return b.p.learningRate * g / (h + b.p.lambdaL2)
}

// bestSplit is the outcome of scanning a leaf's samples for its best split: the
// feature and bin boundary that maximize the XGBoost gain, the raw-value threshold
// that bin boundary maps to, and whether a positive-gain split honoring the minimum
// leaf size exists.
type bestSplit struct {
	ok     bool
	gain   float64
	feat   int
	bin    int
	thresh float64
}

// growCand is a leaf waiting in the growth frontier: the tree node standing in for
// the leaf, the samples that reach it, its depth, and its best split already found so
// the leaf-wise loop can pick the highest-gain leaf without recomputing.
type growCand struct {
	node  *treeNode
	idx   []int
	depth int
	split bestSplit
	seq   int
}

// candHeap is a max-heap over the growth frontier keyed by split gain, so popping
// yields the leaf whose split improves the tree the most. Ties break toward the leaf
// created earlier, which keeps growth order deterministic.
type candHeap []*growCand

func (h candHeap) Len() int { return len(h) }
func (h candHeap) Less(i, j int) bool {
	if h[i].split.gain != h[j].split.gain {
		return h[i].split.gain > h[j].split.gain
	}
	return h[i].seq < h[j].seq
}
func (h candHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candHeap) Push(x any)   { *h = append(*h, x.(*growCand)) }
func (h *candHeap) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return c
}

// grow builds the tree leaf-wise from the root sample set. It seeds the frontier with
// the root, then repeatedly splits the highest-gain leaf until the leaf budget is
// spent or no leaf can split further, converting each chosen node from leaf to
// internal in place so the pointers the naive walk and compiler read stay valid.
func (b *treeBuilder) grow(rootIdx []int) *treeNode {
	maxLeaves := b.p.maxLeaves
	if maxLeaves <= 0 {
		if b.p.maxDepth <= 0 {
			maxLeaves = MaxLeaves - 1
		} else {
			maxLeaves = 1 << 30
		}
	}
	if maxLeaves > MaxLeaves {
		maxLeaves = MaxLeaves // never exceed the QuickScorer one-word leaf budget
	}

	root := newLeaf(b.leafValue(rootIdx))
	if len(rootIdx) < 2*b.p.minSamplesLeaf {
		return root
	}

	frontier := &candHeap{}
	heap.Init(frontier)
	b.offer(frontier, root, rootIdx, 0)

	leaves := 1
	for frontier.Len() > 0 && leaves < maxLeaves {
		c := heap.Pop(frontier).(*growCand)
		sp := c.split
		left, right := b.partition(c.idx, sp)
		if len(left) < b.p.minSamplesLeaf || len(right) < b.p.minSamplesLeaf {
			continue // split no longer viable, drop this leaf from the frontier
		}
		// Convert the chosen leaf into an internal node in place: the node pointer is
		// already referenced by its parent, so mutating it keeps the tree connected.
		c.node.leaf = false
		c.node.feature = sp.feat
		c.node.thresh = sp.thresh
		c.node.left = newLeaf(b.leafValue(left))
		c.node.right = newLeaf(b.leafValue(right))
		leaves++
		b.offer(frontier, c.node.left, left, c.depth+1)
		b.offer(frontier, c.node.right, right, c.depth+1)
	}
	return root
}

// offer finds the best split for a fresh leaf and, if one exists within the depth and
// minimum-leaf bounds, pushes it onto the frontier so the leaf-wise loop can weigh it
// against the other leaves. A leaf too small or too deep to split is simply left off
// the frontier, so it stays a leaf.
func (b *treeBuilder) offer(h *candHeap, node *treeNode, idx []int, depth int) {
	if len(idx) < 2*b.p.minSamplesLeaf {
		return
	}
	if b.p.maxDepth > 0 && depth >= b.p.maxDepth {
		return
	}
	sp := b.findSplit(idx)
	if !sp.ok {
		return
	}
	b.seq++
	heap.Push(h, &growCand{node: node, idx: idx, depth: depth, split: sp, seq: b.seq})
}

// findSplit scans every feature's per-bin histogram over the sample set and returns
// the bin boundary that maximizes the XGBoost split gain,
// G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l). Because the bins are ordered and the
// threshold is the bin's upper bound, at one bin per distinct value this finds the
// same split an exhaustive sorted scan would. The right-side minimum leaf size is
// checked as the scan accumulates left, and since counts only move from right to left
// the scan can stop once the right side would fall below the minimum.
func (b *treeBuilder) findSplit(idx []int) bestSplit {
	var gTot, hTot float64
	for _, i := range idx {
		gTot += b.grad[i]
		hTot += b.hess[i]
	}
	parent := gTot * gTot / (hTot + b.p.lambdaL2)
	n := len(idx)

	best := bestSplit{}
	for f := 0; f < b.b.numFeatures; f++ {
		nb := b.b.nbins[f]
		hg := b.hg[:nb]
		hh := b.hh[:nb]
		hc := b.hc[:nb]
		for k := 0; k < nb; k++ {
			hg[k] = 0
			hh[k] = 0
			hc[k] = 0
		}
		col := b.b.idx[f]
		for _, i := range idx {
			bin := col[i]
			hg[bin] += b.grad[i]
			hh[bin] += b.hess[i]
			hc[bin]++
		}
		var gL, hL float64
		cL := 0
		// A split after bin k sends bins 0..k left and k+1.. right, so the last bin
		// never yields a split (its right side is empty); stop at nb-1.
		for k := 0; k < nb-1; k++ {
			gL += hg[k]
			hL += hh[k]
			cL += hc[k]
			if cL < b.p.minSamplesLeaf {
				continue
			}
			cR := n - cL
			if cR < b.p.minSamplesLeaf {
				break // counts only leave the right side as k grows, so it stays too small
			}
			gR := gTot - gL
			hR := hTot - hL
			gain := gL*gL/(hL+b.p.lambdaL2) + gR*gR/(hR+b.p.lambdaL2) - parent
			if gain > best.gain {
				best = bestSplit{ok: true, gain: gain, feat: f, bin: k, thresh: b.b.upper[f][k]}
			}
		}
	}
	return best
}

// partition splits a leaf's samples by the chosen bin boundary: a sample goes left
// when its feature bin is at or below the split bin, right otherwise. This matches the
// raw-value test the stored threshold encodes, so the served tree branches the same
// way the fit did.
func (b *treeBuilder) partition(idx []int, sp bestSplit) (left, right []int) {
	col := b.b.idx[sp.feat]
	for _, i := range idx {
		if int(col[i]) <= sp.bin {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return left, right
}
