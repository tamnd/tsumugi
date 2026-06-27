package rank

import "sort"

// fitParams controls a single regression tree fit. The depth bound keeps the tree
// within the QuickScorer leaf budget (depth d gives at most 2^d leaves, so depth 6
// stays at or under 64), minSamplesLeaf guards against splitting on noise, and
// lambdaL2 regularizes the Newton leaf values.
type fitParams struct {
	maxDepth       int
	minSamplesLeaf int
	lambdaL2       float64
	learningRate   float64
}

// fitTree grows one regression tree on the gradient-boosting targets: grad is the
// per-sample LambdaMART gradient and hess its second-order weight. It builds the
// tree over the given sample indices with the XGBoost split gain and Newton leaf
// values, scaled by the learning rate so the returned tree is the shrunk
// contribution to the ensemble.
func fitTree(features [][]float64, grad, hess []float64, idx []int, p fitParams) *treeNode {
	b := &treeBuilder{features: features, grad: grad, hess: hess, p: p}
	return b.build(idx, 0)
}

type treeBuilder struct {
	features [][]float64
	grad     []float64
	hess     []float64
	p        fitParams
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

func (b *treeBuilder) build(idx []int, depth int) *treeNode {
	if depth >= b.p.maxDepth || len(idx) < 2*b.p.minSamplesLeaf {
		return newLeaf(b.leafValue(idx))
	}
	feat, thresh, ok := b.bestSplit(idx)
	if !ok {
		return newLeaf(b.leafValue(idx))
	}
	var left, right []int
	for _, i := range idx {
		if b.features[i][feat] <= thresh {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	if len(left) < b.p.minSamplesLeaf || len(right) < b.p.minSamplesLeaf {
		return newLeaf(b.leafValue(idx))
	}
	return newSplit(feat, thresh, b.build(left, depth+1), b.build(right, depth+1))
}

// bestSplit scans every feature for the threshold that maximizes the XGBoost split
// gain, G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l), over the sample set. It returns
// the chosen feature, the midpoint threshold, and whether a positive-gain split
// honoring the minimum leaf size was found.
func (b *treeBuilder) bestSplit(idx []int) (int, float64, bool) {
	var gTot, hTot float64
	for _, i := range idx {
		gTot += b.grad[i]
		hTot += b.hess[i]
	}
	parent := gTot * gTot / (hTot + b.p.lambdaL2)

	bestGain := 0.0
	bestFeat := -1
	bestThresh := 0.0
	numFeatures := len(b.features[idx[0]])
	order := make([]int, len(idx))
	for f := 0; f < numFeatures; f++ {
		copy(order, idx)
		sort.Slice(order, func(a, c int) bool { return b.features[order[a]][f] < b.features[order[c]][f] })
		var gL, hL float64
		for s := 0; s < len(order)-1; s++ {
			i := order[s]
			gL += b.grad[i]
			hL += b.hess[i]
			vl := b.features[order[s]][f]
			vr := b.features[order[s+1]][f]
			if vl == vr {
				continue // cannot split between equal values
			}
			left := s + 1
			right := len(order) - left
			if left < b.p.minSamplesLeaf || right < b.p.minSamplesLeaf {
				continue
			}
			gR := gTot - gL
			hR := hTot - hL
			gain := gL*gL/(hL+b.p.lambdaL2) + gR*gR/(hR+b.p.lambdaL2) - parent
			if gain > bestGain {
				bestGain = gain
				bestFeat = f
				bestThresh = (vl + vr) / 2
			}
		}
	}
	if bestFeat < 0 {
		return 0, 0, false
	}
	return bestFeat, bestThresh, true
}
