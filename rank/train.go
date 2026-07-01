package rank

import (
	"math"
	"sort"
)

// Dataset is a labeled learning-to-rank training set in LETOR shape: a flat list
// of feature rows, each row's graded relevance label, and the query groups that
// partition the rows, Groups[q] giving the number of consecutive rows belonging to
// query q. Every row of Features has NumFeatures columns.
type Dataset struct {
	Features    [][]float64
	Labels      []float64
	Groups      []int
	NumFeatures int
}

// Params controls the LambdaMART fit. The defaults are a reasonable ranking
// configuration: a few hundred trees with a small learning rate, grown leaf-wise and
// kept within the QuickScorer one-word leaf budget by MaxLeaves, with feature values
// histogram-binned so a fit is linear in the sample count.
//
// MaxLeaves bounds a tree by its number of leaves, the LightGBM shape; it is clamped
// to the QuickScorer leaf ceiling. MaxDepth, when positive, is an extra per-branch
// depth guard; with MaxLeaves set it can be left zero. If both are zero the fit falls
// back to a leaf budget one under the ceiling. MaxBins is the histogram resolution,
// defaulting to 255 when zero; a feature with fewer distinct values than MaxBins is
// binned exactly.
type Params struct {
	Rounds         int
	LearningRate   float64
	MaxLeaves      int
	MaxDepth       int
	MinSamplesLeaf int
	LambdaL2       float64
	MaxBins        int
	NDCGCutoff     int
}

// DefaultParams returns a sane LambdaMART configuration.
func DefaultParams() Params {
	return Params{
		Rounds:         300,
		LearningRate:   0.1,
		MaxLeaves:      32,
		MaxDepth:       0,
		MinSamplesLeaf: 5,
		LambdaL2:       1.0,
		MaxBins:        255,
		NDCGCutoff:     10,
	}
}

// Ensemble is a trained sequence of regression trees together with the feature
// width they were trained on. It is the artifact training produces; Compile turns
// it into the served QuickScorer model, and Save and LoadEnsemble persist it.
type Ensemble struct {
	trees       []*treeNode
	numFeatures int

	// schemaVersion and schemaHash record the feature-matrix schema the model was
	// trained against, so a serving node can refuse a model whose columns do not
	// line up with the shards it would score. Zero means unstamped: a hand-built or
	// legacy model that carries no schema guarantee, which the loader allows but does
	// not verify.
	schemaVersion uint16
	schemaHash    uint64
}

// NumTrees returns the ensemble size.
func (e *Ensemble) NumTrees() int { return len(e.trees) }

// SchemaVersion returns the feature-matrix schema version the ensemble was trained
// against, or zero if it was never stamped.
func (e *Ensemble) SchemaVersion() uint16 { return e.schemaVersion }

// SchemaHash returns the fingerprint of the feature-matrix schema the ensemble was
// trained against, or zero if it was never stamped.
func (e *Ensemble) SchemaHash() uint64 { return e.schemaHash }

// SetSchema stamps the ensemble with the feature-matrix schema it was trained
// against, the version and fingerprint a serving node checks before it ranks. A
// producer building a model for a collection calls this with the collection's
// feature schema so the artifact records what it expects to read.
func (e *Ensemble) SetSchema(version uint16, hash uint64) {
	e.schemaVersion = version
	e.schemaHash = hash
}

// Compile turns the trained ensemble into the served QuickScorer model, carrying the
// schema stamp through so the served model knows the feature schema it was trained
// against.
func (e *Ensemble) Compile() *Model {
	m := Compile(e.trees, e.numFeatures)
	m.schemaVersion = e.schemaVersion
	m.schemaHash = e.schemaHash
	return m
}

// Train fits a LambdaMART ensemble by gradient boosting against the NDCG-weighted
// pairwise objective. Each round it ranks every query by the current scores,
// accumulates per-document lambda gradients and hessians from the pairs the current
// order gets wrong weighted by the NDCG change a swap would make, fits one
// regression tree to those gradients, and adds it to the ensemble. The result is an
// order that pushes the most relevant documents up where it moves NDCG the most.
func Train(d *Dataset, p Params) *Ensemble {
	n := len(d.Features)
	scores := make([]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	fp := fitParams{
		maxLeaves:      p.MaxLeaves,
		maxDepth:       p.MaxDepth,
		minSamplesLeaf: p.MinSamplesLeaf,
		lambdaL2:       p.LambdaL2,
		learningRate:   p.LearningRate,
	}
	maxBins := p.MaxBins
	if maxBins <= 0 {
		maxBins = 255
	}
	// The feature values never change between rounds, only the gradients fitted
	// against them, so the histogram binning is computed once and reused for every
	// tree.
	bins := newBinned(d.Features, maxBins)
	trees := make([]*treeNode, 0, p.Rounds)
	for round := 0; round < p.Rounds; round++ {
		for i := range grad {
			grad[i] = 0
			hess[i] = 0
		}
		lambdas(d, scores, p.NDCGCutoff, grad, hess)
		t := fitTree(bins, grad, hess, idx, fp)
		trees = append(trees, t)
		for i := range scores {
			scores[i] += t.walk(d.Features[i])
		}
	}
	return &Ensemble{trees: trees, numFeatures: d.NumFeatures}
}

// lambdas accumulates the LambdaMART gradient and hessian for every document over
// every query group, given the current model scores. Within a query, for each pair
// where one document is more relevant than the other, the gradient pushes the more
// relevant one up and the less relevant one down by an amount that grows with how
// wrong the current order has them and with the NDCG change swapping them would
// cause, so the model spends its capacity where the metric moves.
func lambdas(d *Dataset, scores []float64, cutoff int, grad, hess []float64) {
	base := 0
	for _, size := range d.Groups {
		rows := base
		base += size
		if size < 2 {
			continue
		}
		// Order this query's documents by current score to get their ranks.
		local := make([]int, size)
		for i := range local {
			local[i] = rows + i
		}
		sort.Slice(local, func(a, b int) bool {
			if scores[local[a]] != scores[local[b]] {
				return scores[local[a]] > scores[local[b]]
			}
			return local[a] < local[b]
		})
		// Discounts at each rank position and the ideal DCG for normalization.
		discount := make([]float64, size)
		for r := 0; r < size; r++ {
			discount[r] = 1 / math.Log2(float64(r+2))
		}
		idcg := idealDCG(d.Labels, rows, size, cutoff)
		if idcg == 0 {
			continue
		}
		pos := make([]int, size) // pos[doc offset within query] -> rank
		for r, gi := range local {
			pos[gi-rows] = r
		}
		for a := 0; a < size; a++ {
			ia := rows + a
			for b := 0; b < size; b++ {
				ib := rows + b
				if d.Labels[ia] <= d.Labels[ib] {
					continue // only ordered pairs, more-relevant a over b
				}
				ga := math.Exp2(d.Labels[ia]) - 1
				gb := math.Exp2(d.Labels[ib]) - 1
				dNDCG := math.Abs((ga-gb)*(discount[pos[a]]-discount[pos[b]])) / idcg
				rho := 1 / (1 + math.Exp(scores[ia]-scores[ib]))
				lambda := dNDCG * rho
				w := dNDCG * rho * (1 - rho)
				grad[ia] += lambda
				grad[ib] -= lambda
				hess[ia] += w
				hess[ib] += w
			}
		}
	}
}

// idealDCG is the best achievable DCG for a query, the normalizer the lambda
// weights divide by, computed from the query's labels at the given cutoff.
func idealDCG(labels []float64, rows, size, cutoff int) float64 {
	ls := make([]float64, size)
	copy(ls, labels[rows:rows+size])
	sort.Sort(sort.Reverse(sort.Float64Slice(ls)))
	k := size
	if cutoff > 0 && cutoff < k {
		k = cutoff
	}
	var dcg float64
	for r := 0; r < k; r++ {
		dcg += (math.Exp2(ls[r]) - 1) / math.Log2(float64(r+2))
	}
	return dcg
}
