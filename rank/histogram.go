package rank

import "sort"

// binned holds the histogram binning of a training matrix: every feature's values
// are mapped once to a small ordered set of bins, so a tree fit finds its split by
// scanning per-bin gradient sums (cost proportional to the bin count) instead of
// re-sorting the samples at every node (cost proportional to n log n per node). This
// is the LightGBM histogram method the ranking spec pins (09-ranking, "LightGBM
// grows trees leaf-wise", the LightGBM training configuration), and it is what keeps
// a tree fit linear in the sample count rather than superlinear.
//
// The binning is computed once over the whole dataset and reused for every boosting
// round, because the feature values never change between rounds; only the gradients
// fitted against them do. A tree still stores raw-value thresholds, reconstructed
// from the bin upper bounds, so the served QuickScorer path reads the tree exactly as
// before and is unaware the fit went through bins.
type binned struct {
	// idx is feature-major: idx[f][row] is the bin of feature f for that row. A
	// uint16 holds up to 65535 bins, well past the 255-bin default, and halves the
	// resident cost against an int32 bin matrix the size of the feature matrix.
	idx [][]uint16

	// upper is feature-major bin upper bounds: upper[f][b] is the largest raw feature
	// value that lands in bin b of feature f. A split after bin b uses upper[f][b] as
	// the tree threshold, so every value in bins 0..b is at or below the threshold and
	// every value in bins b+1.. is above it. That makes a binned split reconstruct an
	// exact raw-value threshold, so at full bin resolution (one bin per distinct value)
	// the histogram split is identical to an exhaustive sorted scan.
	upper [][]float64

	nbins       []int
	numFeatures int
	n           int
}

// newBinned bins every feature of the row-major feature matrix into at most maxBins
// ordered bins. A feature with at most maxBins distinct values gets one bin per
// distinct value, so its splits stay exact; a denser feature is binned by equal
// frequency so each bin carries a comparable share of the samples, the resolution
// where it matters most.
func newBinned(features [][]float64, maxBins int) *binned {
	n := len(features)
	nf := 0
	if n > 0 {
		nf = len(features[0])
	}
	if maxBins < 2 {
		maxBins = 2
	}
	if maxBins > 1<<16-1 {
		maxBins = 1<<16 - 1
	}
	b := &binned{
		idx:         make([][]uint16, nf),
		upper:       make([][]float64, nf),
		nbins:       make([]int, nf),
		numFeatures: nf,
		n:           n,
	}
	col := make([]float64, n)
	for f := 0; f < nf; f++ {
		for i := 0; i < n; i++ {
			col[i] = features[i][f]
		}
		upper := binUpperBounds(col, maxBins)
		b.upper[f] = upper
		b.nbins[f] = len(upper)
		bins := make([]uint16, n)
		for i := 0; i < n; i++ {
			bins[i] = uint16(binOf(upper, features[i][f]))
		}
		b.idx[f] = bins
	}
	return b
}

// binUpperBounds returns the strictly increasing per-bin upper bounds for one
// feature column. With at most maxBins distinct values the bounds are the distinct
// values themselves, one bin each; with more, the bounds are equal-frequency
// quantiles of the sorted values, and the maximum value is always a bound so every
// value maps into a bin.
func binUpperBounds(col []float64, maxBins int) []float64 {
	vals := make([]float64, len(col))
	copy(vals, col)
	sort.Float64s(vals)

	distinct := make([]float64, 0, len(vals))
	for i, v := range vals {
		if i == 0 || v != vals[i-1] {
			distinct = append(distinct, v)
		}
	}
	if len(distinct) == 0 {
		return []float64{0}
	}
	if len(distinct) <= maxBins {
		return distinct
	}

	// More distinct values than bins: equal-frequency quantiles over the sorted
	// values with duplicates, so each bin holds a comparable count. Deduping keeps
	// the bounds strictly increasing when a quantile lands on a repeated value.
	upper := make([]float64, 0, maxBins)
	for b := 1; b <= maxBins; b++ {
		pos := b*len(vals)/maxBins - 1
		if pos < 0 {
			pos = 0
		}
		if pos >= len(vals) {
			pos = len(vals) - 1
		}
		u := vals[pos]
		if len(upper) == 0 || u > upper[len(upper)-1] {
			upper = append(upper, u)
		}
	}
	return upper
}

// binOf maps a raw value to its bin: the smallest b with upper[b] >= v. Because upper
// is non-decreasing, the mapping is monotone, so a larger value never lands in a lower
// bin, which is what lets a bin-boundary split stand in for a raw-value threshold.
func binOf(upper []float64, v float64) int {
	i := sort.Search(len(upper), func(i int) bool { return upper[i] >= v })
	if i >= len(upper) {
		i = len(upper) - 1
	}
	return i
}
