package rank

import (
	"math"
	"testing"
)

// TestBinningIsMonotone checks the core binning invariant the leaf-wise split scan
// relies on: a larger raw value never lands in a lower bin, so a bin-boundary split
// stands in for a raw-value threshold. It also checks that a feature with few distinct
// values is binned exactly, one bin per value.
func TestBinningIsMonotone(t *testing.T) {
	r := lcg(7)
	const n, nf = 500, 4
	features := make([][]float64, n)
	for i := range features {
		row := make([]float64, nf)
		row[0] = r.next()                   // dense, continuous
		row[1] = float64(int(r.next() * 5)) // 5 distinct values
		row[2] = r.next() * 100
		row[3] = 3.0 // constant
		features[i] = row
	}
	b := newBinned(features, 255)

	for f := 0; f < nf; f++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if features[i][f] < features[j][f] && b.idx[f][i] > b.idx[f][j] {
					t.Fatalf("feature %d: value %v binned to %d but smaller value %v binned to %d",
						f, features[i][f], b.idx[f][i], features[j][f], b.idx[f][j])
				}
			}
		}
	}

	// Feature 1 has 5 distinct values, so it gets exactly 5 bins.
	if b.nbins[1] != 5 {
		t.Fatalf("feature 1: expected 5 bins for 5 distinct values, got %d", b.nbins[1])
	}
	// The constant feature collapses to a single bin.
	if b.nbins[3] != 1 {
		t.Fatalf("feature 3: expected 1 bin for a constant column, got %d", b.nbins[3])
	}
}

// TestBinThresholdRoundsToRawValue checks that the bin upper bounds are exact raw
// values from the column, so a split threshold read off a bin boundary is a real
// value every left-side sample is at or below. With more distinct values than bins,
// the bounds stay strictly increasing.
func TestBinThresholdRoundsToRawValue(t *testing.T) {
	col := make([]float64, 1000)
	r := lcg(11)
	for i := range col {
		col[i] = r.next()
	}
	upper := binUpperBounds(col, 16)
	if len(upper) == 0 || len(upper) > 16 {
		t.Fatalf("expected 1..16 bounds, got %d", len(upper))
	}
	for i := 1; i < len(upper); i++ {
		if upper[i] <= upper[i-1] {
			t.Fatalf("bounds not strictly increasing at %d: %v <= %v", i, upper[i], upper[i-1])
		}
	}
	// Every value maps into a bin, and the top bound covers the maximum.
	max := col[0]
	for _, v := range col {
		if v > max {
			max = v
		}
	}
	if upper[len(upper)-1] < max {
		t.Fatalf("top bound %v below column max %v", upper[len(upper)-1], max)
	}
	for _, v := range col {
		bin := binOf(upper, v)
		if bin < 0 || bin >= len(upper) {
			t.Fatalf("value %v mapped to out-of-range bin %d", v, bin)
		}
		if v > upper[bin] {
			t.Fatalf("value %v exceeds its bin bound %v", v, upper[bin])
		}
	}
}

// exhaustiveSplit is a test-only oracle for the histogram split scan: it finds the
// XGBoost best split by considering every distinct-value boundary directly on the raw
// values, the way the pre-histogram trainer did. At full bin resolution the histogram
// scan must agree with it feature, threshold, and gain.
func exhaustiveSplit(features [][]float64, grad, hess []float64, idx []int, minLeaf int, lambda float64) bestSplit {
	var gTot, hTot float64
	for _, i := range idx {
		gTot += grad[i]
		hTot += hess[i]
	}
	parent := gTot * gTot / (hTot + lambda)
	nf := len(features[idx[0]])
	best := bestSplit{}
	for f := 0; f < nf; f++ {
		order := make([]int, len(idx))
		copy(order, idx)
		// insertion sort by feature value ascending
		for i := 1; i < len(order); i++ {
			for j := i; j > 0 && features[order[j]][f] < features[order[j-1]][f]; j-- {
				order[j], order[j-1] = order[j-1], order[j]
			}
		}
		var gL, hL float64
		for s := 0; s < len(order)-1; s++ {
			i := order[s]
			gL += grad[i]
			hL += hess[i]
			vl := features[order[s]][f]
			vr := features[order[s+1]][f]
			if vl == vr {
				continue
			}
			left := s + 1
			right := len(order) - left
			if left < minLeaf || right < minLeaf {
				continue
			}
			gR := gTot - gL
			hR := hTot - hL
			gain := gL*gL/(hL+lambda) + gR*gR/(hR+lambda) - parent
			if gain > best.gain {
				best = bestSplit{ok: true, gain: gain, feat: f, thresh: vl}
			}
		}
	}
	return best
}

// TestHistogramMatchesExhaustiveGain proves the histogram split scan is exact at full
// resolution: with one bin per distinct value it must pick the same feature and gain
// an exhaustive raw-value scan does, and the bin-boundary threshold must separate the
// samples the same way the raw threshold does. This is the split-correctness gate for
// the fast fit.
func TestHistogramMatchesExhaustiveGain(t *testing.T) {
	r := lcg(23)
	const n, nf, minLeaf = 300, 5, 5
	const lambda = 1.0
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
		hess[i] = r.next() + 0.1
		idx[i] = i
	}

	// Full-resolution bins: one bin per distinct value, so the histogram scan is exact.
	b := newBinned(features, 1<<16-1)
	tb := &treeBuilder{b: b, grad: grad, hess: hess, p: fitParams{minSamplesLeaf: minLeaf, lambdaL2: lambda}}
	maxBins := 0
	for _, nb := range b.nbins {
		if nb > maxBins {
			maxBins = nb
		}
	}
	tb.hg = make([]float64, maxBins)
	tb.hh = make([]float64, maxBins)
	tb.hc = make([]int, maxBins)

	got := tb.findSplit(idx)
	want := exhaustiveSplit(features, grad, hess, idx, minLeaf, lambda)

	if got.ok != want.ok {
		t.Fatalf("split existence mismatch: histogram ok=%v exhaustive ok=%v", got.ok, want.ok)
	}
	if got.feat != want.feat {
		t.Fatalf("feature mismatch: histogram %d exhaustive %d", got.feat, want.feat)
	}
	if math.Abs(got.gain-want.gain) > 1e-9 {
		t.Fatalf("gain mismatch: histogram %v exhaustive %v", got.gain, want.gain)
	}
	// The histogram threshold is the left bin's upper bound, a real value; every
	// left-side sample must be at or below it and every right-side sample above it,
	// splitting the set exactly where the raw threshold does.
	for _, i := range idx {
		leftHist := features[i][got.feat] <= got.thresh
		leftRaw := features[i][want.feat] <= want.thresh
		if leftHist != leftRaw {
			t.Fatalf("row %d lands on different sides: histogram left=%v raw left=%v", i, leftHist, leftRaw)
		}
	}
}
