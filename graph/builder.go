package graph

import "sort"

// Builder accumulates a graph by edges and encodes it into a GRA1 region. Edges
// arrive as (from, to) dense docID pairs in any order; Build sorts and dedupes
// each node's out-list, derives the transpose, and codes both planes.
type Builder struct {
	n      int
	out    [][]int32
	params Params
}

// NewBuilder returns a builder over a dense node space [0, n).
func NewBuilder(n int) *Builder {
	return &Builder{n: n, out: make([][]int32, n), params: DefaultParams()}
}

// WithParams overrides the adjacency-coder settings before any edges are added.
func (b *Builder) WithParams(p Params) *Builder {
	b.params = p
	return b
}

// AddEdge records a directed link from -> to. Self-loops and duplicates are
// dropped at Build, so a caller may add freely.
func (b *Builder) AddEdge(from, to int) {
	if from < 0 || from >= b.n || to < 0 || to >= b.n || from == to {
		return
	}
	b.out[from] = append(b.out[from], int32(to))
}

// Build encodes the forward and transpose planes and frames the region.
func (b *Builder) Build() []byte {
	// Sort and dedupe each out-list, count edges.
	var edges uint64
	for x := 0; x < b.n; x++ {
		b.out[x] = sortDedup(b.out[x])
		edges += uint64(len(b.out[x]))
	}

	// Derive the transpose: for each edge u->v, v gains in-neighbor u.
	in := make([][]int32, b.n)
	for u := 0; u < b.n; u++ {
		for _, v := range b.out[u] {
			in[v] = append(in[v], int32(u))
		}
	}
	for v := 0; v < b.n; v++ {
		sort.Slice(in[v], func(i, j int) bool { return in[v][i] < in[v][j] })
	}

	fwdAdj, fwdOff := encodePlane(b.out, b.params)
	xpAdj, xpOff := encodePlane(in, b.params)
	fwdEF := buildEF(fwdOff).encode()
	xpEF := buildEF(xpOff).encode()

	h := header{
		version:   regionVersion,
		params:    b.params,
		nodeCount: uint32(b.n),
		edgeCount: edges,
		fwdAdjLen: uint64(len(fwdAdj)),
		fwdEFLen:  uint64(len(fwdEF)),
		xpAdjLen:  uint64(len(xpAdj)),
		xpEFLen:   uint64(len(xpEF)),
	}
	region := h.encode()
	region = append(region, fwdAdj...)
	region = append(region, fwdEF...)
	region = append(region, xpAdj...)
	region = append(region, xpEF...)
	return region
}

func sortDedup(s []int32) []int32 {
	if len(s) == 0 {
		return s
	}
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	w := 1
	for i := 1; i < len(s); i++ {
		if s[i] != s[w-1] {
			s[w] = s[i]
			w++
		}
	}
	return s[:w]
}

// encodePlane codes every node's adjacency list back to back and returns the
// bitstream bytes and the N+1 bit offsets where each record starts. It is the
// in-RAM path: the whole plane is resident as a slice. The out-of-core builder
// shares the same record coding through encodeNodeCore, fed from a sliding window
// instead of the full slice, so both paths emit byte-identical bytes.
func encodePlane(lists [][]int32, p Params) (data []byte, offsets []uint64) {
	w := &bitWriter{}
	n := len(lists)
	offsets = make([]uint64, n+1)
	refDepth := make([]int, n)
	for x := 0; x < n; x++ {
		offsets[x] = w.bits
		// The window back-references are slice lookups against the resident plane.
		xx := x
		listAt := func(back int) []int32 { return lists[xx-back] }
		depthAt := func(back int) int { return refDepth[xx-back] }
		refDepth[x] = encodeNodeCore(w, x, lists[x], listAt, depthAt, p)
	}
	offsets[n] = w.bits
	return w.finish(), offsets
}

type interval struct {
	left   int
	length int
}

// encodeNodeCore writes one node's adjacency record: degree, optional reference
// and copy mask, intervals of consecutive ids, and the remaining residual gaps.
// It reaches prior nodes only through listAt(back) and depthAt(back), the list
// and reference depth of node x-back, so it does not require the full plane to be
// resident: a streaming encoder can serve those from a window of the last Window
// records. It returns the reference depth to record for node x. The two callers
// (the resident encodePlane and the windowed out-of-core encoder) share this body
// so their bitstreams are identical for the same lists.
func encodeNodeCore(w *bitWriter, x int, s []int32, listAt func(back int) []int32, depthAt func(back int) int, p Params) (depth int) {
	d := len(s)
	w.writeGamma(uint64(d))
	if d == 0 {
		return 0
	}

	// Reference: pick the node within the window whose list shares the most
	// elements with this one, if any, and copy those.
	r, copyBits := chooseReferenceCore(x, s, listAt, depthAt, p)
	w.writeGamma(uint64(r))
	var copied []int32
	if r > 0 {
		ref := listAt(r)
		runs := boolRuns(copyBits)
		w.writeGamma(uint64(len(runs)))
		for _, rl := range runs {
			w.writeGamma(uint64(rl))
		}
		copied = applyRuns(ref, runs)
		depth = depthAt(r) + 1
	}

	// What the copy did not cover.
	rem := minus(s, copied)

	// Intervals: maximal runs of consecutive ids at least LMin long.
	ivs, residual := splitIntervals(rem, p.LMin)
	w.writeGamma(uint64(len(ivs)))
	prevRight := 0
	for j, iv := range ivs {
		if j == 0 {
			w.writeSignedGamma(int64(iv.left) - int64(x))
		} else {
			w.writeGamma(uint64(iv.left - prevRight - 1))
		}
		w.writeGamma(uint64(iv.length - p.LMin))
		prevRight = iv.left + iv.length - 1
	}

	// Residuals: whatever is left, as zeta-coded gaps anchored at the node id.
	prev := 0
	for i, v := range residual {
		if i == 0 {
			w.writeSignedZeta(int64(v)-int64(x), p.ZetaK)
		} else {
			w.writeZeta(uint64(int(v)-prev-1), p.ZetaK)
		}
		prev = int(v)
	}
	return depth
}

// chooseReferenceCore finds the best node to copy from within the window, subject
// to the reference-chain cap. It scans back distances 1..Window (closest first,
// so the smallest back distance wins a tie under the strict comparison) through
// the listAt and depthAt accessors, and returns the back distance and the
// per-element copy mask over that node's list, or 0 and nil when no node in the
// window shares an element.
func chooseReferenceCore(x int, s []int32, listAt func(back int) []int32, depthAt func(back int) int, p Params) (int, []bool) {
	maxBack := p.Window
	if x < maxBack {
		maxBack = x
	}
	best := 0
	bestR := 0
	var bestMask []bool
	for back := 1; back <= maxBack; back++ {
		if depthAt(back) >= p.MaxRef {
			continue
		}
		mask, cnt := copyMask(listAt(back), s)
		if cnt > best {
			best = cnt
			bestR = back
			bestMask = mask
		}
	}
	if best == 0 {
		return 0, nil
	}
	return bestR, bestMask
}

// copyMask marks which elements of ref also appear in s, both sorted ascending.
func copyMask(ref, s []int32) ([]bool, int) {
	mask := make([]bool, len(ref))
	cnt := 0
	i, j := 0, 0
	for i < len(ref) && j < len(s) {
		switch {
		case ref[i] == s[j]:
			mask[i] = true
			cnt++
			i++
			j++
		case ref[i] < s[j]:
			i++
		default:
			j++
		}
	}
	return mask, cnt
}

// boolRuns turns a copy mask into alternating run lengths starting with the
// copied phase. The first run is zero when the mask starts false. The decoder
// replays the same alternation.
func boolRuns(mask []bool) []int {
	runs := []int{}
	phase := true // copied first
	count := 0
	for _, m := range mask {
		if m == phase {
			count++
			continue
		}
		runs = append(runs, count)
		phase = !phase
		count = 1
	}
	runs = append(runs, count)
	return runs
}

// applyRuns reconstructs the copied elements from a reference list and the runs.
func applyRuns(ref []int32, runs []int) []int32 {
	var out []int32
	idx := 0
	phase := true
	for _, rl := range runs {
		if phase {
			out = append(out, ref[idx:idx+rl]...)
		}
		idx += rl
		phase = !phase
	}
	return out
}

// minus returns the elements of s not in sub, both sorted, sub a subset of s.
func minus(s, sub []int32) []int32 {
	if len(sub) == 0 {
		return append([]int32(nil), s...)
	}
	out := make([]int32, 0, len(s)-len(sub))
	j := 0
	for _, v := range s {
		if j < len(sub) && sub[j] == v {
			j++
			continue
		}
		out = append(out, v)
	}
	return out
}

// splitIntervals separates rem into maximal consecutive runs of at least lMin
// (the intervals) and everything else (the residuals), both in order.
func splitIntervals(rem []int32, lMin int) ([]interval, []int32) {
	var ivs []interval
	var residual []int32
	i := 0
	for i < len(rem) {
		j := i + 1
		for j < len(rem) && rem[j] == rem[j-1]+1 {
			j++
		}
		if j-i >= lMin {
			ivs = append(ivs, interval{left: int(rem[i]), length: j - i})
		} else {
			residual = append(residual, rem[i:j]...)
		}
		i = j
	}
	return ivs, residual
}
