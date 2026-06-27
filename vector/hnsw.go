package vector

import (
	"math"
	"math/rand"
	"sort"
)

// hnswGraph is the in-memory navigable small-world graph. The build distance is
// the int8 dot over the scalar-quantized rotated vectors, an integer kernel that
// tracks the exact dot to within a fraction of a percent yet is far cheaper to
// evaluate the millions of times a build needs, so the edges connect true
// neighbors and the search needs only a narrow beam. It is the same metric the
// two-part search navigates with, so build and query agree. The int8 rows are
// held for the build and dropped after; only the links persist. Each node keeps a
// neighbor list per layer it reaches.
type hnswGraph struct {
	m, m0    int
	rows     [][]int8  // int8 rotated rows, build distance only
	links    [][]int32 // links[node] holds layer-0 neighbors
	up       []map[int][]int32
	entry    int32
	maxLayer int
	ml       float64
	rng      *rand.Rand
}

func newHNSW(rows [][]int8, m, m0 int, efConstruction int, seed int64) *hnswGraph {
	g := &hnswGraph{
		m:        m,
		m0:       m0,
		rows:     rows,
		links:    make([][]int32, len(rows)),
		up:       make([]map[int][]int32, len(rows)),
		entry:    -1,
		maxLayer: 0,
		ml:       1 / math.Log(float64(m)),
		rng:      rand.New(rand.NewSource(seed)),
	}
	for i := range rows {
		g.insert(int32(i), efConstruction)
	}
	return g
}

// dist is the build metric: smaller is nearer, so it returns the negated int8 dot
// of the two rows.
func (g *hnswGraph) dist(a, b int32) float64 {
	return -float64(dotI8(g.rows[a], g.rows[b]))
}

func (g *hnswGraph) maxAt(layer int) int {
	if layer == 0 {
		return g.m0
	}
	return g.m
}

func (g *hnswGraph) neighbors(node int32, layer int) []int32 {
	if layer == 0 {
		return g.links[node]
	}
	if g.up[node] == nil {
		return nil
	}
	return g.up[node][layer]
}

func (g *hnswGraph) setNeighbors(node int32, layer int, ns []int32) {
	if layer == 0 {
		g.links[node] = ns
		return
	}
	if g.up[node] == nil {
		g.up[node] = map[int][]int32{}
	}
	g.up[node][layer] = ns
}

func (g *hnswGraph) assignLayer() int {
	return int(-math.Log(g.rng.Float64()+1e-12) * g.ml)
}

func (g *hnswGraph) insert(node int32, ef int) {
	if g.entry < 0 {
		g.entry = node
		g.maxLayer = 0
		return
	}
	l := g.assignLayer()
	ep := g.entry
	// Phase 1: greedy descent through the layers above l.
	for layer := g.maxLayer; layer > l; layer-- {
		ep = g.greedy(node, ep, layer)
	}
	// Phase 2: beam search and connect on layers min(l, maxLayer) down to 0.
	start := l
	if start > g.maxLayer {
		start = g.maxLayer
	}
	for layer := start; layer >= 0; layer-- {
		w := g.beam(node, ep, layer, ef)
		nbrs := g.selectDiverse(node, w, g.maxAt(layer))
		g.setNeighbors(node, layer, nbrs)
		for _, nb := range nbrs {
			g.addLink(nb, node, layer)
		}
		if len(w) > 0 {
			ep = w[0].id
		}
	}
	if l > g.maxLayer {
		g.maxLayer = l
		g.entry = node
	}
}

func (g *hnswGraph) addLink(node, other int32, layer int) {
	cur := g.neighbors(node, layer)
	for _, x := range cur {
		if x == other {
			return
		}
	}
	cur = append(cur, other)
	if len(cur) > g.maxAt(layer) {
		// Over budget: re-run the diversity heuristic on the full set.
		cs := make([]cand, len(cur))
		for i, c := range cur {
			cs[i] = cand{id: c, d: g.dist(node, c)}
		}
		cur = g.selectDiverse(node, cs, g.maxAt(layer))
	}
	g.setNeighbors(node, layer, cur)
}

func (g *hnswGraph) greedy(q, ep int32, layer int) int32 {
	cur := ep
	curD := g.dist(q, cur)
	for {
		improved := false
		for _, nb := range g.neighbors(cur, layer) {
			d := g.dist(q, nb)
			if d < curD {
				cur, curD = nb, d
				improved = true
			}
		}
		if !improved {
			return cur
		}
	}
}

// cand is a (node, distance) pair used in the beam and the diversity heuristic.
// Distance is the build/search metric where smaller is nearer.
type cand struct {
	id int32
	d  float64
}

// beam is the layer-0 (and upper-layer) beam search: expand the closest
// unexpanded node, keep the best ef seen, stop when the frontier cannot improve.
// It returns the result set sorted nearest first.
func (g *hnswGraph) beam(q, ep int32, layer, ef int) []cand {
	visited := map[int32]bool{ep: true}
	d0 := g.dist(q, ep)
	candHeap := &minHeap{{ep, d0}}
	resHeap := &maxHeap{{ep, d0}}
	for candHeap.Len() > 0 {
		c := candHeap.popMin()
		if resHeap.Len() >= ef && c.d > (*resHeap)[0].d {
			break
		}
		for _, nb := range g.neighbors(c.id, layer) {
			if visited[nb] {
				continue
			}
			visited[nb] = true
			d := g.dist(q, nb)
			if resHeap.Len() < ef || d < (*resHeap)[0].d {
				candHeap.pushItem(cand{nb, d})
				resHeap.pushItem(cand{nb, d})
				if resHeap.Len() > ef {
					resHeap.popMax()
				}
			}
		}
	}
	out := make([]cand, resHeap.Len())
	copy(out, *resHeap)
	sort.Slice(out, func(i, j int) bool {
		if out[i].d != out[j].d {
			return out[i].d < out[j].d
		}
		return out[i].id < out[j].id
	})
	return out
}

// selectDiverse keeps a spread of neighbors rather than the m closest: a
// candidate is kept only if it is closer to the new node than to every neighbor
// already kept, which preserves the longer-range links the greedy walk escapes
// along. Input need not be sorted; it is sorted here by distance ascending.
func (g *hnswGraph) selectDiverse(node int32, w []cand, m int) []int32 {
	sort.Slice(w, func(i, j int) bool {
		if w[i].d != w[j].d {
			return w[i].d < w[j].d
		}
		return w[i].id < w[j].id
	})
	kept := make([]int32, 0, m)
	for _, c := range w {
		if c.id == node {
			continue
		}
		if len(kept) == m {
			break
		}
		ok := true
		for _, k := range kept {
			if g.dist(c.id, k) < c.d {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, c.id)
		}
	}
	return kept
}
