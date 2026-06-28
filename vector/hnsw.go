package vector

import (
	"math"
	"math/rand"
	"sort"
)

// hnswGraph is the in-memory navigable small-world graph. The build distance is a
// pluggable kernel over the node indices (distFn): the spec builds over the symmetric
// one-bit Hamming popcount, an order of magnitude cheaper than a float dot and good
// enough because the graph only has to get the search into the right neighborhood, with
// the rerank fixing the final order (05 line 402, 406). The int8 dot is the alternative
// when a measurement shows the popcount build loses too much recall on a corpus. Build
// and query navigate with the same metric so they agree. Whatever data the distance reads
// (one-bit codes or int8 rows) is held for the build and dropped after; only the links
// persist. Each node keeps a neighbor list per layer it reaches.
type hnswGraph struct {
	m, m0    int
	n        int
	distFn   func(a, b int32) float64
	links    [][]int32 // links[node] holds layer-0 neighbors
	up       []map[int][]int32
	entry    int32
	maxLayer int
	ml       float64
	rng      *rand.Rand
}

// newHNSWDist builds a graph of n nodes over an arbitrary node-to-node distance, smaller
// nearer. It is the general constructor the builder calls with the chosen build metric.
func newHNSWDist(n int, distFn func(a, b int32) float64, m, m0, efConstruction int, seed int64) *hnswGraph {
	g := &hnswGraph{
		m:        m,
		m0:       m0,
		n:        n,
		distFn:   distFn,
		links:    make([][]int32, n),
		up:       make([]map[int][]int32, n),
		entry:    -1,
		maxLayer: 0,
		ml:       1 / math.Log(float64(m)),
		rng:      rand.New(rand.NewSource(seed)),
	}
	for i := 0; i < n; i++ {
		g.insert(int32(i), efConstruction)
	}
	return g
}

// newHNSW builds a graph over the int8 dot of the given rows, the convenience form the
// tests use. The negated dot makes smaller nearer.
func newHNSW(rows [][]int8, m, m0 int, efConstruction int, seed int64) *hnswGraph {
	return newHNSWDist(len(rows), func(a, b int32) float64 {
		return -float64(dotI8(rows[a], rows[b]))
	}, m, m0, efConstruction, seed)
}

// reachableCount returns how many nodes the search can reach from the entry point
// over the layer-0 links. The walk starts at the entry and beams over layer-0, so a
// node no layer-0 path reaches is invisible to dense search, a silent recall hole.
// The diversity shrink can drop a back-link, so connectivity is a property to verify
// after the build, not one the insert order guarantees. It is a directed reach over
// g.links because that is exactly the edge set neighbors0 walks at query time.
func (g *hnswGraph) reachableCount() int {
	n := len(g.links)
	if n == 0 || g.entry < 0 {
		return n
	}
	seen := make([]bool, n)
	seen[g.entry] = true
	count := 1
	stack := []int32{g.entry}
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, nb := range g.links[cur] {
			if !seen[nb] {
				seen[nb] = true
				count++
				stack = append(stack, nb)
			}
		}
	}
	return count
}

// repair grafts every node the search cannot reach from the entry back into the
// reachable graph and returns the reachable count after the pass. On
// near-duplicate-heavy corpora the diversity heuristic seals a tight cluster into an
// island with no in-edge from the rest of the graph: the cluster's members keep only
// each other as neighbors and every outside node drops them, so the entry walk never
// reaches them and they are invisible to dense search. A web corpus is full of such
// near-duplicates, so this is the common case, not a corner one, and erroring out
// would refuse to index real data. Instead each orphan is given one layer-0 in-edge
// from a reachable node near it, found by a greedy walk from the entry so the new
// edge is short and the graft is local. The graft prefers a reachable node that still
// has a free layer-0 slot so no existing edge is dropped; only when no reachable node
// anywhere has a free slot does it replace a grafter's farthest neighbor, the one the
// search is least likely to follow. Grafting an orphan pulls in everything reachable
// from it, so the reachable set is grown incrementally as the pass runs and a single
// pass reconnects the whole graph in the no-eviction case; the caller repeats until
// the reachable count stops rising to absorb the rare eviction that re-orphans a node.
func (g *hnswGraph) repair() int {
	n := len(g.links)
	if n == 0 || g.entry < 0 {
		return n
	}
	seen := make([]bool, n)
	seen[g.entry] = true
	reached := 1
	stack := []int32{g.entry}
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, nb := range g.links[cur] {
			if !seen[nb] {
				seen[nb] = true
				reached++
				stack = append(stack, nb)
			}
		}
	}
	if reached == n {
		return n
	}
	// spare holds reachable nodes that still have a free layer-0 slot, the graft
	// points that need no eviction. It is grown as newly reached nodes turn up.
	var spare []int32
	for i := 0; i < n; i++ {
		if seen[i] && len(g.links[i]) < g.m0 {
			spare = append(spare, int32(i))
		}
	}
	// absorb marks start and everything reachable from it as seen, feeding the spare
	// pool with any newly reached node that has a free slot.
	absorb := func(start int32) {
		ls := []int32{start}
		for len(ls) > 0 {
			c := ls[len(ls)-1]
			ls = ls[:len(ls)-1]
			if len(g.links[c]) < g.m0 {
				spare = append(spare, c)
			}
			for _, nb := range g.links[c] {
				if !seen[nb] {
					seen[nb] = true
					reached++
					ls = append(ls, nb)
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		if seen[i] {
			continue
		}
		u := int32(i)
		// Prefer a reachable node near u; fall back to any reachable node with a free
		// slot; evict only when nothing has room.
		v := g.greedy(u, g.entry, 0)
		if len(g.links[v]) >= g.m0 {
			for len(spare) > 0 {
				c := spare[len(spare)-1]
				if seen[c] && len(g.links[c]) < g.m0 {
					v = c
					break
				}
				spare = spare[:len(spare)-1]
			}
		}
		if len(g.links[v]) < g.m0 {
			g.links[v] = append(g.links[v], u)
		} else {
			v = g.greedy(u, g.entry, 0)
			cur := g.links[v]
			worst, wd := 0, g.dist(v, cur[0])
			for j := 1; j < len(cur); j++ {
				if d := g.dist(v, cur[j]); d > wd {
					worst, wd = j, d
				}
			}
			cur[worst] = u
		}
		seen[u] = true
		reached++
		absorb(u)
	}
	return reached
}

// dist is the build metric: smaller is nearer. It delegates to the pluggable kernel the
// graph was built with.
func (g *hnswGraph) dist(a, b int32) float64 {
	return g.distFn(a, b)
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
