package graph

import (
	"math"
	"sort"
)

// This file computes the M5 link signals over a graph region: PageRank and its
// host and domain aggregates, TrustRank, the spam-mass gap, and the link-count
// signals. Each is a query-independent per-document number that M3 quantizes into
// a feature column. The link-analysis signals are power iterations over the
// transpose; the link-count signals are single passes. None of them mutate the
// region.

// PRConfig tunes the PageRank power iteration.
type PRConfig struct {
	Alpha    float64 // damping, the follow-a-link probability, default 0.85
	MaxIters int     // iteration cap, default 100
	Tol      float64 // L1 convergence tolerance, default 1e-6
}

// DefaultPRConfig returns the canonical PageRank settings. At alpha 0.85 the
// error shrinks by 0.85 an iteration, so a hundred iterations is far past any
// useful precision and the loop almost always stops early on the tolerance.
func DefaultPRConfig() PRConfig {
	return PRConfig{Alpha: 0.85, MaxIters: 100, Tol: 1e-6}
}

// csr is the transpose materialized as compressed sparse rows: in[i] for node v
// is cols[rowPtr[v]:rowPtr[v+1]]. The power iterations read this in-core. The
// spec frames PageRank as streaming the on-disk transpose with the rank vectors
// resident in RAM; for a single shard the whole transpose fits, so it is decoded
// once into CSR and the iterations run over flat arrays instead of re-decoding
// every list every pass.
type csr struct {
	rowPtr   []int
	cols     []int32
	outdeg   []float64
	dangling []int
}

func buildCSR(g *Region) *csr {
	return buildCSRDir(g, g.InNeighbors, g.OutDegree)
}

// buildCSRRev materializes the reversed graph: a node's in-neighbors on the
// reverse are its out-neighbors on the original, and its reverse out-degree is its
// original in-degree. Inverse PageRank and Anti-TrustRank are the same forward
// iteration run over this transpose-of-the-transpose.
func buildCSRRev(g *Region) *csr {
	return buildCSRDir(g, g.OutNeighbors, g.InDegree)
}

// buildCSRDir materializes one direction of the graph as compressed sparse rows:
// inOf gives the in-neighbors a node sums rank from and degOf the out-degree the
// sender divides its rank by.
func buildCSRDir(g *Region, inOf func(int) []int, degOf func(int) int) *csr {
	n := g.nodeCount
	c := &csr{rowPtr: make([]int, n+1), outdeg: make([]float64, n)}
	total := 0
	rows := make([][]int, n)
	for v := 0; v < n; v++ {
		rows[v] = inOf(v)
		total += len(rows[v])
		d := degOf(v)
		c.outdeg[v] = float64(d)
		if d == 0 {
			c.dangling = append(c.dangling, v)
		}
	}
	c.cols = make([]int32, 0, total)
	for v := 0; v < n; v++ {
		c.rowPtr[v] = len(c.cols)
		for _, u := range rows[v] {
			c.cols = append(c.cols, int32(u))
		}
	}
	c.rowPtr[n] = len(c.cols)
	return c
}

func (c *csr) inNeighbors(v int) []int32 { return c.cols[c.rowPtr[v]:c.rowPtr[v+1]] }

// powerIter runs the random-surfer power iteration over c with personalization
// vector p, which sums to one. Both the teleport and the dangling-mass
// redistribution restart on p, so a uniform p gives PageRank and a p biased onto a
// seed set gives the seed-biased variants. The result sums to one.
func powerIter(c *csr, n int, p []float64, cfg PRConfig) []float64 {
	r := make([]float64, n)
	rn := make([]float64, n)
	copy(r, p)
	for it := 0; it < cfg.MaxIters; it++ {
		var dm float64
		for _, x := range c.dangling {
			dm += r[x]
		}
		for v := 0; v < n; v++ {
			var acc float64
			for _, u := range c.inNeighbors(v) {
				acc += r[u] / c.outdeg[u]
			}
			rn[v] = (1-cfg.Alpha)*p[v] + cfg.Alpha*(acc+dm*p[v])
		}
		var delta float64
		for i := range r {
			delta += math.Abs(rn[i] - r[i])
		}
		r, rn = rn, r
		if delta < cfg.Tol {
			break
		}
	}
	return r
}

// uniformVec is the teleport vector of plain PageRank: every node equally likely.
func uniformVec(n int) []float64 {
	p := make([]float64, n)
	u := 1.0 / float64(n)
	for i := range p {
		p[i] = u
	}
	return p
}

// seedVec is the personalization vector biased onto seeds, each seed sharing the
// mass equally and every other node zero. With no seeds it falls back to uniform,
// so a seed-biased rank with an empty seed set degenerates to plain PageRank.
func seedVec(n int, seeds []int) []float64 {
	if len(seeds) == 0 {
		return uniformVec(n)
	}
	p := make([]float64, n)
	w := 1.0 / float64(len(seeds))
	for _, s := range seeds {
		if s >= 0 && s < n {
			p[s] = w
		}
	}
	return p
}

// PageRank runs the random-surfer power iteration and returns the stationary
// distribution, one rank a node summing to one. Dangling nodes, the ones with no
// out-links, have their mass redistributed uniformly each step so the vector does
// not leak; on a real crawl the dangling set is a large minority and skipping
// this would bleed rank every iteration.
func PageRank(g *Region, cfg PRConfig) []float64 {
	n := g.nodeCount
	if n == 0 {
		return nil
	}
	return powerIter(buildCSR(g), n, uniformVec(n), cfg)
}

// InversePageRank runs PageRank on the reversed graph, ranking a node by how well
// it reaches the rest of the graph rather than how well the graph reaches it. The
// best-connected hubs by out-reach are the trust-seed candidates the build filters
// down, the automatic half of the seed set the spec's doc 07 describes.
func InversePageRank(g *Region, cfg PRConfig) []float64 {
	n := g.nodeCount
	if n == 0 {
		return nil
	}
	return powerIter(buildCSRRev(g), n, uniformVec(n), cfg)
}

// TrustRank is PageRank with the teleport biased onto a trusted seed set: trust
// starts on the seeds and flows forward along links, on the premise that good
// pages rarely link to spam. Same iteration, same dangling handling, only the
// teleport vector differs. With no seeds it degenerates to uniform PageRank.
func TrustRank(g *Region, seeds []int, cfg PRConfig) []float64 {
	n := g.nodeCount
	if n == 0 {
		return nil
	}
	return powerIter(buildCSR(g), n, seedVec(n, seeds), cfg)
}

// AntiTrustRank propagates distrust backward from a spam seed set: it is TrustRank
// on the reversed graph, so distrust flows from a spam page to the pages that link
// to it. A high score is a direct spam flag, and a non-trivial score on a
// trust-seed candidate is reason to drop it from the trust seeds. With no spam
// seeds it degenerates to uniform.
func AntiTrustRank(g *Region, spamSeeds []int, cfg PRConfig) []float64 {
	n := g.nodeCount
	if n == 0 {
		return nil
	}
	return powerIter(buildCSRRev(g), n, seedVec(n, spamSeeds), cfg)
}

// SpamMass returns the relative spam mass per node, 1 - trust/pagerank after
// rescaling trust so its median ratio over the seed set is one. A legitimate page
// draws its rank from trusted sources so trust explains it and the mass is near
// zero; a spam page draws its rank from an untrusted farm so trust falls short
// and the mass approaches one. Values are clamped to [0, 1].
func SpamMass(pr, trust []float64, seeds []int) []float64 {
	n := len(pr)
	out := make([]float64, n)
	c := trustScale(pr, trust, seeds)
	for i := 0; i < n; i++ {
		if pr[i] <= 0 {
			out[i] = 0
			continue
		}
		m := 1 - c*trust[i]/pr[i]
		if m < 0 {
			m = 0
		} else if m > 1 {
			m = 1
		}
		out[i] = m
	}
	return out
}

// trustScale finds the constant that makes the median trust/pagerank ratio over
// the seeds equal one; the seeds are trusted by definition, so their trust should
// fully explain their rank.
func trustScale(pr, trust []float64, seeds []int) float64 {
	if len(seeds) == 0 {
		return 1
	}
	ratios := make([]float64, 0, len(seeds))
	for _, s := range seeds {
		if s >= 0 && s < len(pr) && pr[s] > 0 && trust[s] > 0 {
			ratios = append(ratios, trust[s]/pr[s])
		}
	}
	if len(ratios) == 0 {
		return 1
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	if med <= 0 {
		return 1
	}
	return 1 / med
}

// InDegrees returns the in-degree of every node, a single pass over the
// transpose offset index. It is the crudest authority count and easily gamed; the
// distinct-linking-domain count below is the robust version.
func InDegrees(g *Region) []int {
	out := make([]int, g.nodeCount)
	for x := 0; x < g.nodeCount; x++ {
		out[x] = g.InDegree(x)
	}
	return out
}

// LinkingDomains returns, per node, the number of distinct domains among its
// in-neighbors. domainOf maps a dense node id to a domain id. A farm can
// manufacture many links from a few domains it controls but cannot cheaply
// manufacture links from many independent domains, so this resists link spam far
// better than raw in-degree.
func LinkingDomains(g *Region, domainOf []int) []int {
	out := make([]int, g.nodeCount)
	for v := 0; v < g.nodeCount; v++ {
		seen := map[int]struct{}{}
		for _, u := range g.InNeighbors(v) {
			if u < len(domainOf) {
				seen[domainOf[u]] = struct{}{}
			}
		}
		out[v] = len(seen)
	}
	return out
}

// HostRank aggregates the page graph up to hosts, runs a weighted PageRank on
// the host graph dropping intra-host edges, and gives each page its host's rank.
// Dropping the internal links is what makes host rank resist the internal-link
// spam that hurts page rank. hostOf maps a dense node id to a host id; the host
// ids need not be dense, they are compacted here.
func HostRank(g *Region, hostOf []int, cfg PRConfig) []float64 {
	return aggregateRank(g, hostOf, cfg)
}

// DomainRank is HostRank over domains: aggregate to domains, drop intra-domain
// edges, weighted PageRank, inherit. It has the least resolution and the least
// noise of the three authority signals.
func DomainRank(g *Region, domainOf []int, cfg PRConfig) []float64 {
	return aggregateRank(g, domainOf, cfg)
}

// aggregateRank is the shared body of host and domain rank: project the page
// edges onto the group graph with inter-group weights, drop intra-group edges,
// run a weighted power iteration, and inherit the group rank back to each page.
func aggregateRank(g *Region, groupOf []int, cfg PRConfig) []float64 {
	n := g.nodeCount
	// Compact group ids to a dense range.
	idOf := map[int]int{}
	for x := 0; x < n; x++ {
		gid := groupOf[x]
		if _, ok := idOf[gid]; !ok {
			idOf[gid] = len(idOf)
		}
	}
	m := len(idOf)
	if m == 0 {
		return make([]float64, n)
	}

	// Weighted inter-group edges, accumulated from the page edges.
	type key struct{ u, v int }
	w := map[key]float64{}
	for u := 0; u < n; u++ {
		gu := idOf[groupOf[u]]
		for _, v := range g.OutNeighbors(u) {
			gv := idOf[groupOf[v]]
			if gu == gv {
				continue
			}
			w[key{gu, gv}]++
		}
	}

	// Group in-edges and out-weights for the weighted iteration.
	inEdges := make([][]wedge, m)
	outW := make([]float64, m)
	for k, weight := range w {
		inEdges[k.v] = append(inEdges[k.v], wedge{u: k.u, w: weight})
		outW[k.u] += weight
	}

	gr := weightedPageRank(m, inEdges, outW, cfg)
	out := make([]float64, n)
	for x := 0; x < n; x++ {
		out[x] = gr[idOf[groupOf[x]]]
	}
	return out
}

type wedge struct {
	u int
	w float64
}

// weightedPageRank is the power iteration on a small weighted graph held fully in
// memory, used for the host and domain graphs.
func weightedPageRank(n int, inEdges [][]wedge, outW []float64, cfg PRConfig) []float64 {
	if n == 0 {
		return nil
	}
	dangling := make([]int, 0)
	for u := 0; u < n; u++ {
		if outW[u] == 0 {
			dangling = append(dangling, u)
		}
	}
	r := make([]float64, n)
	rn := make([]float64, n)
	for i := range r {
		r[i] = 1.0 / float64(n)
	}
	teleport := (1 - cfg.Alpha) / float64(n)
	for it := 0; it < cfg.MaxIters; it++ {
		var dm float64
		for _, x := range dangling {
			dm += r[x]
		}
		base := teleport + cfg.Alpha*dm/float64(n)
		for v := 0; v < n; v++ {
			var acc float64
			for _, e := range inEdges[v] {
				acc += r[e.u] * e.w / outW[e.u]
			}
			rn[v] = base + cfg.Alpha*acc
		}
		var delta float64
		for i := range r {
			delta += math.Abs(rn[i] - r[i])
		}
		r, rn = rn, r
		if delta < cfg.Tol {
			break
		}
	}
	return r
}
