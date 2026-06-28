package collection

import (
	"sort"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

// inversePageRankCandidates caps how many top-out-reach nodes the build takes as
// automatic trust-seed candidates, the spec's "top few thousand by inverse
// PageRank" before the anti-trust filter and the curated union.
const inversePageRankCandidates = 2000

// graphSignals holds the collection-wide link signals, one value per document
// indexed by the document's position in docs (which is its global node id). The
// build bakes each into the matching feature column; serving reads the columns,
// never the graph.
type graphSignals struct {
	pageRank       []float64
	hostRank       []float64
	domainRank     []float64
	trust          []float64
	spamMass       []float64
	inDegree       []int
	linkingDomains []int
	linkingHosts   []int
	reciprocity    []float64
	hostLinkDiv    []float64
}

// slice returns the signals for the document range [lo, hi), the slice a single
// shard bakes into its feature matrix. The ranks are indexed by the same host+url
// order the shards are cut from, so each field's [lo:hi] lines up with docs[lo:hi].
func (s graphSignals) slice(lo, hi int) graphSignals {
	return graphSignals{
		pageRank:       s.pageRank[lo:hi],
		hostRank:       s.hostRank[lo:hi],
		domainRank:     s.domainRank[lo:hi],
		trust:          s.trust[lo:hi],
		spamMass:       s.spamMass[lo:hi],
		inDegree:       s.inDegree[lo:hi],
		linkingDomains: s.linkingDomains[lo:hi],
		linkingHosts:   s.linkingHosts[lo:hi],
		reciprocity:    s.reciprocity[lo:hi],
		hostLinkDiv:    s.hostLinkDiv[lo:hi],
	}
}

// buildDir builds the collection-wide canonical-URL to node-id directory over the
// documents. The node id of a document is its position in docs, the global id the
// build assigns in host+url order, and a canonical URL that several documents share
// keeps the first occurrence's id. The directory is the minimal perfect hash plus a
// membership fingerprint, the spec's few-bits-a-key replacement for a plain map, and
// it resolves a link target to its node id only when the target is a document the
// collection actually holds.
func buildDir(docs []convert.Document) *mph.Dir {
	urls := make([][]byte, 0, len(docs))
	ids := make([]uint32, 0, len(docs))
	for i, d := range docs {
		if cu, ok := analyze.CanonicalURL(d.URL); ok {
			urls = append(urls, []byte(cu))
			ids = append(ids, uint32(i))
		}
	}
	return mph.BuildDir(urls, ids, mph.DefaultGamma)
}

// buildGraph resolves every document's outbound links against the collection-wide
// directory and returns the global link graph. A link whose target is a document in
// the collection becomes an edge; a link to a page the crawl never captured is
// dropped, which on a breadth-first crawl is almost all of them. A self loop (two
// raw URLs sharing a canonical form, one linking the other: the same page) carries
// no rank and is dropped.
func buildGraph(docs []convert.Document, dir *mph.Dir) *graph.Region {
	gb := graph.NewBuilder(len(docs))
	for i, d := range docs {
		for _, tgt := range analyze.Links(d) {
			if j, ok := dir.Lookup([]byte(tgt)); ok && int(j) != i {
				gb.AddEdge(i, int(j))
			}
		}
	}
	g, err := graph.Open(gb.Build())
	if err != nil {
		// NewBuilder().Build() always produces a region graph.Open accepts; a
		// failure here is a programming error in the graph package, not a data
		// condition the build can recover from.
		panic(err)
	}
	return g
}

// groupings maps each document to a dense host id and a dense domain id, the
// grouping keys the host-rank, domain-rank, and distinct-linking-domain signals
// bucket pages into. The host is the document's host (parsed from its URL when the
// crawl left it blank); the domain is the registered domain (eTLD+1), so pages
// under one domain a farm controls collapse to a single voting unit.
func groupings(docs []convert.Document) (hostOf, domainOf []int) {
	hostOf = make([]int, len(docs))
	domainOf = make([]int, len(docs))
	hostID := map[string]int{}
	domID := map[string]int{}
	for i, d := range docs {
		h := d.Host
		if h == "" {
			h = analyze.HostOf(d.URL)
		}
		id, ok := hostID[h]
		if !ok {
			id = len(hostID)
			hostID[h] = id
		}
		hostOf[i] = id

		dom := analyze.RegisteredDomain(h)
		did, ok := domID[dom]
		if !ok {
			did = len(domID)
			domID[dom] = did
		}
		domainOf[i] = did
	}
	return hostOf, domainOf
}

// resolveSeeds turns a curated list of seed URLs into node ids, dropping any whose
// canonical form is not a document in the collection. The order follows the input
// so the seed set is reproducible from the build's recorded seed list.
func resolveSeeds(curated []string, dir *mph.Dir) []int {
	var seeds []int
	seen := map[int]bool{}
	for _, raw := range curated {
		cu, ok := analyze.CanonicalURL(raw)
		if !ok {
			continue
		}
		if id, ok := dir.Lookup([]byte(cu)); ok && !seen[int(id)] {
			seen[int(id)] = true
			seeds = append(seeds, int(id))
		}
	}
	return seeds
}

// trustSeedSet builds the TrustRank seed set the spec's doc 07 describes: the
// curated list as the spine, extended with the top nodes by inverse PageRank (the
// best-connected hubs by out-reach), dropping any candidate whose anti-trust score
// is more than twice the uniform baseline, because a page that points at spam
// should not anchor trust however well-connected it is. With no spam seeds the
// anti-trust vector is uniform and the filter is inert, so a fresh build with no
// curated lists seeds purely from inverse PageRank.
func trustSeedSet(g *graph.Region, dir *mph.Dir, curated []string, anti []float64, cfg graph.PRConfig) []int {
	seeds := resolveSeeds(curated, dir)
	seen := map[int]bool{}
	for _, s := range seeds {
		seen[s] = true
	}

	n := g.NodeCount()
	if n == 0 {
		return seeds
	}
	inv := graph.InversePageRank(g, cfg)
	cand := topKIndices(inv, inversePageRankCandidates)
	drop := 2.0 / float64(n) // twice the uniform anti-trust baseline
	for _, v := range cand {
		if seen[v] {
			continue
		}
		if anti != nil && anti[v] > drop {
			continue
		}
		seen[v] = true
		seeds = append(seeds, v)
	}
	sort.Ints(seeds)
	return seeds
}

// topKIndices returns the indices of the k largest values, ties broken by index so
// the selection is deterministic across builds.
func topKIndices(vals []float64, k int) []int {
	idx := make([]int, len(vals))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if vals[idx[a]] != vals[idx[b]] {
			return vals[idx[a]] > vals[idx[b]]
		}
		return idx[a] < idx[b]
	})
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

// collectionOrder computes the dense docID ordering doc 06 specifies over the
// whole collection: host and domain grouping first, then Recursive Graph
// Bisection over the resolved link graph. It returns order, where order[i] is the
// index in docs that belongs at new dense id i.
//
// The build permutes docs by this order before cutting shards, so a document's
// dense docID is its position in the order, the single ordering the graph
// compression, the identity assignment, and the posting impact locality all
// share. The documents reach this function already sorted by host then url, which
// is the host-grouping first pass (a host's pages are contiguous), so the
// grouping sort is a stable no-op here and Recursive Graph Bisection spends the
// leftover freedom: it clusters pages that share out-neighbors, which is what
// collapses the adjacency gaps the codec spends bits on.
//
// PageRank and every other signal is a property of the graph, not its labeling,
// so permuting the documents and recomputing the signals over the permuted slice
// leaves each document's signals unchanged; the reorder changes the ids, not the
// numbers. On a breadth-first crawl the graph barely materializes, so the order
// is close to the input host+url order; it earns its bits on a crawl with depth,
// where a host's pages share out-neighbors and cluster.
func collectionOrder(docs []convert.Document) []int {
	n := len(docs)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if n <= 1 {
		return order
	}
	dir := buildDir(docs)
	g := buildGraph(docs, dir)
	out := graph.OutLists(g)
	hostOf, _ := groupings(docs)
	return graph.Reorder(out, hostOf, graph.DefaultBPConfig())
}

// globalSignals computes every collection-wide link signal and returns one value
// per document, indexed by the document's position in docs.
//
// M15a found the web graph is almost entirely cross-shard: a broad crawl samples
// roughly one page per host, so within any single shard almost no link target is
// itself present and a per-shard signal sees an empty graph. The only place a real
// signal can be computed is over the whole collection at once, which is what this
// pass does. The build already buffers and globally numbers the whole collection,
// so docs[i]'s global node id is simply i.
//
// PageRank and the seed-biased ranks (TrustRank, SpamMass) and the aggregate ranks
// (host, domain) and the link counts (in-degree, distinct linking domains) all run
// here off the one resolved graph; each lands in its feature column, the number
// serving reads. The seeds follow the spec: curated lists are an input recorded
// with the build, extended automatically by inverse PageRank and filtered by
// anti-trust.
func globalSignals(docs []convert.Document, trustSeeds, spamSeeds []string) graphSignals {
	n := len(docs)
	if n == 0 {
		return graphSignals{}
	}
	dir := buildDir(docs)
	g := buildGraph(docs, dir)
	cfg := graph.DefaultPRConfig()
	hostOf, domainOf := groupings(docs)

	pr := graph.PageRank(g, cfg)
	spam := resolveSeeds(spamSeeds, dir)
	anti := graph.AntiTrustRank(g, spam, cfg)
	trust := trustSeedSet(g, dir, trustSeeds, anti, cfg)
	tr := graph.TrustRank(g, trust, cfg)

	return graphSignals{
		pageRank:       pr,
		hostRank:       graph.HostRank(g, hostOf, cfg),
		domainRank:     graph.DomainRank(g, domainOf, cfg),
		trust:          tr,
		spamMass:       graph.SpamMass(pr, tr, trust),
		inDegree:       graph.InDegrees(g),
		linkingDomains: graph.LinkingDomains(g, domainOf),
		linkingHosts:   graph.LinkingHosts(g, hostOf),
		reciprocity:    graph.Reciprocity(g),
		hostLinkDiv:    graph.HostLinkDiversity(g, hostOf),
	}
}

// globalRanks computes collection-wide PageRank over the whole link graph and
// returns one rank per document. It is the PageRank slice of globalSignals, kept as
// a focused entry point for the rank-only tests.
func globalRanks(docs []convert.Document) []float64 {
	n := len(docs)
	if n == 0 {
		return nil
	}
	dir := buildDir(docs)
	g := buildGraph(docs, dir)
	return graph.PageRank(g, graph.DefaultPRConfig())
}

// globalEdgeCount counts the edges the collection-wide directory resolves, the
// measure that distinguishes the cross-shard graph from the near-empty
// intra-shard one. It repeats the resolution without the power iteration so a test
// can record the yield without depending on rank values.
func globalEdgeCount(docs []convert.Document) int {
	dir := buildDir(docs)
	var edges int
	for i, d := range docs {
		for _, tgt := range analyze.Links(d) {
			if j, ok := dir.Lookup([]byte(tgt)); ok && int(j) != i {
				edges++
			}
		}
	}
	return edges
}
