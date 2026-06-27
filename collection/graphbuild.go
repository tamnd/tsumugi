package collection

import (
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

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

// globalRanks computes collection-wide PageRank over the whole link graph and
// returns one rank per document, indexed by the document's position in docs.
//
// M15a found the web graph is almost entirely cross-shard: a broad crawl samples
// roughly one page per host, so within any single shard almost no link target is
// itself present and a per-shard PageRank sees an empty graph. The only place a
// real rank can be computed is over the whole collection at once, which is what
// this pass does.
//
// The build already buffers the whole collection and numbers documents globally
// by their host+url position, so the global node id of docs[i] is simply i. This
// pass builds the collection-wide canonical-URL directory, resolves every
// document's outbound links against it (the links M15a extracts but a single
// shard could not resolve), and runs the in-core power iteration over the
// resulting global graph. The rank lands in each shard's FeatPageRank column,
// which is the number the serving path reads; the adjacency itself never leaves
// the build.
//
// At collection scale the directory is a plain map and the graph is held in core,
// which matches the build's existing whole-collection buffering. The minimal
// perfect hash the spec specifies for the URL directory (a few bits a key instead
// of a full map) and the out-of-core iteration for a two-billion-document corpus
// are the same rank computed in less memory; they are a later milestone and
// change the cost of this pass, not its output.
func globalRanks(docs []convert.Document) []float64 {
	n := len(docs)
	if n == 0 {
		return nil
	}

	dir := buildDir(docs)

	gb := graph.NewBuilder(n)
	for i, d := range docs {
		for _, tgt := range analyze.Links(d) {
			// j != i drops a link that resolves back to the source, which happens
			// when two raw URLs in the collection share a canonical form and one
			// links to the other: they are the same page, so the edge is a self
			// loop and carries no rank.
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
	return graph.PageRank(g, graph.DefaultPRConfig())
}

// globalEdgeCount counts the edges the collection-wide directory resolves, the
// measure that distinguishes the cross-shard graph from the near-empty
// intra-shard one. It repeats globalRanks' resolution without the power iteration
// so a test can record the yield without depending on rank values.
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
