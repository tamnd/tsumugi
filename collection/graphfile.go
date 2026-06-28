package collection

import (
	"path/filepath"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/graph"
)

// graphFileName is the collection-level cross-shard graph artifact, a sibling of
// the shards in the collection directory. It holds the whole collection's link
// graph over the global dense id space [0, N), the transpose plane the out-of-core
// StreamPageRank streams one in-list at a time at rank time. It is a single-region
// .tsumugi container so it reuses the framing, CRC, and memory mapping the shards
// already have; the graph region is stored uncompressed (CodecNone) so the reader's
// region bytes alias the mapping and the adjacency stays on disk instead of being
// decompressed into the heap.
const graphFileName = "graph.tsumugi"

// graphFilePath returns the collection graph artifact's path inside a collection
// directory.
func graphFilePath(dir string) string { return filepath.Join(dir, graphFileName) }

// writeCollectionGraph writes the encoded collection-wide graph region as the
// collection artifact. nodeCount is the global node space size N (the collection
// document count), recorded as the container's doc count; the node base is zero
// because the artifact spans the whole collection from global id zero. The region
// is stored CodecNone so OpenGraphSource maps it zero-copy.
func writeCollectionGraph(dir string, region []byte, nodeCount int) error {
	w, err := tsumugi.Create(graphFilePath(dir))
	if err != nil {
		return err
	}
	w.SetDocCount(uint32(nodeCount))
	w.SetNodeBase(0)
	if err := w.AddRegion(tsumugi.RegionGraph, tsumugi.CodecNone, 0, 0, region); err != nil {
		return err
	}
	return w.Close()
}

// GraphSource is the disk-backed, memory-mapped InNeighborSource over the
// collection graph artifact. It maps graph.tsumugi, parses the graph region in
// place, and forwards NodeCount/InNeighbors/OutDegree to the parsed region, which
// decodes one neighbor list per call and holds none of the adjacency resident, so
// the streaming PageRank's working set is its rank vectors, not the graph. It
// satisfies graph.InNeighborSource.
type GraphSource struct {
	r      *tsumugi.Reader
	region *graph.Region
}

// OpenGraphSource maps the collection graph artifact and returns a streaming
// source. The only resident state is the mapping handle and the parsed region
// header and offset indexes; in-lists are decoded on demand from the mapping.
func OpenGraphSource(dir string) (*GraphSource, error) {
	r, err := tsumugi.Open(graphFilePath(dir))
	if err != nil {
		return nil, err
	}
	b, err := r.Region(tsumugi.RegionGraph)
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	g, err := graph.Open(b)
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	return &GraphSource{r: r, region: g}, nil
}

// NodeCount returns the global node space size N.
func (s *GraphSource) NodeCount() int { return s.region.NodeCount() }

// InNeighbors decodes a node's in-list from the mapped transpose plane on demand.
func (s *GraphSource) InNeighbors(v int) []int { return s.region.InNeighbors(v) }

// OutDegree reads a node's out-degree code without expanding its out-list, the
// per-node value graph.OutDegreesFromSource gathers into the resident out-degree
// array StreamPageRank divides each sender's rank by.
func (s *GraphSource) OutDegree(v int) int { return s.region.OutDegree(v) }

// Close releases the mapping.
func (s *GraphSource) Close() error { return s.r.Close() }
