package collection

import (
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

// buildShardGraph builds one shard's graph region from its slice of documents,
// resolving each outbound link against the collection-wide directory exactly as the
// shard writer did before the M15 reorder folded this out of writeShard. A target in
// this shard becomes an intra-shard edge keyed by the local dense docID; a target in
// another shard becomes a cross-shard edge keyed by the target's global node id, the
// name the cross-shard rank loops route against; a target the crawl never captured does
// not resolve and is dropped. The region carries the slice's partition global ids as
// its node id table. Building it here, before the signals, is the reorder: the signals
// are computed off these persisted per-shard regions rather than a second merged in-core
// graph, and writeShard then embeds the very bytes the signals were computed from.
func buildShardGraph(docs []convert.Document, lo int, gids []uint64, dir *mph.Dir) []byte {
	gb := graph.NewBuilder(len(docs)).WithNodeIDs(gids[lo : lo+len(docs)])
	for i, d := range docs {
		for _, tgt := range analyze.Links(d) {
			j, ok := dir.Lookup([]byte(tgt))
			if !ok || int(j) == lo+i {
				continue
			}
			if int(j) >= lo && int(j) < lo+len(docs) {
				gb.AddEdge(i, int(j)-lo)
			} else {
				gb.AddCrossEdge(i, gids[j])
			}
		}
	}
	return gb.Build()
}

// shardLayout is one shard's place in a build: the half-open document range it spans
// and the encoded graph region built for it. The build computes the signals over the
// regions of every shardLayout, then writes each shard from its document slice and its
// prebuilt region, so the bytes the signals read and the bytes the shard stores are one.
type shardLayout struct {
	lo, hi  int
	gregion []byte
}

// buildShardGraphs builds the per-shard graph regions in shard order, the M15 reorder's
// first pass. It cuts the documents at the shard size, builds each shard's graph region,
// and opens it, returning one shardLayout per shard and the opened regions in shard
// order, the []*graph.Region the cross-shard signal forms in shardedSignals consume. The
// opened regions alias the layout's region bytes, so the same bytes feed the signals and
// the shard files. graph.Open holds none of the adjacency resident, only the region's
// header and offset indexes, so the working set is the encoded regions, not an expanded
// adjacency, the property that lets the signals stream over the shards at scale.
func buildShardGraphs(docs []convert.Document, gids []uint64, dir *mph.Dir, shardSize int) ([]shardLayout, []*graph.Region, error) {
	var layouts []shardLayout
	var regions []*graph.Region
	for lo := 0; lo < len(docs); lo += shardSize {
		hi := lo + shardSize
		if hi > len(docs) {
			hi = len(docs)
		}
		gr := buildShardGraph(docs[lo:hi], lo, gids, dir)
		g, err := graph.Open(gr)
		if err != nil {
			return nil, nil, err
		}
		layouts = append(layouts, shardLayout{lo: lo, hi: hi, gregion: gr})
		regions = append(regions, g)
	}
	return layouts, regions, nil
}
