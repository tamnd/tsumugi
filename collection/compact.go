package collection

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"sort"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
)

// Compact merges a collection's shards into fewer, larger ones at the given shard
// size. It reads the documents back from each shard's forward store, reorders the
// whole set, and rebuilds, which reclaims the fragmentation that Add leaves behind
// when many small later crawls accumulate. The rebuild writes into a staging
// directory and swaps it in only once every new shard is written, so a failed compact
// leaves the original collection untouched, the safety the immutable-shard discipline
// buys.
func Compact(dir string, shardSize int, epoch uint64) (Result, error) {
	if shardSize <= 0 {
		shardSize = DefaultShardSize
	}
	infos, err := List(dir)
	if err != nil {
		return Result{}, err
	}
	if len(infos) == 0 {
		return Result{}, fmt.Errorf("collection: no shards to compact in %s", dir)
	}

	docs, hosts, err := gatherDocs(infos)
	if err != nil {
		return Result{}, err
	}
	// Preserve the posting ordering the collection was built with: a compact of an
	// impact-ordered collection must rebuild impact-ordered, or it would silently
	// downgrade the shards to BM25F and serve them through the wrong plane.
	impact, err := shardsAreImpact(infos)
	if err != nil {
		return Result{}, err
	}
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Host != docs[j].Host {
			return docs[i].Host < docs[j].Host
		}
		return docs[i].URL < docs[j].URL
	})

	staging := filepath.Join(dir, ".compact")
	if err := os.RemoveAll(staging); err != nil {
		return Result{}, err
	}
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return Result{}, err
	}
	// Resolve the directory and reassign the partition global node ids over the reordered
	// set, the same ids the shards a fresh build writes carry as their graph id table,
	// before recomputing any signal.
	urlDir := buildDir(docs)
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())

	// Recompute the collection-wide link signals over the reordered set the same way a
	// fresh build does: build every shard's graph region first, then join them with the
	// cross-shard rank loops, so the compacted shards carry the signals as if built whole
	// without ranking over a second merged in-core graph. A compact has no seed list to
	// thread through, so it seeds trust from inverse PageRank alone, the same default a
	// build with no curated seeds uses.
	layouts, regions, err := buildShardGraphs(docs, gids, urlDir, shardSize)
	if err != nil {
		_ = os.RemoveAll(staging)
		return Result{}, fmt.Errorf("build shard graphs: %w", err)
	}
	sig := shardedSignals(regions, docs, gids, nil, nil, urlDir, DefaultPartitionParams())

	// A compact rebuilds with no curated seeds, the same default a build with none uses,
	// so its configuration digest folds in the empty seed lists. The epoch is passed in
	// so a pinned-epoch compact is byte-identical, the same reproducibility a build gets.
	meta := shardMeta{epoch: epoch, configHash: buildConfigHash(shardSize, nil, nil, impact), impact: impact}

	res := Result{Docs: len(docs), Hosts: hosts}
	var base uint32
	index := 0
	for _, sl := range layouts {
		n, err := writeShard(shardPath(staging, index), docs[sl.lo:sl.hi], sig.slice(sl.lo, sl.hi), base, sl.gregion, meta)
		if err != nil {
			_ = os.RemoveAll(staging)
			return Result{}, err
		}
		res.Bytes += n
		base += uint32(sl.hi - sl.lo)
		index++
		res.Shards++
	}

	// Swap: remove the old shards, move the staged ones up, drop the staging dir.
	for _, in := range infos {
		if err := os.Remove(in.Path); err != nil {
			return Result{}, err
		}
	}
	staged, err := filepath.Glob(filepath.Join(staging, shardGlob))
	if err != nil {
		return Result{}, err
	}
	for _, sp := range staged {
		if err := os.Rename(sp, filepath.Join(dir, filepath.Base(sp))); err != nil {
			return Result{}, err
		}
	}
	if err := os.RemoveAll(staging); err != nil {
		return Result{}, err
	}
	// A compact rebuilds the whole collection over the reordered set from global id
	// zero, so refresh the collection-wide graph artifact the same way a fresh build
	// writes it, keeping the streamed cross-shard graph in step with the new ids.
	if graphRegion := buildGraphRegionBytes(docs, urlDir); len(graphRegion) > 0 {
		if err := writeCollectionGraph(dir, graphRegion, len(docs)); err != nil {
			return Result{}, fmt.Errorf("write collection graph: %w", err)
		}
	}
	// The shard set changed, so the old artifact's manifest and routing are stale.
	// Rebuild it over the compacted shards.
	if err := WriteIndex(dir, epoch); err != nil {
		return Result{}, fmt.Errorf("write index: %w", err)
	}
	return res, nil
}

// gatherDocs reads every document back out of a set of shards' forward stores,
// reconstructing the crawl document from the stored url and body. The host is parsed
// back from the url, and the title is re-derived at rebuild time, so the forward store
// only has to carry the url and the body for a compact to be lossless for ranking.
func gatherDocs(infos []ShardInfo) ([]convert.Document, int, error) {
	var docs []convert.Document
	hosts := map[string]struct{}{}
	for _, in := range infos {
		r, err := tsumugi.Open(in.Path)
		if err != nil {
			return nil, 0, err
		}
		if !r.HasRegion(tsumugi.RegionForward) {
			_ = r.Close()
			return nil, 0, fmt.Errorf("collection: shard %s has no forward store to compact from", filepath.Base(in.Path))
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			return nil, 0, err
		}
		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			return nil, 0, err
		}
		n := fwd.DocCount()
		for id := uint32(0); id < n; id++ {
			u, _ := fwd.Column("url", id)
			body, _ := fwd.Column("body", id)
			d := convert.Document{URL: string(u), Body: string(body), Host: hostOf(string(u))}
			docs = append(docs, d)
			hosts[d.Host] = struct{}{}
		}
		fwd.Close()
		_ = r.Close()
	}
	return docs, len(hosts), nil
}

// shardsAreImpact reports whether the collection's shards are impact-ordered, read from the
// first shard that carries a lexical region. The build writes every shard the same way, so
// one shard settles the mode; a collection with no lexical region anywhere is treated as
// docID-ordered, the default a rebuild with no impact flag produces.
func shardsAreImpact(infos []ShardInfo) (bool, error) {
	for _, in := range infos {
		r, err := tsumugi.Open(in.Path)
		if err != nil {
			return false, err
		}
		if !r.HasRegion(tsumugi.RegionLexical) {
			_ = r.Close()
			continue
		}
		b, err := r.Region(tsumugi.RegionLexical)
		if err != nil {
			_ = r.Close()
			return false, err
		}
		reg, err := lexical.Open(b)
		if err != nil {
			_ = r.Close()
			return false, err
		}
		impact := reg.IsImpact()
		_ = r.Close()
		return impact, nil
	}
	return false, nil
}

// hostOf parses the host back out of a url, the field the build ordered on, so a
// compacted collection reorders by host the same way a fresh build does.
func hostOf(raw string) string {
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Host
	}
	return raw
}
