package collection

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/forward"
)

// Compact merges a collection's shards into fewer, larger ones at the given shard
// size. It reads the documents back from each shard's forward store, reorders the
// whole set, and rebuilds, which reclaims the fragmentation that Add leaves behind
// when many small later crawls accumulate. The rebuild writes into a staging
// directory and swaps it in only once every new shard is written, so a failed compact
// leaves the original collection untouched, the safety the immutable-shard discipline
// buys.
func Compact(dir string, shardSize int) (Result, error) {
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
	// Recompute the collection-wide link signals over the reordered set, the same
	// pass a fresh build runs, so the compacted shards carry the signals as if built
	// whole. A compact has no seed list to thread through, so it seeds trust from
	// inverse PageRank alone, the same default a build with no curated seeds uses.
	sig, graphRegion, urlDir := globalSignals(docs, nil, nil)
	// Reassign the partition global node ids over the reordered set, the same id the
	// shards a fresh build writes carry as their graph id table.
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())

	res := Result{Docs: len(docs), Hosts: hosts}
	var base uint32
	index := 0
	for lo := 0; lo < len(docs); lo += shardSize {
		hi := lo + shardSize
		if hi > len(docs) {
			hi = len(docs)
		}
		n, err := writeShard(shardPath(staging, index), docs[lo:hi], sig.slice(lo, hi), base, lo, gids, urlDir)
		if err != nil {
			_ = os.RemoveAll(staging)
			return Result{}, err
		}
		res.Bytes += n
		base += uint32(hi - lo)
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
	if len(graphRegion) > 0 {
		if err := writeCollectionGraph(dir, graphRegion, len(docs)); err != nil {
			return Result{}, fmt.Errorf("write collection graph: %w", err)
		}
	}
	// The shard set changed, so the old artifact's manifest and routing are stale.
	// Rebuild it over the compacted shards.
	if err := WriteIndex(dir, uint64(time.Now().Unix())); err != nil {
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

// hostOf parses the host back out of a url, the field the build ordered on, so a
// compacted collection reorders by host the same way a fresh build does.
func hostOf(raw string) string {
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Host
	}
	return raw
}
