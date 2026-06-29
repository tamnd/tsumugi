package collection

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/graph"
)

// TestShardedHostRankMatchesMergedFromBuild proves the sharded host and domain rank
// (graph.StreamGroupRank) computed over the per-shard graph regions a real build writes
// equals the single-Region graph.HostRank and graph.DomainRank over the merged collection
// graph, page for page. The sharded path reads each node's host group from its partition
// global id (the id table slice 44 wired into every shard) without ever materializing the
// merged page graph, the scale form doc 07 L985-994 specifies; this gate ties it to the
// real build output, so the id tables, the cross-shard edge lists, and the partition split
// the build emits all line up with what the projection decodes.
func TestShardedHostRankMatchesMergedFromBuild(t *testing.T) {
	tmp := t.TempDir()
	src := writeCrossLinkedJSONL(t, tmp, "web.jsonl")
	out := filepath.Join(tmp, "col")
	const shardSize = 40
	res, err := Build(Options{Source: src, Out: out, ShardSize: shardSize})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("want a multi-shard collection, got %d shards", res.Shards)
	}

	docs, gids, _ := buildLayout(t, src)
	seqBits := partitionSeqBits(docs, DefaultPartitionParams())

	// Label every node by its partition host group and by the dense id of its registered
	// domain, the same two groupings the sharded closures decode from the global id, so the
	// merged ranks and the sharded ranks share a labeling and any difference is the
	// projection, not the labels.
	hostOf := make([]int, len(docs))
	domainOf := make([]int, len(docs))
	domainID := map[string]int{}
	domainOfGroup := map[int]int{}
	for i, d := range docs {
		grp := int(gids[i] >> seqBits)
		hostOf[i] = grp
		dom := analyze.RegisteredDomain(d.Host)
		id, ok := domainID[dom]
		if !ok {
			id = len(domainID)
			domainID[dom] = id
		}
		domainOf[i] = id
		domainOfGroup[grp] = id
	}

	cfg := graph.PRConfig{Alpha: 0.85, MaxIters: 500, Tol: 1e-10}
	mono := buildGraph(docs, buildDir(docs))
	wantHost := graph.HostRank(mono, hostOf, cfg)
	wantDomain := graph.DomainRank(mono, domainOf, cfg)

	regions, closeAll := loadShardGraphs(t, out)
	defer closeAll()

	gotHost := graph.StreamGroupRank(regions, func(g uint64) int { return int(g >> seqBits) }, cfg)
	gotDomain := graph.StreamGroupRank(regions, func(g uint64) int { return domainOfGroup[int(g>>seqBits)] }, cfg)

	var maxHost, maxDomain float64
	for i := range docs {
		s := i / shardSize
		d := i % shardSize
		if diff := math.Abs(gotHost[s][d] - wantHost[i]); diff > maxHost {
			maxHost = diff
		}
		if diff := math.Abs(gotDomain[s][d] - wantDomain[i]); diff > maxDomain {
			maxDomain = diff
		}
	}
	if maxHost > 1e-9 {
		t.Fatalf("sharded host rank diverges from merged: maxErr %g", maxHost)
	}
	if maxDomain > 1e-9 {
		t.Fatalf("sharded domain rank diverges from merged: maxErr %g", maxDomain)
	}
	t.Logf("docs=%d shards=%d hostErr=%g domainErr=%g", len(docs), len(regions), maxHost, maxDomain)
}
