package collection

import (
	"sort"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/langid"
	"github.com/tamnd/tsumugi/mph"
)

// shardedSignals assembles the complete graphSignals from the persisted per-shard
// graph regions, the drop-in replacement for globalSignals that never builds a
// second full-corpus in-core graph. It is the assembly the M15 reorder rests on:
// every link signal globalSignals computes over one merged graph (slices 42-90 each
// gave its own cross-shard form, gated bit-for-bit against the merged oracle) is
// computed here off the shard regions and folded back to one value per document.
//
// shards are the shard graph regions in shard (NodeBase) order, docs the documents
// in their final dense order, gids each document's partition global id, dir the
// canonical-URL directory, and params the partition split. The shards are contiguous
// slices of docs in order, so a document's index is its dense position concatenated
// across the shards in shard order, the mapping shardStarts and locate carry.
//
// The result equals globalSignals(docs, ...) field for field: the ranks to float32
// precision (each cross-shard rank loop matches the in-core rank to that precision,
// far finer than the byte-quantized feature column keeps), the integer counts exactly,
// the categorical language id exactly, the content signals trivially because they are
// the same function of the same bodies.
func shardedSignals(shards []*graph.Region, docs []convert.Document, gids []uint64, trustSeeds, spamSeeds []string, dir *mph.Dir, params PartitionParams) graphSignals {
	n := len(docs)
	if n == 0 {
		return graphSignals{}
	}
	cfg := graph.DefaultPRConfig()
	starts := shardStarts(shards)

	// The partition packs the host group into the high bits above the sequence field,
	// so g >> seqBits is a document's host group, one distinct group per host: the same
	// grouping groupings(docs) induces, named by the partition id rather than first-seen
	// order. The group ranks and distinct-linking counts scatter a per-group value to
	// member pages, so they depend on the grouping not the labeling and agree with the
	// merged forms whatever the labeling.
	seqBits := partitionSeqBits(docs, params)
	hostGroupOf := func(g uint64) int { return int(g >> seqBits) }

	// Map each host group to its registered domain's dense id, so a domain's hosts share
	// a domain value, the same relation the merged domainOf carries.
	domainOfGroup := map[int]int{}
	domID := map[string]int{}
	for i := range docs {
		grp := int(gids[i] >> seqBits)
		if _, ok := domainOfGroup[grp]; ok {
			continue
		}
		h := docs[i].Host
		if h == "" {
			h = analyze.HostOf(docs[i].URL)
		}
		dom := analyze.RegisteredDomain(h)
		did, ok := domID[dom]
		if !ok {
			did = len(domID)
			domID[dom] = did
		}
		domainOfGroup[grp] = did
	}
	domainGroupOf := func(g uint64) int { return domainOfGroup[int(g>>seqBits)] }

	// The ranks run in the order globalSignals follows: forward PageRank, then the spam
	// seeds bias the reversed anti-trust rank, the inverse rank picks the trust-seed
	// candidates, the anti-trust filter drops the ones that point at spam, and the trust
	// seeds bias the forward rank.
	pr := flattenRanks(graph.StreamCrossPageRank(shards, cfg))

	spamIdx := resolveSeeds(spamSeeds, dir)
	spamDense := perShardSeeds(spamIdx, shards, starts)
	anti := flattenRanks(graph.StreamCrossReversedPageRankP(shards, graph.SeedCrossTeleport(shards, spamDense), cfg))

	inv := flattenRanks(graph.StreamCrossReversedPageRank(shards, cfg))
	trustIdx := selectTrustSeeds(trustSeeds, dir, inv, anti, n)
	trustDense := perShardSeeds(trustIdx, shards, starts)
	tr := flattenRanks(graph.StreamCrossPageRankP(shards, graph.SeedCrossTeleport(shards, trustDense), cfg))

	sm := graph.SpamMass(pr, tr, trustIdx)

	// The outbound-spam ratio reads each out-neighbor's spam mass by its global id, so
	// build the global-id-to-document-index map once and close over SpamMass through it.
	idxOfGid := make(map[uint64]int, n)
	for i := range gids {
		idxOfGid[gids[i]] = i
	}
	spamOfGlobal := func(g uint64) float64 { return sm[idxOfGid[g]] }

	// The content signals are functions of the document bodies, not the graph, so they
	// run exactly as globalSignals computes them, off the same one-pass detection.
	detLang, detConf := detectLanguages(docs, langid.New())

	sig := graphSignals{
		pageRank:       pr,
		hostRank:       flattenFloats(graph.StreamGroupRank(shards, hostGroupOf, cfg)),
		domainRank:     flattenFloats(graph.StreamGroupRank(shards, domainGroupOf, cfg)),
		trust:          tr,
		spamMass:       sm,
		inDegree:       flattenInts(graph.CrossInDegrees(shards)),
		linkingDomains: flattenInts(graph.CrossLinkingDomains(shards, domainGroupOf)),
		linkingHosts:   flattenInts(graph.CrossLinkingHosts(shards, hostGroupOf)),
		reciprocity:    flattenFloats(graph.CrossReciprocity(shards)),
		hostLinkDiv:    flattenFloats(graph.CrossHostLinkDiversity(shards, hostGroupOf)),
		nearDup:        nearDupPenalties(docs, pr),
		outboundSpam:   flattenFloats(graph.CrossOutboundSpamRatio(shards, spamOfGlobal, graph.DefaultSpamThreshold)),
		langConsist:    languageConsistencyFrom(docs, detLang, detConf),
		langID:         languageIDsFrom(detLang, detConf),
	}
	// The composite static rank is the same blend over the assembled columns, computed
	// last so it reads them.
	sig.staticRank = compositeStaticRank(docs, sig)
	return sig
}

// shardStarts returns the running sum of per-shard node counts, so starts[s] is the
// document index of shard s's first node and starts[len(shards)] is the total. A
// document's index is its dense position within its shard plus its shard's start,
// because the shards are contiguous slices of docs in shard order.
func shardStarts(shards []*graph.Region) []int {
	starts := make([]int, len(shards)+1)
	for s, g := range shards {
		starts[s+1] = starts[s] + g.NodeCount()
	}
	return starts
}

// locate maps a flat document index back to its shard and dense position within it,
// the inverse of shardStarts via a binary search over the running sums.
func locate(starts []int, idx int) (shard, dense int) {
	s := sort.Search(len(starts), func(i int) bool { return starts[i] > idx }) - 1
	return s, idx - starts[s]
}

// perShardSeeds routes each selected seed (a flat document index) to the dense seed
// list of the shard that holds it, the [][]int SeedCrossTeleport takes.
func perShardSeeds(idxSeeds []int, shards []*graph.Region, starts []int) [][]int {
	dense := make([][]int, len(shards))
	for _, s := range idxSeeds {
		sh, d := locate(starts, s)
		dense[sh] = append(dense[sh], d)
	}
	return dense
}

// flattenRanks folds the per-shard float32 cross-shard ranks back to flat document
// order, widening to float64 the same way globalSignals widens its in-core ranks.
func flattenRanks(perShard [][]float32) []float64 {
	var out []float64
	for _, s := range perShard {
		for _, v := range s {
			out = append(out, float64(v))
		}
	}
	return out
}

// flattenFloats folds the per-shard float64 cross-shard values back to flat order.
func flattenFloats(perShard [][]float64) []float64 {
	var out []float64
	for _, s := range perShard {
		out = append(out, s...)
	}
	return out
}

// flattenInts folds the per-shard int cross-shard counts back to flat order.
func flattenInts(perShard [][]int) []int {
	var out []int
	for _, s := range perShard {
		out = append(out, s...)
	}
	return out
}
