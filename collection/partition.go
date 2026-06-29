package collection

import (
	"sort"
	"strings"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
)

// This file assigns the corpus-wide global node id by the host and domain partition
// doc 06 (L391-407) and doc 02 specify. The global node id is the corpus-stable name
// an edge uses for a far endpoint, and the cross-shard edge list (slice 41) gap-encodes
// a node's far targets against it, so the bits an edge the list costs are set by how
// those target ids are spread: uniform-random ids leave every gap a full-width value
// (slice 41's 51-bit ceiling), while ids that put a host's pages in a contiguous id
// range collapse a page's far targets, which are mostly several pages on a handful of
// other hosts, into short runs with small gaps.
//
// The partition packs that locality into the id itself, doc 02's "a few high bits of
// structure, for example a host group prefix": the high bits are the host's group id,
// its rank among all hosts ordered by reversed registered domain so a domain's hosts
// are adjacent in group space, and the low bits are the page's sequence within its
// host. A page's far out-neighbors on one other host then share that host's high-bit
// prefix and differ only in the low sequence bits, so the sorted-target gap encoding
// pays a handful of bits a run instead of a full id.
//
// This is the global id only. The dense docID, the shard-local index Recursive Graph
// Bisection assigns for the postings and forward order, is a separate id (doc 02's
// three-id model), and the id table (slice 40) is the map between them; the partition
// here never touches the dense order.

// PartitionParams controls how the 64-bit global id space splits between the host-group
// prefix and the within-host page sequence.
type PartitionParams struct {
	// SeqBits is a floor on the low bits reserved for a page's sequence within its host,
	// so a host may hold at least 2^SeqBits pages and the remaining high bits hold its
	// group id. AssignGlobalIDs always widens SeqBits to fit the largest host, so the
	// field is a floor for growth headroom, not a cap; the default floor is zero, which
	// fits the split tight to the corpus and keeps the ids dense.
	SeqBits uint
}

// DefaultPartitionParams returns the canonical split: a zero sequence-bit floor, which
// makes AssignGlobalIDs fit the low field exactly to the largest host and pack the host
// group into every bit above it, doc 02's host group prefix in the high bits with no
// wasted id space. The tight fit keeps the ids dense, which is what makes the gaps in
// the cross-shard edge list small: a wide fixed split would inflate the ids and the gaps
// with them. A non-zero floor reserves per-host headroom for a corpus that recrawls
// hosts deeper over time, at the cost of denser ids now.
func DefaultPartitionParams() PartitionParams {
	return PartitionParams{SeqBits: 0}
}

// reversedDomain returns the registered domain with its dot-separated labels reversed,
// the key doc 06 (L398) orders hosts by so a domain's hosts cluster: example.com sorts
// as com.example, so www.example.com and blog.example.com sit together under com.example
// and ahead of example.org's com... no, org.example. Reversing makes the public suffix
// the primary sort key, which groups a domain's hosts and keeps sibling domains apart.
func reversedDomain(host string) string {
	dom := analyze.RegisteredDomain(host)
	if dom == "" {
		return ""
	}
	labels := strings.Split(dom, ".")
	for i, j := 0, len(labels)-1; i < j; i, j = i+1, j-1 {
		labels[i], labels[j] = labels[j], labels[i]
	}
	return strings.Join(labels, ".")
}

// hostGroupKey is the sort key that orders hosts so a domain's hosts are adjacent: the
// reversed registered domain first (so siblings under one domain cluster and distinct
// domains stay apart), then the full host (so a domain's own hosts sort stably among
// themselves). The null byte separator keeps a short reversed domain from colliding
// with a longer one's prefix.
func hostGroupKey(host string) string {
	return reversedDomain(host) + "\x00" + host
}

// partitionSeqBits returns the sequence-bit width AssignGlobalIDs splits the id at for
// docs under p: the requested floor, widened until the low field holds the largest host.
// It is the one place the widening rule lives, so a caller that needs to decode a global
// id back to its host group (the high bits above the sequence, what the host and domain
// rank projects by) recovers the same split the assignment used. The host of a document is
// its Host field, falling back to the canonical URL's host, the same key AssignGlobalIDs
// groups by, so the largest-host count matches.
func partitionSeqBits(docs []convert.Document, p PartitionParams) uint {
	seqBits := p.SeqBits
	counts := make(map[string]int, len(docs))
	var maxHost int
	for _, d := range docs {
		h := d.Host
		if h == "" {
			h = analyze.HostOf(d.URL)
		}
		counts[h]++
		if counts[h] > maxHost {
			maxHost = counts[h]
		}
	}
	for seqBits < 63 && uint64(maxHost) > (uint64(1)<<seqBits) {
		seqBits++
	}
	return seqBits
}

// AssignGlobalIDs assigns every document a corpus-wide global node id by the host and
// domain partition: it groups the documents by host, orders the hosts by reversed
// registered domain (so a domain's hosts take adjacent group ids), assigns each host a
// group id by that order, and gives each document the id (group id << seqBits) | its
// sequence within its host. The result is indexed by the input order, id[i] the global
// id of docs[i], and is a collision-free assignment, every distinct document a distinct
// id, because group ids are distinct across hosts and sequences are distinct within one.
//
// SeqBits widens past the requested floor if any host holds more than 2^SeqBits pages,
// so the low field always fits the largest host and the assignment never overflows into
// a neighbouring group. The host whose documents are passed is taken from the document's
// Host field, falling back to the canonical URL's host, so a document the converter
// already tagged with a host is grouped without re-parsing.
func AssignGlobalIDs(docs []convert.Document, p PartitionParams) []uint64 {
	n := len(docs)
	ids := make([]uint64, n)
	if n == 0 {
		return ids
	}

	// Group document indices by host.
	byHost := make(map[string][]int)
	hostKey := make(map[string]string)
	for i, d := range docs {
		h := d.Host
		if h == "" {
			h = analyze.HostOf(d.URL)
		}
		byHost[h] = append(byHost[h], i)
		if _, ok := hostKey[h]; !ok {
			hostKey[h] = hostGroupKey(h)
		}
	}

	// Order the hosts by reversed-domain group key so a domain's hosts are adjacent.
	hosts := make([]string, 0, len(byHost))
	for h := range byHost {
		hosts = append(hosts, h)
	}
	sort.Slice(hosts, func(i, j int) bool {
		ki, kj := hostKey[hosts[i]], hostKey[hosts[j]]
		if ki != kj {
			return ki < kj
		}
		return hosts[i] < hosts[j]
	})

	// Widen the sequence field if the largest host overflows the requested floor, the same
	// split partitionSeqBits exposes to a caller that later decodes the id back to its group.
	seqBits := partitionSeqBits(docs, p)

	// Assign each host a group id by its order, and each document the packed id, its
	// pages sequenced by url so a host's pages are in a stable within-host order.
	for groupID, h := range hosts {
		idxs := byHost[h]
		sort.Slice(idxs, func(a, b int) bool { return docs[idxs[a]].URL < docs[idxs[b]].URL })
		base := uint64(groupID) << seqBits
		for seq, i := range idxs {
			ids[i] = base | uint64(seq)
		}
	}
	return ids
}
