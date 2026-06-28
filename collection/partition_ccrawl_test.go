package collection

import (
	"hash/fnv"
	"os"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
)

// TestPartitionCompressesCrossEdgesOnCCrawl measures the payoff slice 41 named and this
// slice delivers: assigning global ids by the host and domain partition collapses the
// cross-shard edge list's bits an edge against the uniform-random ceiling. It builds one
// realistic far-edge set over the real crawl hosts, where each page links to several
// pages on a handful of other hosts (the web's dominant link pattern), then encodes that
// same edge set twice, once with the partition's host-clustered ids and once with the
// arbitrary url-hash ids a plain MPH would assign, and compares the cross-shard blob
// size. The host-clustered ids must cost materially fewer bits an edge, because a page's
// far targets on one host become a short run of small gaps instead of full-width values.
func TestPartitionCompressesCrossEdgesOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	seen := make(map[string]struct{})
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		if _, dup := seen[cu]; dup {
			continue
		}
		seen[cu] = struct{}{}
		h := d.Host
		if h == "" {
			h = analyze.HostOf(d.URL)
		}
		docs = append(docs, convert.Document{URL: cu, Host: h})
	}
	_ = src.Close()
	n := len(docs)
	if n == 0 {
		t.Skip("no canonical URLs in parquet")
	}

	// Group the real pages by host and rank the hosts, so the synthetic edges can point
	// at whole other hosts the way real outbound links do.
	hostPages := make(map[string][]int)
	for i, d := range docs {
		hostPages[d.Host] = append(hostPages[d.Host], i)
	}
	hosts := make([]string, 0, len(hostPages))
	for h := range hostPages {
		hosts = append(hosts, h)
	}
	sort.Strings(hosts)
	hostRank := make(map[string]int, len(hosts))
	for r, h := range hosts {
		hostRank[h] = r
	}

	// Each page links to a few pages on each of a few other hosts: the realistic far-edge
	// shape, several targets sharing a host. The edges are (dense source, target page
	// index) pairs, encoded the same way under both id schemes so only the ids differ.
	type pair struct {
		from int
		tgt  int
	}
	const targetHosts = 3
	const pagesPerHost = 4
	var edges []pair
	for i, d := range docs {
		r := hostRank[d.Host]
		for k := 1; k <= targetHosts; k++ {
			th := hosts[(r+k*977)%len(hosts)]
			pages := hostPages[th]
			for s := 0; s < pagesPerHost && s < len(pages); s++ {
				tp := pages[(i*3+s*101)%len(pages)]
				if tp != i {
					edges = append(edges, pair{from: i, tgt: tp})
				}
			}
		}
	}
	if len(edges) == 0 {
		t.Fatal("no synthetic far edges generated")
	}

	// Partition ids: the host-clustered assignment under test.
	partition := AssignGlobalIDs(docs, DefaultPartitionParams())

	// Baseline ids: an arbitrary url hash in the same 40-bit range, what a plain MPH that
	// ignores host locality assigns. fnv64 over the canonical url, masked to 40 bits.
	hashed := make([]uint64, n)
	for i, d := range docs {
		h := fnv.New64a()
		_, _ = h.Write([]byte(d.URL))
		hashed[i] = h.Sum64() & ((uint64(1) << 40) - 1)
	}

	crossBytes := func(ids []uint64) int {
		empty := graph.NewBuilder(n).Build()
		b := graph.NewBuilder(n)
		for _, e := range edges {
			b.AddCrossEdge(e.from, ids[e.tgt])
		}
		return len(b.Build()) - len(empty)
	}

	partBytes := crossBytes(partition)
	baseBytes := crossBytes(hashed)
	partBits := float64(partBytes*8) / float64(len(edges))
	baseBits := float64(baseBytes*8) / float64(len(edges))

	t.Logf("pages=%d hosts=%d far-edges=%d partition=%.2f bits/edge baseline=%.2f bits/edge (%.1fx smaller)",
		n, len(hosts), len(edges), partBits, baseBits, baseBits/partBits)

	if partBits >= baseBits {
		t.Fatalf("partition ids did not beat the url-hash baseline: %.2f vs %.2f bits/edge", partBits, baseBits)
	}
	// The collapse is real, not marginal: the host-clustered ids should cost a clear
	// fraction of the arbitrary-id ceiling on this realistic far-edge shape.
	if partBits > baseBits*0.75 {
		t.Fatalf("partition only reached %.2f bits/edge against a %.2f baseline, expected a clear collapse", partBits, baseBits)
	}
}
