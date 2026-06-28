package collection

import (
	"hash/fnv"
	"strconv"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
)

func doc(url, host string) convert.Document { return convert.Document{URL: url, Host: host} }

// TestAssignGlobalIDsDistinct checks the assignment is collision-free and packs the
// host group into the high bits and the within-host sequence into the low bits.
func TestAssignGlobalIDsDistinct(t *testing.T) {
	docs := []convert.Document{
		doc("https://www.example.com/a", "www.example.com"),
		doc("https://www.example.com/c", "www.example.com"),
		doc("https://www.example.com/b", "www.example.com"),
		doc("https://blog.example.com/x", "blog.example.com"),
		doc("https://other.org/p", "other.org"),
	}
	p := PartitionParams{SeqBits: 8}
	ids := AssignGlobalIDs(docs, p)

	seen := make(map[uint64]bool)
	for i, id := range ids {
		if seen[id] {
			t.Fatalf("id %d (doc %d) is a duplicate", id, i)
		}
		seen[id] = true
	}

	// Same host shares the high-bit group prefix; different hosts do not.
	grp := func(id uint64) uint64 { return id >> p.SeqBits }
	if grp(ids[0]) != grp(ids[1]) || grp(ids[0]) != grp(ids[2]) {
		t.Fatalf("www.example.com pages have different group prefixes: %d %d %d", ids[0], ids[1], ids[2])
	}
	if grp(ids[0]) == grp(ids[3]) {
		t.Fatalf("www and blog hosts share a group prefix: %d %d", ids[0], ids[3])
	}

	// Within www.example.com the sequence runs 0,1,2 in url order (a, b, c), so the
	// doc the input lists second (url /c) gets the highest sequence.
	seq := func(id uint64) uint64 { return id & ((1 << p.SeqBits) - 1) }
	if seq(ids[0]) != 0 { // /a
		t.Fatalf("url /a got sequence %d, want 0", seq(ids[0]))
	}
	if seq(ids[2]) != 1 { // /b
		t.Fatalf("url /b got sequence %d, want 1", seq(ids[2]))
	}
	if seq(ids[1]) != 2 { // /c
		t.Fatalf("url /c got sequence %d, want 2", seq(ids[1]))
	}
}

// TestAssignGlobalIDsDomainAdjacency checks doc 06's reversed-domain ordering: the hosts
// of one registered domain take adjacent group ids, even when their plain host strings
// interleave alphabetically with another domain's hosts.
func TestAssignGlobalIDsDomainAdjacency(t *testing.T) {
	// Plain host order interleaves the two domains (alpha.site.com, beta.other.com,
	// gamma.site.com), but reversed-domain order groups com.other then com.site.
	docs := []convert.Document{
		doc("https://alpha.site.com/", "alpha.site.com"),
		doc("https://beta.other.com/", "beta.other.com"),
		doc("https://gamma.site.com/", "gamma.site.com"),
	}
	p := PartitionParams{SeqBits: 8}
	ids := AssignGlobalIDs(docs, p)
	grp := func(id uint64) uint64 { return id >> p.SeqBits }

	// The two site.com hosts must have consecutive group ids, with other.com's host
	// not wedged between them.
	gAlpha, gGamma, gBeta := grp(ids[0]), grp(ids[2]), grp(ids[1])
	if gAlpha == gGamma {
		t.Fatal("two distinct hosts share a group id")
	}
	lo, hi := gAlpha, gGamma
	if lo > hi {
		lo, hi = hi, lo
	}
	if hi-lo != 1 {
		t.Fatalf("site.com hosts are not adjacent: groups %d and %d", gAlpha, gGamma)
	}
	if gBeta > lo && gBeta < hi {
		t.Fatalf("other.com host %d is wedged between the site.com hosts %d..%d", gBeta, lo, hi)
	}
}

// TestAssignGlobalIDsWiden checks the sequence field grows past the floor when a host
// holds more pages than the floor's bits allow, so the assignment never overflows a
// group into its neighbour.
func TestAssignGlobalIDsWiden(t *testing.T) {
	// Floor of 2 bits holds 4 pages; give one host 6, which forces a widening.
	var docs []convert.Document
	for _, u := range []string{"a", "b", "c", "d", "e", "f"} {
		docs = append(docs, doc("https://big.example.com/"+u, "big.example.com"))
	}
	docs = append(docs, doc("https://small.example.org/x", "small.example.org"))
	ids := AssignGlobalIDs(docs, PartitionParams{SeqBits: 2})

	seen := make(map[uint64]bool)
	for i, id := range ids {
		if seen[id] {
			t.Fatalf("id %d (doc %d) collided after widening", id, i)
		}
		seen[id] = true
	}
	// The two groups must stay separated: every big.example.com id below the
	// small.example.org id (com sorts before org under the reversed domain), and no
	// overlap, which a too-narrow sequence field would have caused.
	for i := 0; i < 6; i++ {
		if ids[i] >= ids[6] {
			t.Fatalf("big host id %d not below the small host id %d: sequence field did not widen", ids[i], ids[6])
		}
	}
}

// TestPartitionCompressesDeepCorpus measures the cross-shard payoff on a corpus with
// depth, many pages a host, the structure the real broad-crawl sample lacks (it has about
// one page a host) and the structure doc 06's host grouping is built for. Each page links
// to several pages on a few other hosts, so the host-clustered ids turn those targets into
// short within-host runs of small gaps, while the url-hash baseline leaves every gap a
// full-width value. The collapse here is much larger than on the flat real corpus.
func TestPartitionCompressesDeepCorpus(t *testing.T) {
	// 200 hosts across 20 domains, 50 pages each: real link depth.
	const domains = 20
	const hostsPerDomain = 10
	const pagesPerHost = 50
	var docs []convert.Document
	for d := 0; d < domains; d++ {
		for h := 0; h < hostsPerDomain; h++ {
			host := "h" + strconv.Itoa(h) + ".dom" + strconv.Itoa(d) + ".com"
			for pg := 0; pg < pagesPerHost; pg++ {
				docs = append(docs, doc("https://"+host+"/p"+strconv.Itoa(pg), host))
			}
		}
	}
	n := len(docs)

	// Group pages by host so edges can target several pages of one other host.
	hostPages := make(map[string][]int)
	var order []string
	for i, dd := range docs {
		if _, ok := hostPages[dd.Host]; !ok {
			order = append(order, dd.Host)
		}
		hostPages[dd.Host] = append(hostPages[dd.Host], i)
	}
	type pair struct{ from, tgt int }
	var edges []pair
	for i, dd := range docs {
		hi := 0
		for hr, h := range order {
			if h == dd.Host {
				hi = hr
				break
			}
		}
		for k := 1; k <= 3; k++ {
			th := order[(hi+k*7)%len(order)]
			pages := hostPages[th]
			for s := 0; s < 6 && s < len(pages); s++ {
				tp := pages[(i+s*13)%len(pages)]
				if tp != i {
					edges = append(edges, pair{i, tp})
				}
			}
		}
	}

	partition := AssignGlobalIDs(docs, DefaultPartitionParams())
	hashed := make([]uint64, n)
	for i, dd := range docs {
		hsh := fnv.New64a()
		_, _ = hsh.Write([]byte(dd.URL))
		hashed[i] = hsh.Sum64() & ((uint64(1) << 40) - 1)
	}

	crossBits := func(ids []uint64) float64 {
		empty := graph.NewBuilder(n).Build()
		b := graph.NewBuilder(n)
		for _, e := range edges {
			b.AddCrossEdge(e.from, ids[e.tgt])
		}
		return float64((len(b.Build())-len(empty))*8) / float64(len(edges))
	}
	partBits := crossBits(partition)
	baseBits := crossBits(hashed)
	t.Logf("deep corpus: pages=%d hosts=%d edges=%d partition=%.2f baseline=%.2f bits/edge (%.1fx)",
		n, len(order), len(edges), partBits, baseBits, baseBits/partBits)
	if partBits > baseBits*0.4 {
		t.Fatalf("with real depth the partition should collapse far below the baseline: %.2f vs %.2f", partBits, baseBits)
	}
}

// TestAssignGlobalIDsEmpty checks the empty corpus returns no ids without panicking.
func TestAssignGlobalIDsEmpty(t *testing.T) {
	if ids := AssignGlobalIDs(nil, DefaultPartitionParams()); len(ids) != 0 {
		t.Fatalf("empty corpus returned %d ids", len(ids))
	}
}
