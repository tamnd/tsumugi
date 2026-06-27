package collection

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
)

// crossHostDocs builds the canonical cross-host fixture: a hub and an unlinked
// page on one host, and spokes spokes on a second host each linking to the hub.
func crossHostDocs(spokes int) []convert.Document {
	docs := []convert.Document{
		{URL: "https://a.example/hub", Host: "a.example", Body: "# Hub"},
		{URL: "https://a.example/lonely", Host: "a.example", Body: "# Lonely\nno links"},
	}
	for i := 0; i < spokes; i++ {
		docs = append(docs, convert.Document{
			URL:  fmt.Sprintf("https://b.example/s%d", i),
			Host: "b.example",
			Body: fmt.Sprintf("# Spoke %d\nsee <https://a.example/hub>", i),
		})
	}
	return docs
}

// TestGlobalSignalsCrossHost proves the collection-wide pass computes every link
// signal correctly, not just PageRank. Ten spokes on host b each link to the hub on
// host a. The hub draws all ten in-links from one domain, so its in-degree is ten
// and its distinct-linking-domain count is one; host a and domain a receive the
// cross-group links so they outrank host b and domain b; and the seed-biased ranks
// stay well-formed (trust normalized, spam mass bounded).
func TestGlobalSignalsCrossHost(t *testing.T) {
	const spokes = 10
	docs := crossHostDocs(spokes)
	sig := globalSignals(docs, nil, nil)

	if sig.inDegree[0] != spokes {
		t.Fatalf("hub in-degree = %d, want %d", sig.inDegree[0], spokes)
	}
	if sig.inDegree[1] != 0 {
		t.Fatalf("lonely page in-degree = %d, want 0", sig.inDegree[1])
	}
	// All ten spokes live on b.example, one registered domain, so the hub counts a
	// single distinct linking domain however many spokes point at it.
	if sig.linkingDomains[0] != 1 {
		t.Fatalf("hub linking domains = %d, want 1", sig.linkingDomains[0])
	}
	if sig.linkingDomains[1] != 0 {
		t.Fatalf("lonely page linking domains = %d, want 0", sig.linkingDomains[1])
	}
	// Host a receives the cross-host links, host b sends them, so host a outranks
	// host b. docs[0] and docs[1] are host a, docs[2] is a spoke on host b.
	if sig.hostRank[0] <= sig.hostRank[2] {
		t.Fatalf("host a rank %g not above host b rank %g", sig.hostRank[0], sig.hostRank[2])
	}
	if sig.domainRank[0] <= sig.domainRank[2] {
		t.Fatalf("domain a rank %g not above domain b rank %g", sig.domainRank[0], sig.domainRank[2])
	}
	// Trust is a normalized distribution and spam mass is a clamped fraction.
	var trustSum float64
	for i, v := range sig.trust {
		trustSum += v
		if v < 0 {
			t.Fatalf("negative trust at %d: %g", i, v)
		}
	}
	if trustSum < 0.99 || trustSum > 1.01 {
		t.Fatalf("trust sums to %g, want ~1", trustSum)
	}
	for i, v := range sig.spamMass {
		if v < 0 || v > 1 {
			t.Fatalf("spam mass at %d = %g, out of [0,1]", i, v)
		}
	}
}

// TestBuildBakesGraphSignals proves the cross-shard link signals reach the feature
// matrix serving reads, not just PageRank. A shard size of two cuts host a (hub plus
// lonely page) into shard 0 and the spokes on host b into shard 1, so host rank,
// domain rank, in-degree, and linking domains are all genuinely cross-shard. After
// the build each column read back from the shards must order the hub above the
// pages that lack its in-links.
func TestBuildBakesGraphSignals(t *testing.T) {
	const spokes = 10
	tmp := t.TempDir()
	src := filepath.Join(tmp, "signals.jsonl")
	f, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	writeRec(t, f, "https://a.example/hub", "a.example", "# Hub")
	writeRec(t, f, "https://a.example/lonely", "a.example", "# Lonely\nplain page")
	for i := 0; i < spokes; i++ {
		writeRec(t, f, fmt.Sprintf("https://b.example/s%d", i), "b.example",
			fmt.Sprintf("# Spoke %d\nsee <https://a.example/hub>", i))
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	out := filepath.Join(tmp, "col")
	if _, err := Build(Options{Source: src, Out: out, ShardSize: 2}); err != nil {
		t.Fatalf("build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(shards) < 2 {
		t.Fatalf("want hosts split across at least 2 shards, got %d", len(shards))
	}

	feat := func(shard int, doc uint32, fid feature.FeatureID) float64 {
		r, err := tsumugi.Open(shards[shard].Path)
		if err != nil {
			t.Fatalf("open shard %d: %v", shard, err)
		}
		defer func() { _ = r.Close() }()
		b, err := r.Region(tsumugi.RegionFeature)
		if err != nil {
			t.Fatalf("feature region shard %d: %v", shard, err)
		}
		fr, err := feature.Open(b)
		if err != nil {
			t.Fatalf("open feature shard %d: %v", shard, err)
		}
		v, ok := fr.Value(doc, fid)
		if !ok {
			t.Fatalf("shard %d doc %d feature %d not set", shard, doc, fid)
		}
		return v
	}

	// Shard 0 holds host a: doc 0 is the hub, doc 1 the unlinked page. Shard 1
	// holds host b: doc 0 is a spoke.
	hubIn := feat(0, 0, feature.FeatInDegree)
	lonelyIn := feat(0, 1, feature.FeatInDegree)
	if hubIn <= lonelyIn {
		t.Fatalf("hub in-degree %g not above unlinked page %g", hubIn, lonelyIn)
	}
	if ld := feat(0, 0, feature.FeatLinkingDomains); ld <= feat(0, 1, feature.FeatLinkingDomains) {
		t.Fatalf("hub linking domains %g not above unlinked page %g", ld, feat(0, 1, feature.FeatLinkingDomains))
	}
	// Host and domain rank cross the shard boundary: the hub's host (shard 0) beats
	// a spoke's host (shard 1).
	if feat(0, 0, feature.FeatHostRank) <= feat(1, 0, feature.FeatHostRank) {
		t.Fatalf("hub host rank not above spoke host rank across the shard boundary")
	}
	if feat(0, 0, feature.FeatDomainRank) <= feat(1, 0, feature.FeatDomainRank) {
		t.Fatalf("hub domain rank not above spoke domain rank across the shard boundary")
	}
}

// TestGraphSignalsOnCCrawl records that every signal column is well-formed on the
// real collection: the rank distributions normalize, spam mass is bounded, and the
// counts are non-negative. The graph itself barely materializes on a breadth-first
// crawl (see TestGlobalGraphOnCCrawl), so this gates the shape of the signals, not
// their spread, which waits on a crawl with depth.
func TestGraphSignalsOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	sig := globalSignals(docs, nil, nil)
	if len(sig.pageRank) != len(docs) {
		t.Fatalf("pagerank length %d != docs %d", len(sig.pageRank), len(docs))
	}
	sum := func(v []float64) float64 {
		var s float64
		for _, x := range v {
			s += x
		}
		return s
	}
	if s := sum(sig.pageRank); s < 0.99 || s > 1.01 {
		t.Fatalf("pagerank sums to %g, want ~1", s)
	}
	if s := sum(sig.trust); s < 0.99 || s > 1.01 {
		t.Fatalf("trust sums to %g, want ~1", s)
	}
	// Host and domain rank are per-group ranks each page inherits, so they sum to
	// one over the distinct groups, not over the pages. Dedup by group before
	// summing.
	hostOf, domainOf := groupings(docs)
	groupSum := func(groupOf []int, vals []float64) float64 {
		seen := map[int]float64{}
		for i, gid := range groupOf {
			seen[gid] = vals[i]
		}
		var s float64
		for _, v := range seen {
			s += v
		}
		return s
	}
	if s := groupSum(hostOf, sig.hostRank); s < 0.99 || s > 1.01 {
		t.Fatalf("host rank sums to %g over distinct hosts, want ~1", s)
	}
	if s := groupSum(domainOf, sig.domainRank); s < 0.99 || s > 1.01 {
		t.Fatalf("domain rank sums to %g over distinct domains, want ~1", s)
	}
	for i, v := range sig.spamMass {
		if v < 0 || v > 1 {
			t.Fatalf("spam mass at %d = %g, out of [0,1]", i, v)
		}
	}
	var totIn, totLD int
	for i := range docs {
		if sig.inDegree[i] < 0 || sig.linkingDomains[i] < 0 {
			t.Fatalf("negative count at %d", i)
		}
		totIn += sig.inDegree[i]
		totLD += sig.linkingDomains[i]
	}
	t.Logf("docs=%d totalInDegree=%d totalLinkingDomains=%d", len(docs), totIn, totLD)
}
