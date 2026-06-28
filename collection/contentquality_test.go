package collection

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
)

// TestContentQualitySignalsOnCCrawl builds a shard from the real crawl export and
// checks the two content-quality signals this slice adds, the boilerplate ratio and
// the outbound-spam ratio, are populated and well-formed across every document.
//
// Boilerplate is the stronger real-data gate: a broad web sample mixes prose pages
// with link-list and navigation pages, so the ratio must both stay in [0,1] and vary
// across the corpus, otherwise the markdown extractor is reading nothing. The
// outbound-spam ratio is checked for well-formedness and logged rather than required
// to be nonzero, because the in-shard link graph resolves few edges and flags few
// spam targets on a broad sample, the same cross-shard link gap the graph tests note.
func TestContentQualitySignalsOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "col")
	res, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 100000, Limit: 8000})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("want 1 shard for the limit, got %d", len(shards))
	}
	r, err := tsumugi.Open(shards[0].Path)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = r.Close() }()

	fb, err := r.Region(tsumugi.RegionFeature)
	if err != nil {
		t.Fatalf("feature region: %v", err)
	}
	fr, err := feature.Open(fb)
	if err != nil {
		t.Fatalf("open feature region: %v", err)
	}

	var boilerNonZero, spamNonZero, distinctBoiler int
	seen := map[int]struct{}{}
	for doc := 0; doc < res.Docs; doc++ {
		bp, ok := fr.Value(uint32(doc), feature.FeatBoilerplate)
		if !ok {
			t.Fatalf("doc %d has no boilerplate value", doc)
		}
		if bp < -0.001 || bp > 1.001 {
			t.Fatalf("doc %d boilerplate %g out of [0,1]", doc, bp)
		}
		if bp > 0.001 {
			boilerNonZero++
		}
		// Bucket to a hundredth to count how many distinct levels the signal takes,
		// the cheap test that it discriminates rather than returning a constant.
		seen[int(bp*100)] = struct{}{}

		os, ok := fr.Value(uint32(doc), feature.FeatOutboundSpam)
		if !ok {
			t.Fatalf("doc %d has no outbound-spam value", doc)
		}
		if os < -0.001 || os > 1.001 {
			t.Fatalf("doc %d outbound spam %g out of [0,1]", doc, os)
		}
		if os > 0.001 {
			spamNonZero++
		}
	}
	distinctBoiler = len(seen)

	if boilerNonZero == 0 {
		t.Fatalf("no document has any boilerplate; the markdown extractor read nothing")
	}
	if distinctBoiler < 5 {
		t.Fatalf("boilerplate took only %d distinct levels; the signal is not discriminating", distinctBoiler)
	}
	t.Logf("docs=%d boilerplateNonZero=%d distinctBoilerLevels=%d outboundSpamNonZero=%d",
		res.Docs, boilerNonZero, distinctBoiler, spamNonZero)
}
