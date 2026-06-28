package collection

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
)

// TestStaticRankOnCCrawl builds a shard from the real crawl export and checks the
// composite static rank is well-formed, discriminating, and actually driven by its
// input signals. The strong real-data gate is the link between the composite and the
// boilerplate quality term: on the broad sample the in-shard link graph is flat, so
// page rank barely separates documents and the quality term carries the variation, so
// pages with more boilerplate (lower quality) must average a lower static rank than
// pages with little boilerplate. That proves the blend reads its columns rather than
// returning a placeholder.
func TestStaticRankOnCCrawl(t *testing.T) {
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

	levels := map[int]struct{}{}
	var loSum, hiSum float64
	var loN, hiN int
	for doc := 0; doc < res.Docs; doc++ {
		sr, ok := fr.Value(uint32(doc), feature.FeatStaticRank)
		if !ok {
			t.Fatalf("doc %d has no static rank", doc)
		}
		if math.IsNaN(sr) || math.IsInf(sr, 0) {
			t.Fatalf("doc %d static rank non-finite: %g", doc, sr)
		}
		levels[int(sr*1000)] = struct{}{}
		bp, _ := fr.Value(uint32(doc), feature.FeatBoilerplate)
		// Split on the boilerplate median-ish cutoff: low chrome vs high chrome.
		if bp < 0.2 {
			loSum += sr
			loN++
		} else if bp > 0.6 {
			hiSum += sr
			hiN++
		}
	}
	if len(levels) < 20 {
		t.Fatalf("static rank took only %d distinct levels; not discriminating", len(levels))
	}
	if loN == 0 || hiN == 0 {
		t.Fatalf("not enough spread in boilerplate to compare (lo=%d hi=%d)", loN, hiN)
	}
	loMean := loSum / float64(loN)
	hiMean := hiSum / float64(hiN)
	if loMean <= hiMean {
		t.Fatalf("low-boilerplate mean static rank %g not above high-boilerplate %g; quality term not driving the composite", loMean, hiMean)
	}
	t.Logf("docs=%d distinctStaticLevels=%d lowBoiler(n=%d)mean=%.4f highBoiler(n=%d)mean=%.4f",
		res.Docs, len(levels), loN, loMean, hiN, hiMean)
}
