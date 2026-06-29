package collection

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
)

// dequantByID indexes a slice of recipes by feature id for a column-wise compare.
func dequantByID(ds []feature.Dequant) map[feature.FeatureID]feature.Dequant {
	m := make(map[feature.FeatureID]feature.Dequant, len(ds))
	for _, d := range ds {
		m[d.ID] = d
	}
	return m
}

// checkFooterDequant opens one shard and checks the per-column dequant constants in its
// footer statistics reproduce the feature region's own recipe exactly, the agreement the
// two-copy design rests on. It returns the number of columns checked so the caller can
// confirm the whole schema was covered.
func checkFooterDequant(t *testing.T, path string) int {
	t.Helper()
	r, err := tsumugi.Open(path)
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
		t.Fatalf("feature.Open: %v", err)
	}

	region := dequantByID(fr.Dequant())
	footer := dequantByID(feature.ReadDequantStats(r.Footer.Stats, feature.DefaultSchema()))
	if len(footer) != len(region) {
		t.Fatalf("footer holds %d dequant recipes, region holds %d", len(footer), len(region))
	}
	for id, want := range region {
		got, ok := footer[id]
		if !ok {
			t.Fatalf("column %d absent from footer dequant block", id)
		}
		if got != want {
			t.Fatalf("column %d: footer %+v, region %+v", id, got, want)
		}
	}
	return len(region)
}

// TestFooterDequantMatchesRegion builds a synthetic collection and checks every shard's
// footer dequant block reproduces its feature region's recipe over the full schema.
func TestFooterDequantMatchesRegion(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "a.jsonl", 120, 0), Out: out, ShardSize: 40}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	infos, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) < 2 {
		t.Fatalf("expected several shards, got %d", len(infos))
	}
	for _, info := range infos {
		if n := checkFooterDequant(t, info.Path); n != len(feature.DefaultSchema()) {
			t.Fatalf("shard %s covered %d columns, want %d", filepath.Base(info.Path), n, len(feature.DefaultSchema()))
		}
	}
}

// TestFooterScalarStats checks the shard-level numbers the build now fills in: the node
// range brackets each shard's global ids, avg_doc_len is the token mean, and term_count
// equals the lexical dictionary size.
func TestFooterScalarStats(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: writeJSONL(t, tmp, "a.jsonl", 90, 0), Out: out, ShardSize: 40}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	infos, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			t.Fatal(err)
		}

		nodeMin, ok := r.Stat(tsumugi.StatNodeMin)
		if !ok || nodeMin != float64(info.NodeBase) {
			t.Fatalf("shard %s node_min = %v (ok=%v), want %d", filepath.Base(info.Path), nodeMin, ok, info.NodeBase)
		}
		nodeMax, ok := r.Stat(tsumugi.StatNodeMax)
		wantMax := float64(info.NodeBase) + float64(info.DocCount) - 1
		if !ok || nodeMax != wantMax {
			t.Fatalf("shard %s node_max = %v (ok=%v), want %v", filepath.Base(info.Path), nodeMax, ok, wantMax)
		}

		tok, ok := r.Stat(tsumugi.StatTokenCount)
		if !ok {
			t.Fatalf("shard %s missing token_count", filepath.Base(info.Path))
		}
		avg, ok := r.Stat(tsumugi.StatAvgDocLen)
		if !ok || math.Abs(avg-tok/float64(info.DocCount)) > 1e-9 {
			t.Fatalf("shard %s avg_doc_len = %v, want %v", filepath.Base(info.Path), avg, tok/float64(info.DocCount))
		}

		lb, err := r.Region(tsumugi.RegionLexical)
		if err != nil {
			t.Fatal(err)
		}
		lr, err := lexical.Open(lb)
		if err != nil {
			t.Fatal(err)
		}
		tc, ok := r.Stat(tsumugi.StatTermCount)
		if !ok || tc != float64(lr.TermCount()) {
			t.Fatalf("shard %s term_count = %v, want %d", filepath.Base(info.Path), tc, lr.TermCount())
		}
		_ = r.Close()
	}
}

// TestFooterStatsOnCCrawl runs the footer-statistics gates over the real crawl: every
// shard's footer dequant block reproduces its feature region's recipe over real,
// varied signal ranges, and the node range and average length are consistent with the
// shard's own counts.
func TestFooterStatsOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	if _, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 4096, Limit: 8000}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	infos, err := List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) == 0 {
		t.Skip("no shards")
	}
	for _, info := range infos {
		if n := checkFooterDequant(t, info.Path); n != len(feature.DefaultSchema()) {
			t.Fatalf("shard %s covered %d columns, want %d", filepath.Base(info.Path), n, len(feature.DefaultSchema()))
		}
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			t.Fatal(err)
		}
		nodeMax, _ := r.Stat(tsumugi.StatNodeMax)
		if nodeMax != float64(info.NodeBase)+float64(info.DocCount)-1 {
			t.Fatalf("shard %s node_max %v inconsistent with base %d count %d", filepath.Base(info.Path), nodeMax, info.NodeBase, info.DocCount)
		}
		avg, ok := r.Stat(tsumugi.StatAvgDocLen)
		if !ok || avg <= 0 {
			t.Fatalf("shard %s avg_doc_len = %v", filepath.Base(info.Path), avg)
		}
		_ = r.Close()
	}
}
