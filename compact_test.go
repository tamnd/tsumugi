package tsumugi_test

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi"
)

// buildStatShard writes a shard with regions of exact, known on-disk sizes so the
// RegionStats arithmetic and budget verdicts can be checked against hand-computed numbers.
// Every region is stored uncompressed, so on-disk equals raw and the per-document cost is
// just the byte length over the document count.
func buildStatShard(t *testing.T, path string, docs uint32, flags uint64, sizes map[tsumugi.RegionKind]int) {
	t.Helper()
	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	w.SetDocCount(docs)
	if flags != 0 {
		w.SetFlags(flags)
	}
	// A fixed write order so the footer's region order is predictable for the assertions.
	order := []tsumugi.RegionKind{
		tsumugi.RegionLexical, tsumugi.RegionForward, tsumugi.RegionFeature, tsumugi.RegionGraph,
	}
	for _, k := range order {
		if n, ok := sizes[k]; ok {
			if err := w.AddRegion(k, tsumugi.CodecNone, 0, 0, make([]byte, n)); err != nil {
				t.Fatalf("add %s: %v", k, err)
			}
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// TestRegionStatsBudgets is the CI gate for the production RegionStats helper the inspect
// --sizes view and the compactness benchmark both read. It builds a full-store shard whose
// regions have known sizes and checks the per-document arithmetic, the doc 13 budget
// verdicts, and the full-store special case for the forward region against hand-computed
// numbers, so the executable budget contract is proven without the real-data sample CI
// does not carry.
func TestRegionStatsBudgets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "full.tsumugi")
	// 100 docs: lexical 200 B/doc (in 120-260), feature 50 B/doc (over the 48 ceiling),
	// graph 30 B/doc (under the 40 ceiling), forward a full store reported without a verdict.
	buildStatShard(t, path, 100, 0, map[tsumugi.RegionKind]int{
		tsumugi.RegionLexical: 20000,
		tsumugi.RegionForward: 100000,
		tsumugi.RegionFeature: 5000,
		tsumugi.RegionGraph:   3000,
	})
	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() { _ = r.Close() }()

	stats := r.RegionStats()
	if len(stats) != 4 {
		t.Fatalf("want 4 region stats, got %d", len(stats))
	}
	byKind := map[tsumugi.RegionKind]tsumugi.RegionStat{}
	for _, s := range stats {
		byKind[s.Kind] = s
	}

	lex := byKind[tsumugi.RegionLexical]
	if lex.OnDisk != 20000 || lex.Raw != 20000 || lex.Ratio != 1.0 {
		t.Errorf("lexical sizes: on-disk %d raw %d ratio %.2f", lex.OnDisk, lex.Raw, lex.Ratio)
	}
	if lex.BytesPerDoc != 200 {
		t.Errorf("lexical bytes/doc = %.0f, want 200", lex.BytesPerDoc)
	}
	if !lex.HasBudget || !lex.InBudget {
		t.Errorf("lexical should be in budget: hasBudget=%v inBudget=%v", lex.HasBudget, lex.InBudget)
	}
	if !strings.Contains(lex.BudgetVerdict(), "ok") {
		t.Errorf("lexical verdict = %q, want an ok", lex.BudgetVerdict())
	}

	feat := byKind[tsumugi.RegionFeature]
	if feat.BytesPerDoc != 50 {
		t.Errorf("feature bytes/doc = %.0f, want 50", feat.BytesPerDoc)
	}
	if !feat.HasBudget || feat.InBudget {
		t.Errorf("feature should be over budget: hasBudget=%v inBudget=%v", feat.HasBudget, feat.InBudget)
	}
	if !strings.Contains(feat.BudgetVerdict(), "OVER") {
		t.Errorf("feature verdict = %q, want an OVER", feat.BudgetVerdict())
	}

	graph := byKind[tsumugi.RegionGraph]
	if graph.BytesPerDoc != 30 || !graph.InBudget {
		t.Errorf("graph bytes/doc = %.0f inBudget=%v, want 30 and in budget", graph.BytesPerDoc, graph.InBudget)
	}

	// A full-document forward store keeps the body, which the search-only line does not
	// cover, so it is reported without a per-document verdict.
	fwd := byKind[tsumugi.RegionForward]
	if fwd.HasBudget {
		t.Errorf("full-store forward should carry no per-doc budget, got hasBudget=%v", fwd.HasBudget)
	}
	if fwd.BudgetVerdict() != "full-store" {
		t.Errorf("full-store forward verdict = %q, want full-store", fwd.BudgetVerdict())
	}
}

// TestRegionStatsSearchOnlyForward pins that the forward region's search-only budget
// applies once the shard carries FlagSearchOnly, the inverse of the full-store case: the
// same forward bytes that were reported without a verdict on a full store are now checked
// against the 200-400 search-only line.
func TestRegionStatsSearchOnlyForward(t *testing.T) {
	path := filepath.Join(t.TempDir(), "search.tsumugi")
	// 100 docs, forward 30000 B = 300 B/doc, inside the 200-400 search-only line.
	buildStatShard(t, path, 100, tsumugi.FlagSearchOnly, map[tsumugi.RegionKind]int{
		tsumugi.RegionForward: 30000,
	})
	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() { _ = r.Close() }()

	stats := r.RegionStats()
	if len(stats) != 1 {
		t.Fatalf("want 1 region stat, got %d", len(stats))
	}
	fwd := stats[0]
	if !fwd.HasBudget {
		t.Fatalf("search-only forward should carry the search-only budget")
	}
	if fwd.BytesPerDoc != 300 || !fwd.InBudget {
		t.Errorf("search-only forward %.0f bytes/doc inBudget=%v, want 300 and in budget", fwd.BytesPerDoc, fwd.InBudget)
	}
	if fwd.Budget != "200-400" {
		t.Errorf("search-only forward budget cell = %q, want 200-400", fwd.Budget)
	}
}

// TestRegionStatsRatioAndEmpty checks the two arithmetic edges: a compressed region
// reports its real on-disk-over-raw ratio, and a zero-document shard reports zero
// per-document cost rather than dividing by zero.
func TestRegionStatsRatioAndEmpty(t *testing.T) {
	// Compressed region: highly compressible zero bytes, so on-disk is far under raw.
	path := filepath.Join(t.TempDir(), "zstd.tsumugi")
	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	w.SetDocCount(10)
	if err := w.AddRegion(tsumugi.RegionGraph, tsumugi.CodecZstd, 0, 0, make([]byte, 40000)); err != nil {
		t.Fatalf("add graph: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() { _ = r.Close() }()
	s := r.RegionStats()[0]
	if s.Raw != 40000 {
		t.Errorf("raw = %d, want 40000", s.Raw)
	}
	if !(s.Ratio > 0 && s.Ratio < 1) {
		t.Errorf("compressed ratio = %.4f, want between 0 and 1", s.Ratio)
	}

	// Zero-document shard: per-document cost is reported as zero, not a divide by zero.
	empty := filepath.Join(t.TempDir(), "empty.tsumugi")
	buildStatShard(t, empty, 0, 0, map[tsumugi.RegionKind]int{tsumugi.RegionLexical: 100})
	er, err := tsumugi.Open(empty)
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	defer func() { _ = er.Close() }()
	if got := er.RegionStats()[0].BytesPerDoc; got != 0 {
		t.Errorf("empty shard bytes/doc = %.2f, want 0", got)
	}
}

// TestBudgetFor checks the exported budget lookup: a budgeted kind returns its doc 13
// table cell, and a kind with no per-document budget, like the dictionary, returns false.
func TestBudgetFor(t *testing.T) {
	if b, ok := tsumugi.BudgetFor(tsumugi.RegionLexical); !ok || b.Display != "120-260" {
		t.Errorf("lexical budget = %+v ok=%v, want the 120-260 cell", b, ok)
	}
	if _, ok := tsumugi.BudgetFor(tsumugi.RegionDictionary); ok {
		t.Errorf("dictionary should carry no per-document budget")
	}
}
