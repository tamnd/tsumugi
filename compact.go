package tsumugi

import "fmt"

// RegionBudget is the per-document on-disk byte ceiling for a region kind, the canon
// master table from spec 2067 doc 13 made executable. Display is the doc 13 table cell
// the inspect tool prints; Ceiling is the bytes-per-document above which a region is
// flagged as over budget, so a codec or quantization regression shows up as one row the
// way doc 03's CRC check names a corrupt region. This file is the single source the
// inspect --sizes budget column and the compactness benchmark both read, so the contract
// has one home rather than a number copied into two places that drift.
type RegionBudget struct {
	Display string  // doc 13 master-table cell, for example "120-260" or "~32"
	Ceiling float64 // on-disk bytes per document over this is flagged over budget
}

// regionBudgets is doc 13's master budget table keyed by region kind, the canon default
// configuration's per-document on-disk bytes: impact postings, a search-only forward
// store, a few-dozen-column feature schema, three-bits-per-edge graph, and a two-part
// quantized vector. The forward line is the search-only number; a full-document store
// keeps the body and is reported without a budget verdict, since the body is content the
// search-only budget was never meant to bound.
var regionBudgets = map[RegionKind]RegionBudget{
	RegionLexical: {Display: "120-260", Ceiling: 260},
	RegionForward: {Display: "200-400", Ceiling: 400}, // search-only; a full store adds the body
	RegionFeature: {Display: "~32", Ceiling: 48},      // ~32 canon, up to ~48 for a rich schema
	RegionGraph:   {Display: "15-40", Ceiling: 40},    // degree 20-50, both directions, 3 bits/edge
	RegionVector:  {Display: "~1024", Ceiling: 1100},  // two-part 1-bit + int8 rerank
}

// BudgetFor returns the doc 13 budget for a region kind and whether one is defined. The
// dictionary region has no per-document budget because its cost amortizes across the whole
// shard rather than scaling with the document count, so it is reported without a verdict.
func BudgetFor(kind RegionKind) (RegionBudget, bool) {
	b, ok := regionBudgets[kind]
	return b, ok
}

// RegionStat is one region's measured compactness: its on-disk and raw bytes, the
// compression ratio, the per-document on-disk cost, and the doc 13 budget it is checked
// against. It is what the inspect --sizes report prints and the compactness benchmark
// gates on, so the executable contract and the human report read the same numbers.
type RegionStat struct {
	Kind        RegionKind
	Codec       Codec
	OnDisk      uint64
	Raw         uint64
	Ratio       float64 // OnDisk/Raw, 1.0 when the region is stored uncompressed
	BytesPerDoc float64 // OnDisk/DocCount, 0 when the shard is empty
	Budget      string  // the doc 13 master-table cell, empty when none applies
	Ceiling     float64 // bytes-per-document ceiling, 0 when none applies
	HasBudget   bool    // a per-document budget applies to this region
	InBudget    bool    // BytesPerDoc is at or under Ceiling, meaningful when HasBudget
}

// RegionStats reports every region's compactness against the doc 13 budget, in the
// footer's region order. The per-document numbers divide on-disk bytes by the document
// count, so a zero-document shard reports zero per-document cost rather than dividing by
// zero. The forward region's budget is the search-only line, applied only when the shard
// carries FlagSearchOnly; a full-document store is reported with HasBudget false so it is
// never flagged against a search-only number it was never built to hit.
func (r *Reader) RegionStats() []RegionStat {
	docs := float64(r.Header.DocCount)
	searchOnly := r.Header.Flags&FlagSearchOnly != 0
	out := make([]RegionStat, 0, len(r.Footer.Regions))
	for _, d := range r.Footer.Regions {
		s := RegionStat{
			Kind:   d.Kind,
			Codec:  d.Codec,
			OnDisk: d.Length,
			Raw:    d.RawLength,
		}
		if d.RawLength > 0 {
			s.Ratio = float64(d.Length) / float64(d.RawLength)
		}
		if docs > 0 {
			s.BytesPerDoc = float64(d.Length) / docs
		}
		if b, ok := regionBudgets[d.Kind]; ok {
			// A full-document forward store keeps the body, which the search-only budget
			// does not cover, so it is reported without a verdict.
			if d.Kind == RegionForward && !searchOnly {
				s.Budget = "full-store"
			} else {
				s.Budget = b.Display
				s.Ceiling = b.Ceiling
				s.HasBudget = true
				s.InBudget = s.BytesPerDoc <= b.Ceiling
			}
		}
		out = append(out, s)
	}
	return out
}

// BudgetVerdict renders a region stat's budget column for a human report: the budget
// cell and an ok-or-over marker, or a bare dash for a region with no per-document budget.
func (s RegionStat) BudgetVerdict() string {
	if s.Budget == "" {
		return "-"
	}
	if !s.HasBudget {
		return s.Budget
	}
	if s.InBudget {
		return fmt.Sprintf("%s ok", s.Budget)
	}
	return fmt.Sprintf("%s OVER", s.Budget)
}
