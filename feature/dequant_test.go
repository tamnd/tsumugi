package feature

import (
	"math"
	"testing"
)

// TestDequantMatchesRegion checks the constants Builder.Dequant hands out reproduce the
// region's own dequantization to the bit, so the copy written to the footer statistics
// is the same recipe the in-region reader applies. It builds the full schema with varied
// values, reads each document/column value back through the region, and confirms decoding
// the same stored level with the exported Dequant gives the identical float.
func TestDequantMatchesRegion(t *testing.T) {
	cols := DefaultSchema()
	b := NewBuilder(cols, SchemaVersion)
	const n = 64
	for d := uint32(0); d < n; d++ {
		for _, c := range cols {
			// A spread of magnitudes per quant scheme so log, linear, signed, and
			// categorical columns all exercise a real range rather than a constant.
			v := float64((int(d)*7+int(c.ID)*3)%50) + 0.25*float64(d)
			b.Set(d, c.ID, v)
		}
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	ds := b.Dequant()
	if len(ds) != len(cols) {
		t.Fatalf("Dequant returned %d recipes, want %d", len(ds), len(cols))
	}
	byID := map[FeatureID]Dequant{}
	for _, d := range ds {
		byID[d.ID] = d
	}
	for _, c := range cols {
		dq, ok := byID[c.ID]
		if !ok {
			t.Fatalf("no dequant recipe for column %d", c.ID)
		}
		for d := uint32(0); d < n; d++ {
			row, _ := r.Row(d)
			q := loadQuant(row, columnOffset(r, c.ID), c.Width)
			want, _ := r.Value(d, c.ID)
			got := dq.Decode(q)
			if math.Abs(got-want) > 1e-12 {
				t.Fatalf("col %d doc %d: footer decode %v, region value %v", c.ID, d, got, want)
			}
		}
	}
}

// TestDequantStatsRoundTrip writes the dequant block into a stats map and reads it back
// through ReadDequantStats, checking every column's constants survive the float64 stats
// round trip and that an absent column is omitted rather than read as zeros.
func TestDequantStatsRoundTrip(t *testing.T) {
	cols := DefaultSchema()
	b := NewBuilder(cols, SchemaVersion)
	for d := uint32(0); d < 32; d++ {
		for _, c := range cols {
			b.Set(d, c.ID, float64(d)+float64(c.ID))
		}
	}
	_ = b.Build()
	ds := b.Dequant()

	stats := map[string]float64{}
	WriteDequantStats(func(k string, v float64) { stats[k] = v }, ds)

	got := ReadDequantStats(stats, cols)
	if len(got) != len(ds) {
		t.Fatalf("read %d recipes, wrote %d", len(got), len(ds))
	}
	byID := map[FeatureID]Dequant{}
	for _, d := range got {
		byID[d.ID] = d
	}
	for _, d := range ds {
		r, ok := byID[d.ID]
		if !ok {
			t.Fatalf("column %d missing after round trip", d.ID)
		}
		if r.P0 != d.P0 || r.P1 != d.P1 || r.P2 != d.P2 || r.Quant != d.Quant || r.Width != d.Width {
			t.Fatalf("column %d: read %+v, wrote %+v", d.ID, r, d)
		}
	}

	// A schema column with no stored constants is omitted, not invented.
	delete(stats, DequantStatKey(cols[0].ID, 1))
	got = ReadDequantStats(stats, cols)
	for _, d := range got {
		if d.ID == cols[0].ID {
			t.Fatalf("column %d returned despite a missing constant", cols[0].ID)
		}
	}
}

// TestDequantBeforeBuild checks Dequant is nil before Build, the contract that the
// recipes are only meaningful once the per-column range has been computed.
func TestDequantBeforeBuild(t *testing.T) {
	b := NewBuilder(DefaultSchema(), SchemaVersion)
	if d := b.Dequant(); d != nil {
		t.Fatalf("Dequant before Build returned %d recipes, want nil", len(d))
	}
}

// columnOffset finds the byte offset of a column in a region's rows, so the test can
// read a stored level the same way the region does without exporting the offset.
func columnOffset(r *Region, id FeatureID) uint16 {
	for _, c := range r.cols {
		if c.ID == id {
			return c.offset
		}
	}
	return 0
}

// BenchmarkDequantDecode times the dequant of one stored level, the per-value work a
// footer-driven reader does. It is the cost of turning a quantized byte back into a real
// value once the recipe is in hand.
func BenchmarkDequantDecode(b *testing.B) {
	d := Dequant{ID: FeatPageRank, Width: 1, Quant: QuantLog, P0: -2, P1: 0.05, P2: 1}
	var sink float64
	for i := 0; i < b.N; i++ {
		sink += d.Decode(uint32(i & 0xff))
	}
	_ = sink
}
