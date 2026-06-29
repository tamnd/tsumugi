package feature

import (
	"math"
	"strconv"
)

// Dequant is the dequantization recipe for one feature column: its id, width, and
// quantization scheme plus the three constants the build derived from the column's
// observed range. It is what turns a stored quantized level back into a real value.
// The build embeds it in the feature region header for the in-region reader and also
// writes it into the shard footer statistics, the container-level contract doc 03
// names, so a reader that holds only the footer can dequantize without decoding the
// feature region.
type Dequant struct {
	ID    FeatureID
	Width uint8
	Quant Quant
	P0    float32
	P1    float32
	P2    float32
}

// Decode turns a stored quantized level back into the real value, the same formula
// the in-region reader applies, so a value dequantized from the footer constants and
// one read straight from the region agree to the bit.
func (d Dequant) Decode(q uint32) float64 {
	switch d.Quant {
	case QuantCategorical:
		// The stored byte is the code, returned verbatim.
		return float64(q)
	case QuantLog:
		return math.Exp(float64(d.P0)+float64(q)*float64(d.P1)) - float64(d.P2)
	case QuantSigned:
		return (float64(q) - math.Round(maxLevel(d.Width)/2)) * float64(d.P0)
	default:
		return float64(d.P0) + float64(q)*float64(d.P1)
	}
}

// Dequant returns the per-column dequantization recipes the last Build derived, in
// column order. It is valid only after Build, which is where the per-column range and
// its constants are computed; before that it returns nil. This is how the collection
// build hands the constants to the footer statistics without re-deriving them.
func (b *Builder) Dequant() []Dequant {
	if b.lay == nil {
		return nil
	}
	out := make([]Dequant, len(b.lay))
	for i, l := range b.lay {
		out[i] = Dequant{ID: l.ID, Width: l.Width, Quant: l.Quant, P0: l.p0, P1: l.p1, P2: l.p2}
	}
	return out
}

// Dequant returns the per-column dequantization recipes the region carries in its own
// header, in row order. It is the in-region copy a reader compares against the footer
// statistics block, and the recipe a footer-less reader falls back to.
func (r *Region) Dequant() []Dequant {
	out := make([]Dequant, len(r.cols))
	for i, c := range r.cols {
		out[i] = Dequant{ID: c.ID, Width: c.Width, Quant: c.Quant, P0: c.p0, P1: c.p1, P2: c.p2}
	}
	return out
}

// dequantStatPrefix namespaces the per-column dequant constants among the shard footer
// statistics so they do not collide with the scalar stat keys.
const dequantStatPrefix = "feature_dequant."

// DequantStatKey is the footer-statistics key holding constant p (0, 1, or 2) of the
// feature column with the given id. The keys are stable strings so the block survives
// the footer's deterministic key-sorted encoding.
func DequantStatKey(id FeatureID, p int) string {
	return dequantStatPrefix + strconv.Itoa(int(id)) + ".p" + strconv.Itoa(p)
}

// WriteDequantStats writes a build's per-column dequant constants into the shard footer
// statistics through set, the doc-08 dequant block carried in the doc-03 statistics
// section. Only the three constants are stored per column; the quant scheme and width
// are recovered from the schema when the block is read, since they are fixed by the
// schema version the region already records.
func WriteDequantStats(set func(key string, v float64), ds []Dequant) {
	for _, d := range ds {
		set(DequantStatKey(d.ID, 0), float64(d.P0))
		set(DequantStatKey(d.ID, 1), float64(d.P1))
		set(DequantStatKey(d.ID, 2), float64(d.P2))
	}
}

// ReadDequantStats reconstructs the per-column dequant recipes from a shard's footer
// statistics, pairing each schema column with the three constants the build stored.
// A column whose constants are absent (a shard built before the block was written, or
// a schema column this build did not produce) is omitted, so a caller can tell a
// present recipe from a missing one rather than reading a zero as a real constant. The
// float32 constants round-trip exactly through the footer's float64 storage.
func ReadDequantStats(stats map[string]float64, schema []Column) []Dequant {
	out := make([]Dequant, 0, len(schema))
	for _, c := range schema {
		p0, ok0 := stats[DequantStatKey(c.ID, 0)]
		p1, ok1 := stats[DequantStatKey(c.ID, 1)]
		p2, ok2 := stats[DequantStatKey(c.ID, 2)]
		if !ok0 || !ok1 || !ok2 {
			continue
		}
		out = append(out, Dequant{
			ID: c.ID, Width: c.Width, Quant: c.Quant,
			P0: float32(p0), P1: float32(p1), P2: float32(p2),
		})
	}
	return out
}
