package tsumugi

import (
	"math"
	"path/filepath"
	"testing"
)

// TestAnalyzerHashStatRoundTrip checks that a 64-bit analyzer hash survives the float64
// stats map unchanged, including values above 2^53 where a float64 can no longer hold
// every integer: the encoding carries the bit pattern, not the numeric value, so the
// usual float precision ceiling does not apply.
func TestAnalyzerHashStatRoundTrip(t *testing.T) {
	cases := []uint64{
		0,
		1,
		1 << 53,
		(1 << 53) + 1,
		math.MaxUint64,
		0xDEADBEEFCAFEF00D,
		0xFFFFFFFFFFFFFFFE,
	}
	for _, h := range cases {
		got := AnalyzerHashFromStat(AnalyzerHashStat(h))
		if got != h {
			t.Fatalf("round trip of %#016x = %#016x, want exact", h, got)
		}
	}
}

// TestAnalyzerHashFooterRoundTrip writes a shard carrying an analyzer hash, reopens it,
// and checks the reader recovers the exact value. A hash above 2^53 proves the footer's
// stats map carries the bit pattern losslessly, not a rounded number.
func TestAnalyzerHashFooterRoundTrip(t *testing.T) {
	const want uint64 = 0xA1B2C3D4E5F60718 // above 2^53, not representable as a float64 integer
	dir := t.TempDir()
	path := filepath.Join(dir, "h.tsumugi")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	w.SetDocCount(1)
	w.SetAnalyzerHash(want)
	if err := w.AddRegion(RegionLexical, CodecNone, 0, 0, []byte("x")); err != nil {
		t.Fatalf("AddRegion: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = r.Close() }()

	got, ok := r.AnalyzerHash()
	if !ok {
		t.Fatalf("AnalyzerHash absent, want present")
	}
	if got != want {
		t.Fatalf("AnalyzerHash = %#016x, want %#016x", got, want)
	}
}

// TestAnalyzerHashAbsent checks a shard built without an analyzer hash reports it absent
// rather than as a zero hash, the unknown case a broker treats as a skipped check.
func TestAnalyzerHashAbsent(t *testing.T) {
	r, err := Open(buildShard(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = r.Close() }()

	if _, ok := r.AnalyzerHash(); ok {
		t.Fatalf("AnalyzerHash present on a shard that recorded none")
	}
}
