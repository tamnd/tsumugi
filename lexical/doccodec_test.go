package lexical

import (
	"reflect"
	"testing"

	"github.com/tamnd/tsumugi/codec"
)

// gapShapes is a set of gap streams covering the edges every codec has to handle:
// empty, a single small gap, a run of small gaps, and gaps at each byte width 1..4
// so StreamVByte's length codes are all exercised.
var gapShapes = [][]uint32{
	{},
	{1},
	{0, 0, 0},
	{1, 2, 3, 4, 5},
	{255, 256, 65535, 65536, 1<<24 - 1, 1 << 24, 1<<32 - 1},
	{7, 1 << 20, 3, 1 << 28, 9, 12, 1 << 8, 1 << 16},
}

// TestDocCodecRoundTrip encodes and decodes every gap shape through every codec and
// requires the decoded gaps to equal the originals. A nil and an empty slice both
// decode back to a zero-length result, which is what the block layer needs.
func TestDocCodecRoundTrip(t *testing.T) {
	for _, c := range blockCodecs {
		for si, gaps := range gapShapes {
			enc := c.dc.encode(nil, gaps)
			got, err := c.dc.decode(enc)
			if err != nil {
				t.Fatalf("%s shape %d decode: %v", c.name, si, err)
			}
			if len(gaps) == 0 {
				if len(got) != 0 {
					t.Fatalf("%s shape %d: want empty, got %v", c.name, si, got)
				}
				continue
			}
			if !reflect.DeepEqual(got, gaps) {
				t.Fatalf("%s shape %d round trip\n got=%v\nwant=%v", c.name, si, got, gaps)
			}
		}
	}
}

// TestVarintCodecIsRawLEB128 pins the varint codec to the exact byte stream the
// pre-codec format wrote: a plain LEB128 uvarint per gap with no framing. This is
// what lets a region built with the varint codec be byte-for-byte the old format.
func TestVarintCodecIsRawLEB128(t *testing.T) {
	gaps := []uint32{1, 130, 300, 70000, 1 << 25}
	got := varintCodec{}.encode(nil, gaps)
	var want []byte
	for _, g := range gaps {
		want = codec.AppendUvarint(want, uint64(g))
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("varint codec is not raw LEB128\n got=%v\nwant=%v", got, want)
	}
}

// TestStreamVByteRejectsTruncation checks the decode guards: a stream cut inside its
// control bytes or its data bytes is reported corrupt, not read out of bounds. The
// container CRC catches bit flips first, so this fires on a structurally short
// stream, and it must never panic.
func TestStreamVByteRejectsTruncation(t *testing.T) {
	full := streamVByteCodec{}.encode(nil, []uint32{1, 1 << 20, 3, 1 << 28, 9})
	for n := 1; n < len(full); n++ {
		// Every proper prefix is shorter than the full stream and should either
		// fail or, at worst, not panic; the trailing-bytes check makes a clean
		// prefix that stops mid-data corrupt.
		_, err := streamVByteCodec{}.decode(full[:n])
		if err == nil {
			t.Fatalf("truncation to %d bytes accepted, want corrupt", n)
		}
	}
}

// TestStreamVByteRejectsTrailingBytes checks that extra bytes after a complete gap
// stream are rejected, so a miscounted block length cannot silently decode a short
// list and leave the rest as garbage.
func TestStreamVByteRejectsTrailingBytes(t *testing.T) {
	enc := streamVByteCodec{}.encode(nil, []uint32{1, 2, 3})
	enc = append(enc, 0xff)
	if _, err := (streamVByteCodec{}).decode(enc); err == nil {
		t.Fatal("trailing byte accepted, want corrupt")
	}
}

// TestDocCodecDensityByGapScale measures bytes per gap for each codec on full
// 128-gap blocks across a range of gap magnitudes, the measurement that explains why
// varint is the default on a small shard yet PFor is worth carrying for a large one.
// Varint spends one byte for any gap under 128, so it is unbeatable while gaps stay
// small; once gaps grow past a byte (the regime a multi-million-doc shard's common
// terms live in) PFor's bit packing pulls ahead. The large-gap case is asserted so
// the crossover is a locked-in fact, not just a logged number.
func TestDocCodecDensityByGapScale(t *testing.T) {
	// block builds a full block of gaps clustered around magnitude m, deterministic so
	// the measurement is reproducible.
	block := func(m uint32) []uint32 {
		gaps := make([]uint32, blockSize)
		for i := range gaps {
			gaps[i] = m + uint32(i%7) // a little spread so a width choice is non-trivial
			if gaps[i] == 0 {
				gaps[i] = 1
			}
		}
		return gaps
	}
	for _, m := range []uint32{1, 8, 64, 200, 5000, 200000} {
		gaps := block(m)
		row := make(map[string]float64)
		for _, c := range blockCodecs {
			n := len(c.dc.encode(nil, gaps))
			row[c.name] = float64(n) / float64(len(gaps))
		}
		t.Logf("gap~%-7d bytes/gap: varint %.3f  streamvbyte %.3f  pfor %.3f",
			m, row["varint"], row["streamvbyte"], row["pfor"])
		if m >= 5000 && row["pfor"] >= row["varint"] {
			t.Fatalf("gap~%d: pfor %.3f should beat varint %.3f at large gaps", m, row["pfor"], row["varint"])
		}
	}
}

// TestPForRejectsTruncation checks the PFor decode guards: a stream cut inside its
// exception list or its packed-bits region is reported corrupt rather than read out
// of bounds, and never panics. The gaps span several widths so the chosen width
// leaves both a packed region and exceptions to truncate into.
func TestPForRejectsTruncation(t *testing.T) {
	full := pforCodec{}.encode(nil, []uint32{1, 1, 2, 1, 1 << 20, 1, 3, 1 << 28})
	for n := 1; n < len(full); n++ {
		if _, err := (pforCodec{}).decode(full[:n]); err == nil {
			t.Fatalf("truncation to %d bytes accepted, want corrupt", n)
		}
	}
}

// TestPForRejectsTrailingBytes checks that extra bytes after a complete PFor stream
// are rejected, so a miscounted block length cannot decode a short list and leave the
// rest as garbage.
func TestPForRejectsTrailingBytes(t *testing.T) {
	enc := pforCodec{}.encode(nil, []uint32{1, 2, 3, 4})
	enc = append(enc, 0xff)
	if _, err := (pforCodec{}).decode(enc); err == nil {
		t.Fatal("trailing byte accepted, want corrupt")
	}
}

// TestCodecByID maps each selector to a codec whose id round-trips, and refuses an
// unknown selector so a region from a newer format is rejected cleanly.
func TestCodecByID(t *testing.T) {
	for _, id := range []uint16{docCodecVarint, docCodecStreamVByte, docCodecPFor} {
		dc, err := codecByID(id)
		if err != nil {
			t.Fatalf("codecByID(%d): %v", id, err)
		}
		if dc.id() != id {
			t.Fatalf("codec id round trip: got %d want %d", dc.id(), id)
		}
	}
	if _, err := codecByID(0xffff); err == nil {
		t.Fatal("unknown codec id accepted, want error")
	}
}
