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

// TestCodecByID maps each selector to a codec whose id round-trips, and refuses an
// unknown selector so a region from a newer format is rejected cleanly.
func TestCodecByID(t *testing.T) {
	for _, id := range []uint16{docCodecVarint, docCodecStreamVByte} {
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
