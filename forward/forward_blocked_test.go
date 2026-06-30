package forward

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

// blockedBody builds a deterministic body of about n bytes whose content varies
// along its length, so a prefix check that passes is checking real position
// alignment rather than a run of identical bytes that would match at any offset.
func blockedBody(n int) []byte {
	var sb strings.Builder
	for i := 0; sb.Len() < n; i++ {
		fmt.Fprintf(&sb, "segment %06d the quick brown fox jumps over the lazy dog ", i)
	}
	return []byte(sb.String()[:n])
}

// TestBlockedRoundTrip checks a CodecZstdDictBlocked column reads back byte-for-byte
// through the full-value path for values spanning zero, one, and many blocks, so the
// block split and the concatenating decode reproduce the value exactly. The full
// decode is what Column, Row, and the compact and train paths read, so any drift here
// would corrupt a display row or a derived feature.
func TestBlockedRoundTrip(t *testing.T) {
	vals := [][]byte{
		blockedBody(1),                  // a single tiny block
		blockedBody(bodyBlockBytes - 1), // just under one block
		blockedBody(bodyBlockBytes),     // exactly one block
		blockedBody(bodyBlockBytes + 1), // just over one block, two blocks
		blockedBody(bodyBlockBytes * 5), // many full blocks
		blockedBody(200000),             // a large body, the windowed case
		nil,                             // unset cell
		[]byte(""),                      // explicitly empty
	}
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	for i, v := range vals {
		if v != nil {
			b.Set(uint32(i), "body", v)
		}
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()
	for i, v := range vals {
		got, ok := r.Column("body", uint32(i))
		if !ok {
			t.Fatalf("doc %d: Column not ok", i)
		}
		if !bytes.Equal(got, v) {
			t.Fatalf("doc %d: full decode len %d != want len %d", i, len(got), len(v))
		}
	}
}

// TestBlockedPrefixMatchesFull is the windowed-read contract: the prefix a
// ColumnPrefixInto returns is byte-for-byte the start of the full value, it holds at
// least minBytes bytes when the value is that long, and it stops at a block boundary
// so a large value decodes only the blocks the window spans rather than all of them.
// This is the L2 body scan's read, so a prefix that drifted from the full value would
// feed the ranker different leading text than a full decode holds.
func TestBlockedPrefixMatchesFull(t *testing.T) {
	full := blockedBody(200000)
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	b.Set(0, "body", full)
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	for _, minBytes := range []int{1, bodyBlockBytes - 1, bodyBlockBytes, bodyBlockBytes + 1, bodyBlockBytes*3 + 7, 1 << 30} {
		got, _, ok := r.ColumnPrefixInto("body", 0, minBytes, nil)
		if !ok {
			t.Fatalf("minBytes %d: not ok", minBytes)
		}
		if !bytes.Equal(got, full[:len(got)]) {
			t.Fatalf("minBytes %d: prefix is not the start of the full value", minBytes)
		}
		// The decode stops at the first block boundary at or past minBytes, so the prefix
		// holds at least minBytes bytes (unless the whole value is shorter) and never the
		// whole 200000-byte body for a small window.
		if len(got) < minBytes && len(got) < len(full) {
			t.Fatalf("minBytes %d: prefix len %d holds fewer than minBytes and is not the whole value", minBytes, len(got))
		}
		wantBlocks := (minBytes + bodyBlockBytes - 1) / bodyBlockBytes
		wantLen := wantBlocks * bodyBlockBytes
		if wantLen > len(full) {
			wantLen = len(full)
		}
		if len(got) != wantLen {
			t.Fatalf("minBytes %d: prefix len %d, want %d (block-aligned)", minBytes, len(got), wantLen)
		}
	}
}

// TestBlockedPrefixDecodesFewerBlocks pins the saving the codec exists for: a small
// window over a large body decodes far fewer bytes than the full value, so the L2
// body-decompression cost is the window, not the whole document.
func TestBlockedPrefixDecodesFewerBlocks(t *testing.T) {
	full := blockedBody(bodyBlockBytes * 20) // 20 blocks
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	b.Set(0, "body", full)
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	got, _, ok := r.ColumnPrefixInto("body", 0, bodyBlockBytes, nil)
	if !ok {
		t.Fatal("not ok")
	}
	if len(got) != bodyBlockBytes {
		t.Fatalf("a one-block window decoded %d bytes, want %d", len(got), bodyBlockBytes)
	}
	if len(got) >= len(full) {
		t.Fatalf("windowed decode did not shrink the work: got %d of %d", len(got), len(full))
	}
}

// TestPrefixFallbackOnNonBlocked checks ColumnPrefixInto on a column that is not
// block-structured returns the whole value, the fallback that keeps the caller uniform
// across codecs and across an old region whose body column predates the blocked codec.
func TestPrefixFallbackOnNonBlocked(t *testing.T) {
	full := blockedBody(200000)
	for _, cod := range []uint8{CodecNone, CodecZstd, CodecZstdDict} {
		b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: cod}})
		b.Set(0, "body", full)
		r, err := Open(b.Build())
		if err != nil {
			t.Fatalf("codec %d: open: %v", cod, err)
		}
		got, _, ok := r.ColumnPrefixInto("body", 0, bodyBlockBytes, nil)
		if !ok || !bytes.Equal(got, full) {
			t.Fatalf("codec %d: prefix fallback did not return the whole value (got %d of %d, ok %v)", cod, len(got), len(full), ok)
		}
		r.Close()
	}
}

// multibyteBody builds a body of nrunes three-byte CJK runes whose code point varies
// along its length. Three bytes per rune means the byte length is 3*nrunes and a block
// boundary at bodyBlockBytes (16384, not a multiple of three) splits a rune, the case
// the rune-aware decode's incomplete-trailing handling exists for.
func multibyteBody(nrunes int) []byte {
	var sb strings.Builder
	for i := 0; i < nrunes; i++ {
		sb.WriteRune(rune(0x4E00 + (i % 4096)))
	}
	return []byte(sb.String())
}

// firstRunes returns the first n runes a range-over-string scan reads from b, the same
// walk scanField runs over the body window. It is the independent oracle the rune-window
// tests check against: the windowed prefix must yield the same leading runes a full
// decode does, including never substituting a replacement rune where the full value has
// a real one (the mid-rune-truncation failure).
func firstRunes(b []byte, n int) []rune {
	out := make([]rune, 0, n)
	for _, r := range string(b) {
		if len(out) == n {
			break
		}
		out = append(out, r)
	}
	return out
}

// TestCompleteRunesAtLeast pins the rune counter the windowed decode stops on: it counts
// an invalid byte as the width-one replacement rune a range loop yields for it, but does
// not count a trailing byte sequence that only begins a multibyte rune, since the next
// block completes that rune and counting it would stop one rune short of what the scan
// reads.
func TestCompleteRunesAtLeast(t *testing.T) {
	han := string(rune(0x4E2D)) // 中, three bytes
	cases := []struct {
		name string
		b    []byte
		n    int
		want bool
	}{
		{"ascii enough", []byte("hello"), 5, true},
		{"ascii short", []byte("hello"), 6, false},
		{"ascii under", []byte("hello"), 3, true},
		{"multibyte complete", []byte("a" + han + "b"), 3, true},
		{"multibyte complete short", []byte("a" + han + "b"), 4, false},
		{"trailing incomplete not counted", []byte(han)[:2], 1, false},
		{"trailing incomplete zero ok", []byte(han)[:2], 0, true},
		{"invalid byte counts", []byte{0xFF, 'a'}, 2, true},
		{"invalid byte short", []byte{0xFF, 'a'}, 3, false},
		{"ascii then incomplete", append([]byte("a"), []byte(han)[:1]...), 1, true},
		{"ascii then incomplete short", append([]byte("a"), []byte(han)[:1]...), 2, false},
		{"empty", nil, 1, false},
	}
	for _, c := range cases {
		if got := completeRunesAtLeast(c.b, c.n); got != c.want {
			t.Errorf("%s: completeRunesAtLeast(%v, %d) = %v, want %v", c.name, c.b, c.n, got, c.want)
		}
	}
}

// TestBlockedPrefixRunesMatchesFull is the rune-window contract: the prefix is byte-for-
// byte the start of the full value, the first minRunes runes a scan reads from it equal
// the first minRunes runes of the full value (so a block boundary that splits a rune
// never feeds the scan a replacement rune where the full value has a real one), and the
// decode stops at a block boundary. The body is multibyte so a boundary genuinely splits
// a rune and the rune window and a byte window differ.
func TestBlockedPrefixRunesMatchesFull(t *testing.T) {
	const nrunes = 40000 // 120000 bytes, between 7 and 8 blocks
	full := multibyteBody(nrunes)
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	b.Set(0, "body", full)
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	// 5461 and 5462 straddle the first block boundary: 16384 bytes is 5461 whole runes
	// plus one byte of the 5462nd, so a window of 5461 must satisfy in one block and 5462
	// must take a second.
	for _, minRunes := range []int{1, 100, 5461, 5462, 10000, nrunes, nrunes + 1} {
		got, _, ok := r.ColumnPrefixRunesInto("body", 0, minRunes, nil)
		if !ok {
			t.Fatalf("minRunes %d: not ok", minRunes)
		}
		if !bytes.Equal(got, full[:len(got)]) {
			t.Fatalf("minRunes %d: prefix is not the start of the full value", minRunes)
		}
		want := minRunes
		if want > nrunes {
			want = nrunes
		}
		gotRunes := firstRunes(got, want)
		wantRunes := firstRunes(full, want)
		if string(gotRunes) != string(wantRunes) {
			t.Fatalf("minRunes %d: first %d runes of prefix differ from the full value", minRunes, want)
		}
		if len(gotRunes) < want {
			t.Fatalf("minRunes %d: prefix yields %d complete runes, fewer than %d", minRunes, len(gotRunes), want)
		}
		if len(got)%bodyBlockBytes != 0 && len(got) != len(full) {
			t.Fatalf("minRunes %d: prefix len %d is not block-aligned and not the whole value", minRunes, len(got))
		}
	}
}

// TestBlockedPrefixRunesASCIIDecodesOneBlock pins the win the rune window exists for: on
// an all-ASCII body, where a rune is one byte, a 10000-rune window decodes a single 16 KB
// block, where the byte window (the rune cap times the worst-case four bytes a rune takes)
// would decode three.
func TestBlockedPrefixRunesASCIIDecodesOneBlock(t *testing.T) {
	full := blockedBody(bodyBlockBytes * 20) // ASCII, 20 blocks
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	b.Set(0, "body", full)
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	gotRunes, _, ok := r.ColumnPrefixRunesInto("body", 0, 10000, nil)
	if !ok {
		t.Fatal("rune window: not ok")
	}
	if len(gotRunes) != bodyBlockBytes {
		t.Fatalf("ascii 10000-rune window decoded %d bytes, want one block %d", len(gotRunes), bodyBlockBytes)
	}
	// The byte window over the same body decodes the worst-case three blocks; the rune
	// window decoding one is the saving.
	gotBytes, _, ok := r.ColumnPrefixInto("body", 0, 10000*utf8.UTFMax, nil)
	if !ok {
		t.Fatal("byte window: not ok")
	}
	if len(gotBytes) <= len(gotRunes) {
		t.Fatalf("rune window (%d) did not decode fewer bytes than the byte window (%d)", len(gotRunes), len(gotBytes))
	}
	if !bytes.Equal(gotRunes, full[:len(gotRunes)]) {
		t.Fatal("rune-window prefix is not the start of the full value")
	}
}

// TestPrefixRunesFallbackOnNonBlocked checks ColumnPrefixRunesInto returns the whole
// value for a non-blocked codec, the same uniform-caller fallback the byte window keeps.
func TestPrefixRunesFallbackOnNonBlocked(t *testing.T) {
	full := blockedBody(200000)
	for _, cod := range []uint8{CodecNone, CodecZstd, CodecZstdDict} {
		b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: cod}})
		b.Set(0, "body", full)
		r, err := Open(b.Build())
		if err != nil {
			t.Fatalf("codec %d: open: %v", cod, err)
		}
		got, _, ok := r.ColumnPrefixRunesInto("body", 0, 10000, nil)
		if !ok || !bytes.Equal(got, full) {
			t.Fatalf("codec %d: rune-window fallback did not return the whole value (got %d of %d, ok %v)", cod, len(got), len(full), ok)
		}
		r.Close()
	}
}

// BenchmarkBlockedWindow measures the decode the rune window saves over the byte window
// on a large all-ASCII body, the common case: the byte window decodes the rune cap times
// the worst-case four bytes a rune takes (three 16 KB blocks), the rune window decodes
// only the blocks that hold the rune cap in complete runes (one block for ASCII). The
// numbers are per-read decode cost, the per-survivor cost the L2 stage pays.
func BenchmarkBlockedWindow(b *testing.B) {
	full := blockedBody(bodyBlockBytes * 20) // ASCII, 20 blocks
	bd := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	bd.Set(0, "body", full)
	r, err := Open(bd.Build())
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer r.Close()

	b.Run("byte-window", func(b *testing.B) {
		var scratch []byte
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var v []byte
			v, scratch, _ = r.ColumnPrefixInto("body", 0, 10000*utf8.UTFMax, scratch)
			b.SetBytes(int64(len(v)))
		}
	})
	b.Run("rune-window", func(b *testing.B) {
		var scratch []byte
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var v []byte
			v, scratch, _ = r.ColumnPrefixRunesInto("body", 0, 10000, scratch)
			b.SetBytes(int64(len(v)))
		}
	})
}

// TestBlockedPrefixReuseDoesNotCorrupt reads many blocked values through one reused
// buffer, alternating the window width, and checks each prefix is correct at the
// moment it is read, the contract the L2 scan relies on with its carried buffer.
func TestBlockedPrefixReuseDoesNotCorrupt(t *testing.T) {
	vals := [][]byte{
		blockedBody(bodyBlockBytes * 8),
		blockedBody(50),
		blockedBody(bodyBlockBytes*3 + 11),
		blockedBody(1),
	}
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDictBlocked}})
	for i, v := range vals {
		b.Set(uint32(i), "body", v)
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()
	var scratch []byte
	for i, v := range vals {
		got, keep, ok := r.ColumnPrefixInto("body", uint32(i), bodyBlockBytes, scratch)
		if !ok {
			t.Fatalf("doc %d: not ok", i)
		}
		if !bytes.Equal(got, v[:len(got)]) {
			t.Fatalf("doc %d through reused buffer: prefix is not the start of the value", i)
		}
		scratch = keep
	}
}
