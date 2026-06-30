package forward

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
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
