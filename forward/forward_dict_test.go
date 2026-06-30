package forward

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/tamnd/tsumugi/codec"
)

// urlLike builds n url-like values that share a long common prefix, the shape the
// forward store's small text columns actually hold, so a shared dictionary has
// real context to work with.
func urlLike(n int) [][]byte {
	out := make([][]byte, n)
	for i := range out {
		out[i] = []byte(fmt.Sprintf("https://www.example.com/articles/2026/06/%06d/index.html", i))
	}
	return out
}

// TestPerValueDictRoundTrip stores a column of many similar values under a shared
// dictionary and reads every one back, the path collection's url and title take.
func TestPerValueDictRoundTrip(t *testing.T) {
	vals := urlLike(2000)
	b := NewBuilder([]Column{{Name: "url", Type: ColString, Codec: CodecZstdDict}})
	for i, v := range vals {
		b.Set(uint32(i), "url", v)
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()
	for i, v := range vals {
		got, ok := r.Column("url", uint32(i))
		if !ok || string(got) != string(v) {
			t.Fatalf("url[%d]: got %q ok=%v want %q", i, got, ok, v)
		}
	}
}

// TestPerValueDictBeatsRaw proves the dictionary column is the smaller encoding:
// the same similar values stored uncompressed against stored per-value against a
// shared dictionary, the size win doc 04 is after.
func TestPerValueDictBeatsRaw(t *testing.T) {
	vals := urlLike(5000)

	raw := NewBuilder([]Column{{Name: "url", Type: ColString, Codec: CodecNone}})
	dict := NewBuilder([]Column{{Name: "url", Type: ColString, Codec: CodecZstdDict}})
	for i, v := range vals {
		raw.Set(uint32(i), "url", v)
		dict.Set(uint32(i), "url", v)
	}
	rawSize := len(raw.Build())
	dictSize := len(dict.Build())
	if dictSize >= rawSize {
		t.Fatalf("dictionary column not smaller: dict=%d raw=%d", dictSize, rawSize)
	}
	t.Logf("url column: raw=%d dict=%d (%.1f%% of raw)", rawSize, dictSize, 100*float64(dictSize)/float64(rawSize))
}

// TestCodecZstdNoDict round-trips a CodecZstd column, the per-value frame path
// with no shared dictionary, including an unset cell.
func TestCodecZstdNoDict(t *testing.T) {
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstd}})
	b.Set(0, "body", []byte("a body long enough that zstd does something with it, repeated repeated repeated"))
	// docID 1 left unset.
	b.Set(2, "body", []byte("another body of similar length, repeated repeated repeated repeated content"))
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()
	if got, ok := r.Column("body", 0); !ok || string(got) != "a body long enough that zstd does something with it, repeated repeated repeated" {
		t.Fatalf("body[0]: %q ok=%v", got, ok)
	}
	if got, ok := r.Column("body", 1); !ok || len(got) != 0 {
		t.Fatalf("body[1]: %q ok=%v want empty", got, ok)
	}
	if got, ok := r.Column("body", 2); !ok || string(got) != "another body of similar length, repeated repeated repeated repeated content" {
		t.Fatalf("body[2]: %q ok=%v", got, ok)
	}
}

// TestColumnIntoMatchesColumn checks the buffer-reusing read returns exactly what
// Column returns, for every codec and an unset cell, when called with a nil buffer,
// a reused buffer, and an oversized buffer. This is the L2 path's read, so any drift
// from Column would feed the ranker different feature text than a display read shows.
func TestColumnIntoMatchesColumn(t *testing.T) {
	bodies := [][]byte{
		[]byte("the quick brown fox jumps over the lazy dog, repeated repeated repeated content"),
		nil, // unset cell
		[]byte("a different body of comparable length, with its own repeated repeated repeated tail"),
		[]byte(""), // explicitly empty
	}
	for _, cod := range []uint8{CodecNone, CodecZstd, CodecZstdDict, CodecZstdDictBlocked} {
		b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: cod}})
		for i, v := range bodies {
			if v != nil {
				b.Set(uint32(i), "body", v)
			}
		}
		r, err := Open(b.Build())
		if err != nil {
			t.Fatalf("codec %d: open: %v", cod, err)
		}
		var scratch []byte
		for i := range bodies {
			want, wok := r.Column("body", uint32(i))
			// nil scratch, then the carried scratch, both must match Column.
			got, keep, gok := r.ColumnInto("body", uint32(i), scratch)
			if gok != wok || string(got) != string(want) {
				t.Fatalf("codec %d doc %d: ColumnInto %q/%v != Column %q/%v", cod, i, got, gok, want, wok)
			}
			scratch = keep
		}
		// An unknown column and an out-of-range docID fail the same way as Column.
		if _, _, ok := r.ColumnInto("nope", 0, scratch); ok {
			t.Fatalf("codec %d: ColumnInto on unknown column reported ok", cod)
		}
		if _, _, ok := r.ColumnInto("body", uint32(len(bodies)), scratch); ok {
			t.Fatalf("codec %d: ColumnInto past the last doc reported ok", cod)
		}
		r.Close()
	}
}

// TestColumnIntoReuseDoesNotCorrupt reads many compressed values through one reused
// buffer and checks each read is correct at the moment it is read, the contract the
// L2 scan relies on: the returned slice is valid until the next reuse, and reuse
// never bleeds a previous value into a shorter one.
func TestColumnIntoReuseDoesNotCorrupt(t *testing.T) {
	vals := [][]byte{
		[]byte("a long value padded out so its decode buffer grows wide, repeated repeated repeated repeated"),
		[]byte("short one"),
		[]byte("medium length value, repeated repeated"),
		[]byte("x"),
	}
	b := NewBuilder([]Column{{Name: "body", Type: ColString, Codec: CodecZstdDict}})
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
		got, keep, ok := r.ColumnInto("body", uint32(i), scratch)
		if !ok || string(got) != string(v) {
			t.Fatalf("doc %d through reused buffer: got %q want %q", i, got, v)
		}
		scratch = keep // carry the grown buffer, including after the widest value
	}
}

// TestCloseIdempotent checks Close is safe to call twice and on a region with no
// compressed columns.
func TestCloseIdempotent(t *testing.T) {
	dictReg := NewBuilder([]Column{{Name: "url", Type: ColString, Codec: CodecZstdDict}})
	dictReg.Set(0, "url", []byte("https://a.example/x"))
	r, err := Open(dictReg.Build())
	if err != nil {
		t.Fatal(err)
	}
	r.Close()
	r.Close() // second close must not panic

	plain := NewBuilder([]Column{{Name: "url", Type: ColString, Codec: CodecNone}})
	plain.Set(0, "url", []byte("x"))
	r2, err := Open(plain.Build())
	if err != nil {
		t.Fatal(err)
	}
	r2.Close() // no decoder to release, must not panic
}

// TestDictRegionCorruptionRejected flips a column's dictionary offset past the end
// of the region and checks Open rejects it instead of reading out of bounds.
func TestDictRegionCorruptionRejected(t *testing.T) {
	b := NewBuilder([]Column{{Name: "url", Type: ColString, Codec: CodecZstdDict}})
	for i, v := range urlLike(64) {
		b.Set(uint32(i), "url", v)
	}
	good := b.Build()

	// The single column's descriptor sits right after the 16-byte header and the
	// 4-byte name length plus the 3-byte name "url". Its dictOff is the next-to-last
	// uint64 of the descriptor tail. Rather than hand-compute it, find the dictOff
	// by reparsing and rewrite it past the region end.
	r, err := Open(good)
	if err != nil {
		t.Fatalf("open good: %v", err)
	}
	dictOff := r.cols[0].dictOff
	r.Close()
	if dictOff == 0 {
		t.Fatal("expected a non-zero dictionary offset")
	}

	// Locate the dictOff field in the header bytes and overwrite it with a value
	// past the region so the in-range check fires.
	bad := append([]byte(nil), good...)
	off := 16 + 4 + len("url") + 1 + 1 + 4 + 8 + 8 // up to dictOff
	if codec.Uint64(bad[off:]) != dictOff {
		t.Fatalf("dictOff not where expected: header=%d parsed=%d", codec.Uint64(bad[off:]), dictOff)
	}
	binary.LittleEndian.PutUint64(bad[off:], uint64(len(bad))+1)
	// The header CRC now mismatches too, which is itself a rejection; both are fine.
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt dictionary offset accepted")
	}
}
