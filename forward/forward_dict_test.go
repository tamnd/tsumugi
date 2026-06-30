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
