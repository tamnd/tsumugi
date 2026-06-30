package tsumugi

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// buildDictShard writes a shard that registers one shared dictionary and stores a
// region against it with CodecZstdDict, then returns its path and the region's
// raw bytes.
func buildDictShard(t *testing.T, dictID uint32, dict, payload []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "dict.tsumugi")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	w.SetDocCount(1)
	if err := w.AddDictionary(dictID, dict); err != nil {
		t.Fatalf("AddDictionary: %v", err)
	}
	if err := w.AddRegion(RegionForward, CodecZstdDict, 0, dictID, payload); err != nil {
		t.Fatalf("AddRegion zstd+dict: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return path
}

func TestDictionaryRoundTrip(t *testing.T) {
	dict := bytes.Repeat([]byte("https://example.com/path/segment "), 64)
	payload := bytes.Repeat([]byte("https://example.com/path/segment?q=1 "), 32)
	path := buildDictShard(t, 7, dict, payload)

	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = r.Close() }()

	if !r.Header.Has(FlagHasDictionary) {
		t.Errorf("FlagHasDictionary not set: %b", r.Header.Flags)
	}
	if !r.HasRegion(RegionDictionary) {
		t.Errorf("RegionDictionary missing")
	}
	desc, ok := r.RegionDesc(RegionForward)
	if !ok || desc.Codec != CodecZstdDict || desc.DictID != 7 {
		t.Fatalf("forward descriptor wrong: %+v", desc)
	}
	got, err := r.Region(RegionForward)
	if err != nil {
		t.Fatalf("Region forward: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("payload mismatch after dict round-trip")
	}
	if d, ok := r.Dictionary(7); !ok || !bytes.Equal(d, dict) {
		t.Errorf("Dictionary(7) = %q, %v", d, ok)
	}
	if desc.Length >= desc.RawLength {
		t.Errorf("zstd+dict did not shrink: on-disk %d raw %d", desc.Length, desc.RawLength)
	}
}

func TestDictionaryMultiple(t *testing.T) {
	dictA := bytes.Repeat([]byte("alpha-context "), 32)
	dictB := bytes.Repeat([]byte("beta-context "), 32)
	payA := bytes.Repeat([]byte("alpha-context value "), 16)
	payB := bytes.Repeat([]byte("beta-context value "), 16)

	path := filepath.Join(t.TempDir(), "multi.tsumugi")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	w.SetDocCount(2)
	if err := w.AddDictionary(11, dictA); err != nil {
		t.Fatalf("AddDictionary A: %v", err)
	}
	if err := w.AddDictionary(22, dictB); err != nil {
		t.Fatalf("AddDictionary B: %v", err)
	}
	if err := w.AddRegion(RegionForward, CodecZstdDict, 0, 11, payA); err != nil {
		t.Fatalf("AddRegion A: %v", err)
	}
	if err := w.AddRegion(RegionFeature, CodecZstdDict, 0, 22, payB); err != nil {
		t.Fatalf("AddRegion B: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = r.Close() }()

	gotA, err := r.Region(RegionForward)
	if err != nil {
		t.Fatalf("Region A: %v", err)
	}
	gotB, err := r.Region(RegionFeature)
	if err != nil {
		t.Fatalf("Region B: %v", err)
	}
	if !bytes.Equal(gotA, payA) {
		t.Errorf("region A mismatch under dict 11")
	}
	if !bytes.Equal(gotB, payB) {
		t.Errorf("region B mismatch under dict 22")
	}
}

// TestDictionaryBeatsPlainOnSharedContent proves the dictionary is actually
// referenced: a small payload whose bytes live in the dictionary compresses much
// smaller against it than the same payload with no dictionary, which has nothing
// to reference and pays full frame overhead.
func TestDictionaryBeatsPlainOnSharedContent(t *testing.T) {
	shared := []byte("the quick brown fox jumps over the lazy dog, and then keeps on running for a while")
	dict := bytes.Repeat(shared, 8)
	payload := shared

	dictPath := buildDictShard(t, 5, dict, payload)
	rd, err := Open(dictPath)
	if err != nil {
		t.Fatalf("Open dict shard: %v", err)
	}
	defer func() { _ = rd.Close() }()
	dictDesc, _ := rd.RegionDesc(RegionForward)

	plainPath := filepath.Join(t.TempDir(), "plain.tsumugi")
	w, err := Create(plainPath)
	if err != nil {
		t.Fatalf("Create plain: %v", err)
	}
	w.SetDocCount(1)
	if err := w.AddRegion(RegionForward, CodecZstd, 0, 0, payload); err != nil {
		t.Fatalf("AddRegion plain: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close plain: %v", err)
	}
	rp, err := Open(plainPath)
	if err != nil {
		t.Fatalf("Open plain shard: %v", err)
	}
	defer func() { _ = rp.Close() }()
	plainDesc, _ := rp.RegionDesc(RegionForward)

	if dictDesc.Length >= plainDesc.Length {
		t.Errorf("dict frame %d not smaller than plain frame %d on shared content",
			dictDesc.Length, plainDesc.Length)
	}
}

func TestDictionaryUnknownIDRejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.tsumugi")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer func() { _ = w.Close() }()
	if err := w.AddRegion(RegionForward, CodecZstdDict, 0, 99, []byte("data")); err == nil {
		t.Errorf("AddRegion with unregistered dict id did not error")
	}
}

func TestAddDictionaryValidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "v.tsumugi")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer func() { _ = w.Close() }()

	if err := w.AddDictionary(0, []byte("x")); err == nil {
		t.Errorf("dict id 0 was accepted")
	}
	if err := w.AddDictionary(1, nil); err == nil {
		t.Errorf("empty dict content was accepted")
	}
	if err := w.AddDictionary(1, []byte("x")); err != nil {
		t.Fatalf("first AddDictionary: %v", err)
	}
	if err := w.AddDictionary(1, []byte("y")); err == nil {
		t.Errorf("duplicate dict id was accepted")
	}
}

func TestDictionaryCorruptionCaught(t *testing.T) {
	dict := bytes.Repeat([]byte("context "), 32)
	payload := bytes.Repeat([]byte("context value "), 16)
	path := buildDictShard(t, 3, dict, payload)

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r, err := OpenBytes(raw)
	if err != nil {
		t.Fatalf("OpenBytes: %v", err)
	}
	desc, ok := r.RegionDesc(RegionDictionary)
	if !ok {
		t.Fatal("no dictionary region")
	}
	_ = r.Close()

	// Flip a byte inside the dictionary region; Open must catch it via the CRC.
	raw[desc.Offset] ^= 0xff
	if _, err := OpenBytes(raw); err != ErrRegionCRC {
		t.Errorf("corrupt dictionary region: got %v, want ErrRegionCRC", err)
	}
}

func TestDeriveDictionary(t *testing.T) {
	if DeriveDictionary(nil, 100) != nil {
		t.Errorf("DeriveDictionary(nil) should be nil")
	}
	if DeriveDictionary([][]byte{[]byte("a")}, 0) != nil {
		t.Errorf("DeriveDictionary(maxBytes=0) should be nil")
	}
	samples := [][]byte{[]byte("aaaa"), []byte("bbbb"), []byte("cccc")}
	d := DeriveDictionary(samples, 6)
	if len(d) != 6 {
		t.Errorf("DeriveDictionary bounded length = %d, want 6", len(d))
	}
	full := DeriveDictionary(samples, 1000)
	if len(full) != 12 {
		t.Errorf("DeriveDictionary full length = %d, want 12", len(full))
	}
}
