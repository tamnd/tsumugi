package tsumugi

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// buildShard writes a small shard with two regions and a few stats, then returns
// its path.
func buildShard(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.tsumugi")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	w.SetDocCount(42)
	w.SetBuildEpoch(1_700_000_000)
	w.SetNodeBase(1000)
	w.SetSchema([]Field{{Name: "url", Type: 1}, {Name: "title", Type: 1, Nullable: true}})
	w.SetStat(StatTokenCount, 12345)
	w.SetStat(StatAvgDocLen, 293.5)

	lexical := bytes.Repeat([]byte("posting-data "), 500)
	if err := w.AddRegion(RegionLexical, CodecZstd, 0, 0, lexical); err != nil {
		t.Fatalf("AddRegion lexical: %v", err)
	}
	forward := []byte("url\x00title\x00snippet\x00")
	if err := w.AddRegion(RegionForward, CodecNone, 0, 0, forward); err != nil {
		t.Fatalf("AddRegion forward: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return path
}

func TestRoundTrip(t *testing.T) {
	path := buildShard(t)
	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = r.Close() }()

	if r.Header.DocCount != 42 {
		t.Errorf("DocCount = %d, want 42", r.Header.DocCount)
	}
	if r.Header.NodeBase != 1000 {
		t.Errorf("NodeBase = %d, want 1000", r.Header.NodeBase)
	}
	if r.Header.BuildEpoch != 1_700_000_000 {
		t.Errorf("BuildEpoch = %d", r.Header.BuildEpoch)
	}
	if !r.Header.Has(FlagHasLexical) || !r.Header.Has(FlagHasForward) {
		t.Errorf("flags missing region bits: %b", r.Header.Flags)
	}
	if r.Header.Has(FlagHasVector) {
		t.Errorf("vector flag set but no vector region")
	}

	if v, ok := r.Stat(StatDocCount); !ok || v != 42 {
		t.Errorf("StatDocCount = %v, %v", v, ok)
	}
	if v, ok := r.Stat(StatAvgDocLen); !ok || v != 293.5 {
		t.Errorf("StatAvgDocLen = %v, %v", v, ok)
	}
	if len(r.Footer.Schema) != 2 || r.Footer.Schema[0].Name != "url" || !r.Footer.Schema[1].Nullable {
		t.Errorf("schema round-trip wrong: %+v", r.Footer.Schema)
	}

	// Region bytes decompress back to the originals.
	lex, err := r.Region(RegionLexical)
	if err != nil {
		t.Fatalf("Region lexical: %v", err)
	}
	if !bytes.Equal(lex, bytes.Repeat([]byte("posting-data "), 500)) {
		t.Errorf("lexical region content mismatch")
	}
	fwd, err := r.Region(RegionForward)
	if err != nil {
		t.Fatalf("Region forward: %v", err)
	}
	if !bytes.Equal(fwd, []byte("url\x00title\x00snippet\x00")) {
		t.Errorf("forward region content mismatch")
	}

	// The zstd region should actually be smaller on disk than raw.
	d, _ := r.RegionDesc(RegionLexical)
	if d.Length >= d.RawLength {
		t.Errorf("zstd did not shrink lexical: on-disk %d raw %d", d.Length, d.RawLength)
	}
}

func TestOpenBytesMatchesOpen(t *testing.T) {
	path := buildShard(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r, err := OpenBytes(raw)
	if err != nil {
		t.Fatalf("OpenBytes: %v", err)
	}
	defer func() { _ = r.Close() }()
	if r.Header.DocCount != 42 {
		t.Errorf("DocCount = %d", r.Header.DocCount)
	}
}

func TestBadMagic(t *testing.T) {
	if _, err := OpenBytes(make([]byte, HeaderSize+TrailerSize)); err != ErrBadMagic {
		t.Errorf("err = %v, want ErrBadMagic", err)
	}
}

func TestShortFile(t *testing.T) {
	if _, err := OpenBytes([]byte("tiny")); err != ErrShortFile {
		t.Errorf("err = %v, want ErrShortFile", err)
	}
}

func TestTruncationRejected(t *testing.T) {
	path := buildShard(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Drop the last 8 bytes: the trailing magic is gone, so the file is rejected.
	if _, err := OpenBytes(raw[:len(raw)-8]); err == nil {
		t.Errorf("truncated file opened without error")
	}
}

func TestHeaderCorruptionCaught(t *testing.T) {
	path := buildShard(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	raw[16] ^= 0xff // flip a doc_count byte, inside the header CRC coverage
	if _, err := OpenBytes(raw); err != ErrHeaderCRC {
		t.Errorf("err = %v, want ErrHeaderCRC", err)
	}
}

func TestRegionCorruptionCaught(t *testing.T) {
	path := buildShard(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r, err := OpenBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	// Corrupt a byte inside the forward region payload.
	d, _ := r.RegionDesc(RegionForward)
	raw[d.Offset] ^= 0xff
	r2, err := OpenBytes(raw)
	if err != nil {
		t.Fatalf("OpenBytes after region corruption: %v", err)
	}
	if _, err := r2.Region(RegionForward); err != ErrRegionCRC {
		t.Errorf("err = %v, want ErrRegionCRC", err)
	}
}

func TestFooterCorruptionCaught(t *testing.T) {
	path := buildShard(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// The footer sits just before the trailer; flip a byte there.
	raw[len(raw)-TrailerSize-4] ^= 0xff
	if _, err := OpenBytes(raw); err != ErrFooterCRC {
		t.Errorf("err = %v, want ErrFooterCRC", err)
	}
}
