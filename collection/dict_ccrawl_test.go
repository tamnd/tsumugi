package collection

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
)

// readURLs pulls up to limit url values from the real crawl export. URLs are the
// canonical small, similar payloads the shared-dictionary mechanism targets: they
// share host, scheme, and path structure across a shard.
func readURLs(t *testing.T, limit int) [][]byte {
	t.Helper()
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()
	var urls [][]byte
	for len(urls) < limit {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		if d.URL == "" {
			continue
		}
		urls = append(urls, []byte(d.URL))
	}
	return urls
}

// TestDictionaryRegionRoundTripCCrawl writes a region of real, concatenated URL
// bytes against a dictionary derived from the same column and proves it round-trips
// byte-for-byte through the container's CodecZstdDict path and shrinks on disk. It
// exercises the shared-dictionary mechanism end to end on real data.
func TestDictionaryRegionRoundTripCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	urls := readURLs(t, 5000)
	if len(urls) < 100 {
		t.Skipf("only %d urls, too few to gate", len(urls))
	}

	var column []byte
	for _, u := range urls {
		column = append(column, u...)
		column = append(column, '\n')
	}
	dict := tsumugi.DeriveDictionary(urls, 16<<10)
	if dict == nil {
		t.Fatal("DeriveDictionary returned nil on real urls")
	}

	path := filepath.Join(t.TempDir(), "urls.tsumugi")
	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	w.SetDocCount(uint32(len(urls)))
	if err := w.AddDictionary(1, dict); err != nil {
		t.Fatalf("AddDictionary: %v", err)
	}
	if err := w.AddRegion(tsumugi.RegionForward, tsumugi.CodecZstdDict, 0, 1, column); err != nil {
		t.Fatalf("AddRegion: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = r.Close() }()

	got, err := r.Region(tsumugi.RegionForward)
	if err != nil {
		t.Fatalf("Region: %v", err)
	}
	if !bytes.Equal(got, column) {
		t.Fatalf("url column mismatch after dict round-trip")
	}
	desc, _ := r.RegionDesc(tsumugi.RegionForward)
	if desc.Codec != tsumugi.CodecZstdDict || desc.DictID != 1 {
		t.Errorf("descriptor wrong: %+v", desc)
	}
	if desc.Length >= desc.RawLength {
		t.Errorf("dict region did not shrink real urls: on-disk %d raw %d", desc.Length, desc.RawLength)
	}
	t.Logf("url column: raw=%d on-disk=%d urls=%d", desc.RawLength, desc.Length, len(urls))
}

// TestDictionaryPerValueWinCCrawl validates the payoff the forward per-value
// consumer (the next slice) builds on: compressing each small URL independently
// against a shared raw dictionary, with the same WithEncoderDictRaw primitive the
// container's CodecZstdDict path uses, beats compressing each one with no shared
// context. Per-value compression is what gives random access without inflating the
// shard; this proves the shared dictionary pays for that access on real data.
func TestDictionaryPerValueWinCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	urls := readURLs(t, 5000)
	if len(urls) < 100 {
		t.Skipf("only %d urls, too few to gate", len(urls))
	}
	dict := tsumugi.DeriveDictionary(urls, 16<<10)

	encDict, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.SpeedBetterCompression),
		zstd.WithEncoderDictRaw(1, dict))
	if err != nil {
		t.Fatalf("encDict: %v", err)
	}
	defer func() { _ = encDict.Close() }()
	encPlain, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	if err != nil {
		t.Fatalf("encPlain: %v", err)
	}
	defer func() { _ = encPlain.Close() }()
	decDict, err := zstd.NewReader(nil, zstd.WithDecoderDictRaw(1, dict))
	if err != nil {
		t.Fatalf("decDict: %v", err)
	}
	defer decDict.Close()

	var dictTotal, plainTotal int
	for _, u := range urls {
		cd := encDict.EncodeAll(u, nil)
		cp := encPlain.EncodeAll(u, nil)
		dictTotal += len(cd)
		plainTotal += len(cp)
		back, err := decDict.DecodeAll(cd, nil)
		if err != nil {
			t.Fatalf("per-value decode: %v", err)
		}
		if !bytes.Equal(back, u) {
			t.Fatalf("per-value round-trip mismatch")
		}
	}
	t.Logf("per-value urls=%d dict=%d plain=%d", len(urls), dictTotal, plainTotal)
	if dictTotal >= plainTotal {
		t.Errorf("per-value dict total %d not smaller than plain %d", dictTotal, plainTotal)
	}
}
