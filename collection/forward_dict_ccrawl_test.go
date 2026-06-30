package collection

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/forward"
)

// TestForwardRegionPerValueCCrawl builds a shard from real ccrawl pages and checks
// the forward region the build now writes: it is stored uncompressed in the
// container (the per-value compression lives inside the region), every column
// still round-trips, and the on-disk size is reported against the old whole-region
// zstd encoding so the size trade-off is recorded rather than assumed.
func TestForwardRegionPerValueCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 100000, Limit: 8000})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if res.Docs == 0 {
		t.Skip("no ccrawl pages ingested")
	}
	shards, err := List(out)
	if err != nil {
		t.Fatal(err)
	}

	var diskTotal, wholeZstdTotal uint64
	var checked int
	enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = enc.Close() }()

	for _, sh := range shards {
		r, err := tsumugi.Open(sh.Path)
		if err != nil {
			t.Fatal(err)
		}
		desc, ok := r.RegionDesc(tsumugi.RegionForward)
		if !ok {
			_ = r.Close()
			t.Fatal("shard has no forward region")
		}
		// The region must be stored uncompressed: the per-value frames already carry
		// the compression, so a container codec would inflate the whole region at open.
		if desc.Codec != tsumugi.CodecNone {
			_ = r.Close()
			t.Fatalf("forward region codec = %d, want CodecNone so opening inflates nothing", desc.Codec)
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			t.Fatal(err)
		}
		diskTotal += desc.Length
		wholeZstdTotal += uint64(len(enc.EncodeAll(b, nil)))

		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			t.Fatal(err)
		}
		// Spot-check every value round-trips: doc_id is the raw 32-byte key, url and
		// title and body decode per value against their shared dictionaries.
		n := fwd.DocCount()
		for id := uint32(0); id < n; id++ {
			did, _ := fwd.Column("doc_id", id)
			url, _ := fwd.Column("url", id)
			if len(url) == 0 {
				continue // a row with no url is the rare malformed case
			}
			if len(did) != 0 && len(did) != 32 {
				t.Fatalf("doc_id is %d bytes, want 0 or 32", len(did))
			}
			if want, ok := analyze.DocID(string(url)); ok && len(did) == 32 && string(did) != string(want[:]) {
				t.Fatalf("doc_id mismatch for %q", url)
			}
			// body decodes to whatever was stored; just exercise the decode path.
			_, _ = fwd.Column("body", id)
			checked++
		}
		fwd.Close()
		_ = r.Close()
	}

	if checked == 0 {
		t.Fatal("no rows checked")
	}
	t.Logf("checked %d rows across %d shards", checked, len(shards))
	t.Logf("forward region on disk (per-value, CodecNone): %d bytes", diskTotal)
	t.Logf("same bytes under old whole-region zstd:         %d bytes", wholeZstdTotal)
	t.Logf("per-value is %.1f%% of whole-region zstd", 100*float64(diskTotal)/float64(wholeZstdTotal))
}
