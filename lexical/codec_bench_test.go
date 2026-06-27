package lexical_test

import (
	"reflect"
	"testing"

	"github.com/tamnd/tsumugi/lexical"
)

// The docID gap codec is a build-time choice recorded in the region header, so a
// region built with either codec must serve the same results: the codec changes how
// the gap stream is packed, not what it decodes to. These tests prove that parity on
// real crawl vocabulary and measure the size and decode-speed trade between the two,
// the numbers the M14b spec row carries.

// buildWithCodec builds an in-memory region over docs with a chosen gap codec.
func buildWithCodec(docs []spimiDoc, codecID uint16) []byte {
	b := lexical.NewBuilder(lexical.DefaultParams()).WithDocCodec(codecID)
	for i, d := range docs {
		b.AddDoc(uint32(i), d.fields())
	}
	return b.Build()
}

// codecQueries are multi-term queries over the crawl vocabulary, chosen to touch
// common terms (long, dense lists where pruning and block decoding both matter).
var codecQueries = []string{
	"the of and to",
	"web page content site",
	"data search index document",
	"one two three four five",
}

// codecVariants is every selectable gap codec, so the parity, size, and speed
// measurements cover each one and a new codec is measured the moment it is added.
var codecVariants = []struct {
	name string
	id   uint16
}{
	{"varint", lexical.CodecVarint},
	{"streamvbyte", lexical.CodecStreamVByte},
	{"pfor", lexical.CodecPFor},
}

// TestDocCodecSearchParity builds the same crawl corpus with every codec and requires
// each to return the same results the varint baseline does, so a codec is proven to
// change only the packing, never the answer.
func TestDocCodecSearchParity(t *testing.T) {
	docs := ccrawlSpimiDocs(t, 4000)
	if len(docs) == 0 {
		t.Skip("no ccrawl documents")
	}
	base, err := lexical.Open(buildWithCodec(docs, lexical.CodecVarint))
	if err != nil {
		t.Fatalf("open varint region: %v", err)
	}
	for _, c := range codecVariants {
		region, err := lexical.Open(buildWithCodec(docs, c.id))
		if err != nil {
			t.Fatalf("open %s region: %v", c.name, err)
		}
		for _, q := range codecQueries {
			want, err := base.Search(q, 20)
			if err != nil {
				t.Fatalf("varint search %q: %v", q, err)
			}
			got, err := region.Search(q, 20)
			if err != nil {
				t.Fatalf("%s search %q: %v", c.name, q, err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("query %q: %s disagrees with varint\n varint=%v\n %s=%v", q, c.name, want, c.name, got)
			}
		}
	}
}

// TestDocCodecSize reports the region size each codec produces on the real crawl
// shard. It is the density half of the codec numbers; the gap stream is the only part
// that differs between builds, so the size delta is the gap-stream delta.
func TestDocCodecSize(t *testing.T) {
	docs := ccrawlSpimiDocs(t, 8000)
	if len(docs) == 0 {
		t.Skip("no ccrawl documents")
	}
	base := len(buildWithCodec(docs, lexical.CodecVarint))
	for _, c := range codecVariants {
		n := len(buildWithCodec(docs, c.id))
		delta := float64(n-base) / float64(base) * 100
		t.Logf("region size over %d docs: %-12s %d bytes (%+.2f%% vs varint)", len(docs), c.name, n, delta)
	}
}

// BenchmarkDocCodecSearch times the query path under each codec on the real crawl
// shard, the decode-speed half of the M14b numbers: the gap codec sits in the hot
// block-decode loop, so this is where StreamVByte earns its place.
func BenchmarkDocCodecSearch(b *testing.B) {
	docs := ccrawlSpimiDocs(b, 8000)
	if len(docs) == 0 {
		b.Skip("no ccrawl documents")
	}
	for _, c := range codecVariants {
		region, err := lexical.Open(buildWithCodec(docs, c.id))
		if err != nil {
			b.Fatalf("open %s region: %v", c.name, err)
		}
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, q := range codecQueries {
					if _, err := region.Search(q, 20); err != nil {
						b.Fatalf("search %q: %v", q, err)
					}
				}
			}
		})
	}
}

// BenchmarkDocCodecExhaustive times a full decode of every posting under each codec
// by running the exhaustive scan path, isolating raw gap-decode throughput from the
// pruning the WAND path layers on top.
func BenchmarkDocCodecExhaustive(b *testing.B) {
	docs := ccrawlSpimiDocs(b, 8000)
	if len(docs) == 0 {
		b.Skip("no ccrawl documents")
	}
	for _, c := range codecVariants {
		region, err := lexical.Open(buildWithCodec(docs, c.id))
		if err != nil {
			b.Fatalf("open %s region: %v", c.name, err)
		}
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, q := range codecQueries {
					if _, err := region.SearchExhaustive(q, 20); err != nil {
						b.Fatalf("exhaustive %q: %v", q, err)
					}
				}
			}
		})
	}
}
