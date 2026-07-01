package collection_test

import (
	"os"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/vector"
)

// BenchmarkDenseRegionBuildCCrawl measures the cost the dense-enabled build adds per shard:
// embedding a shard's worth of real crawl bodies with the default static encoder and
// framing the quantized vector region. It reports the region size and the per-document
// encode-and-add cost, the two numbers that decide whether turning the dense plane on fits
// the disk and build budget at the hundred-thousand-shard scale. The HNSW build over the
// shard's vectors dominates, so the per-op figure is amortized over the whole shard.
func BenchmarkDenseRegionBuildCCrawl(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		b.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	const shard = 1000
	bodies := make([][]string, 0, shard)
	for len(bodies) < shard {
		d, ok, err := src.Next()
		if err != nil {
			b.Fatalf("read doc: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		bodies = append(bodies, lexical.Analyze(d.Body))
	}
	if len(bodies) == 0 {
		b.Skip("no bodies read from ccrawl")
	}

	enc := dense.NewDefault(denseDim)
	var regionBytes int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vb := vector.NewBuilder(denseDim)
		for _, terms := range bodies {
			vb.Add(enc.Encode(terms))
		}
		out, err := vb.Build()
		if err != nil {
			b.Fatalf("build vector region: %v", err)
		}
		regionBytes = len(out)
	}
	b.StopTimer()

	b.ReportMetric(float64(regionBytes), "region_bytes")
	b.ReportMetric(float64(regionBytes)/float64(len(bodies)), "bytes/doc")
}
