package collection_test

import (
	"os"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
	"github.com/tamnd/tsumugi/vector"
)

// denseDim is the kept dimension the test embeds at, a Matryoshka-scale dimension small
// enough to build fast and large enough that random indexing keeps distinct documents
// apart.
const denseDim = 256

// TestDenseEncodeCCrawl proves the dense query encoder end to end on the real crawl,
// through the same vector region the shards serve. It embeds real page bodies with the
// static encoder, builds a real VEC1 region from those vectors, then encodes each page's
// own lead line as a query with the same encoder and asks the region's int8 cosine, the
// production rerank path, to score the query against its own document and against an
// unrelated one. The query's terms are a subset of its document's terms, so a correct
// encoder and a correct region rank the matching document above the unrelated one on the
// large majority of pages. This is the load-bearing dense-plane property: a query and a
// document embedded by the same encoder live in one space, the dense analog of the shared
// analyzer the lexical plane proves elsewhere, and it is checked here on real text routed
// through the real rotation-and-quantize reader rather than on synthetic vectors.
func TestDenseEncodeCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	enc := dense.NewStatic(dense.NewHashTable(denseDim, 8, 1))

	// Collect pages whose body and lead line both carry enough text to embed, keeping the
	// lead line (the query) and the doc vector (the document) for each.
	const want = 300
	type page struct {
		leadTerms []string
		docID     uint32
	}
	var pages []page
	b := vector.NewBuilder(denseDim)
	for seen := 0; seen < 40000 && len(pages) < want; seen++ {
		doc, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read doc: %v", err)
		}
		if !ok {
			break
		}
		lead := firstLine(doc.Body)
		leadTerms := lexical.Analyze(lead)
		bodyTerms := lexical.Analyze(doc.Body)
		if len(leadTerms) < 3 || len(bodyTerms) < 8 {
			continue
		}
		docVec := enc.Encode(bodyTerms)
		// A body whose terms are all unknown pools to the zero vector, no dense signal;
		// skip it so the region carries only documents with a vector to rank.
		zero := true
		for _, x := range docVec {
			if x != 0 {
				zero = false
				break
			}
		}
		if zero {
			continue
		}
		pages = append(pages, page{leadTerms: leadTerms, docID: uint32(len(pages))})
		b.Add(docVec)
	}
	if len(pages) < 50 {
		t.Skipf("only %d usable pages in the slice, too few to gate", len(pages))
	}

	region, err := vector.Open(b.Build())
	if err != nil {
		t.Fatalf("open region: %v", err)
	}

	// For each page, encode its lead line as a query through the full ParsedQuery path so
	// the wire codec is exercised too, decode it back to the float32 vector the reader
	// takes, and compare the int8 cosine to its own document against an unrelated one a
	// fixed stride away. A win is the self score strictly above the other; ties (an
	// all-stopword lead line that encoded to no signal) are dropped from the tally.
	wins, compared := 0, 0
	for i, p := range pages {
		pq := query.ParsedQuery{}
		for _, term := range p.leadTerms {
			pq.Terms = append(pq.Terms, query.QueryTerm{Term: term})
		}
		pq.ApplyDense(enc)
		if pq.DenseVec == nil {
			continue
		}
		qv := query.DecodeDenseVec(pq.DenseVec)

		other := pages[(i+len(pages)/2)%len(pages)].docID
		if other == p.docID {
			continue
		}
		self, ok1 := region.Cosine(qv, p.docID)
		alt, ok2 := region.Cosine(qv, other)
		if !ok1 || !ok2 {
			t.Fatalf("region cosine unavailable for docID %d/%d", p.docID, other)
		}
		compared++
		if self > alt {
			wins++
		}
	}
	if compared < 50 {
		t.Skipf("only %d pages produced a dense query, too few to gate", compared)
	}
	frac := float64(wins) / float64(compared)
	// The lead line is a strict subset of the body, so the self match should dominate; a
	// bag-of-words encoder over noisy crawl boilerplate will not win every single pair, so
	// the gate is a large majority rather than a clean sweep.
	if frac < 0.80 {
		t.Errorf("self-document ranked above unrelated on only %d/%d = %.2f pages, want >= 0.80",
			wins, compared, frac)
	}
	t.Logf("dense pages=%d compared=%d selfWinFrac=%.2f", len(pages), compared, frac)
}

// BenchmarkDenseEncodeCCrawl measures the per-query encode cost on real lead lines, the
// once-per-query broker step, to confirm it is microseconds and far under the latency
// budget.
func BenchmarkDenseEncodeCCrawl(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		b.Fatalf("open source: %v", err)
	}
	enc := dense.NewStatic(dense.NewHashTable(denseDim, 8, 1))
	var queries [][]string
	for len(queries) < 500 {
		doc, ok, err := src.Next()
		if err != nil || !ok {
			break
		}
		terms := lexical.Analyze(firstLine(doc.Body))
		if len(terms) >= 3 {
			queries = append(queries, terms)
		}
	}
	_ = src.Close()
	if len(queries) == 0 {
		b.Skip("no usable lead lines")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = enc.Encode(queries[i%len(queries)])
	}
}
