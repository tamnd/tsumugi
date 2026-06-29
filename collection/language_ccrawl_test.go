package collection

import (
	"os"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/langid"
)

// readCCrawlDocs reads up to limit documents from the real crawl export, the helper the
// real-data signal tests share. It returns the documents in source order, which is the
// order the build ingests them, so a signal computed here lines up with the build's.
func readCCrawlDocs(t *testing.T, limit int) []convert.Document {
	t.Helper()
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()
	var docs []convert.Document
	for len(docs) < limit {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		docs = append(docs, d)
	}
	return docs
}

// TestLanguageConsistencyOnCCrawl runs the language-consistency signal over the real crawl
// export and checks it is well-formed and discriminating on a broad multi-language sample.
//
// The ccrawl slice is close to one page per host, so most hosts are single-page and a
// confidently-detected page on its own host agrees with itself, scoring consistent. The
// Latin-script confidence floor is strict (many short or close-call Latin pages do not
// clear it), so the neutral bucket is large, roughly the share of pages the detector will
// not commit to a language. A true mismatch needs a multi-page host with a page in another
// language, which is rare on this near-flat corpus, so mismatches are few but present, the
// spam signature the signal exists to catch. The gate asserts every score is one of the
// three the signal emits, that consistency is a substantial fraction (the signal finds
// agreement broadly), that a placed page is overwhelmingly consistent rather than a
// mismatch (mismatch is the rare exception, not the norm), and that all three outcomes
// occur, so the signal reads the corpus rather than returning a constant.
func TestLanguageConsistencyOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	docs := readCCrawlDocs(t, 8000)
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	c := languageConsistency(docs, langid.New())
	if len(c) != len(docs) {
		t.Fatalf("got %d scores for %d docs", len(c), len(docs))
	}

	var consistent, neutral, mismatch int
	for i, v := range c {
		switch v {
		case 1:
			consistent++
		case languageNeutral:
			neutral++
		case 0:
			mismatch++
		default:
			t.Fatalf("doc %d score %g is not one of {0, %g, 1}", i, v, languageNeutral)
		}
	}

	// Consistency is found broadly: a substantial fraction of the sample sits on a host in
	// its own language.
	if consistent*4 <= len(docs) {
		t.Fatalf("only %d of %d pages consistent; the signal is not finding agreement", consistent, len(docs))
	}
	// A placed page is overwhelmingly consistent rather than a mismatch: language mismatch
	// is the rare spam signature, not the common case, so it must be a small minority of
	// the pages the detector did place (consistent plus mismatch).
	if mismatch*10 >= consistent {
		t.Fatalf("mismatch %d is not rare against consistent %d; the host-dominant comparison is mislabeling", mismatch, consistent)
	}
	// All three outcomes must occur, or the signal is degenerate on this sample.
	if mismatch == 0 {
		t.Fatalf("no language mismatch found on a broad multi-language sample; the host-dominant comparison is not discriminating")
	}
	if neutral == 0 {
		t.Fatalf("no neutral page found; the confidence floor is not gating short or mixed pages")
	}
	t.Logf("docs=%d consistent=%d neutral=%d mismatch=%d", len(docs), consistent, neutral, mismatch)
}

// BenchmarkLanguageConsistency measures the whole signal (detection over every body plus
// the host-dominant aggregation) over a real sample, the build-side one-time cost.
func BenchmarkLanguageConsistency(b *testing.B) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		b.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for len(docs) < 2000 {
		d, ok, err := src.Next()
		if err != nil || !ok {
			break
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		b.Skip("no documents in parquet")
	}
	det := langid.New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = languageConsistency(docs, det)
	}
}
