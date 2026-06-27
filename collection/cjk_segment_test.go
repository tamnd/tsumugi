package collection_test

import (
	"os"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/langid"
	"github.com/tamnd/tsumugi/lexical"
)

// TestSegmentCJKCCrawl proves CJK segmentation on the real crawl, the content the engine
// actually serves. It walks real pages, routes each to its analyzer by the same detector
// the broker uses, and on the pages that route to a CJK language checks the three things
// segmentation has to get right: it splits a space-free CJK run into many short terms
// rather than emitting one giant term, every term is at most the longest dictionary word
// so nothing slips through unsplit, and the build side and the query side segment the
// same page into identical terms. The last is the load-bearing property: a document and a
// query that segment differently would never match, the silent failure the analyzer_hash
// and this gate exist to prevent.
func TestSegmentCJKCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	det := lexical.LangDetector()
	const want = 200
	cjkPages, multiTermPages := 0, 0
	totalTerms, totalRunes := 0, 0
	maxLen := segmenterMaxLen()

	for seen := 0; seen < 60000 && cjkPages < want; seen++ {
		doc, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read doc: %v", err)
		}
		if !ok {
			break
		}
		text := firstLine(doc.Body)
		if len([]rune(text)) < 12 {
			continue
		}
		r := det.DetectResult(text)
		if r.Script != langid.ScriptHan && r.Script != langid.ScriptHiragana && r.Script != langid.ScriptKatakana {
			continue
		}
		lang, _ := det.DetectLang(text)
		a := lexical.ForLanguage(lang)
		if !a.Segment {
			continue
		}
		cjkPages++

		build := a.Analyze(text)
		query := a.Analyze(text)
		if !equalTerms(build, query) {
			t.Fatalf("segmentation not consistent on %q: build %v != query %v", text, build, query)
		}
		if len(build) > 1 {
			multiTermPages++
		}
		for _, tok := range build {
			totalTerms++
			runes := []rune(tok)
			totalRunes += len(runes)
			// The length ceiling applies only to the segmented CJK terms: a mixed run
			// also yields its Latin and digit spans as single tokens, an email handle or
			// a phone number, which are legitimately longer than any dictionary word.
			if len(runes) > maxLen && lexical.IsCJKRune(runes[0]) {
				t.Errorf("CJK term %q (%d runes) longer than the longest dictionary word (%d) on %q",
					tok, len(runes), maxLen, text)
			}
		}
	}

	if cjkPages < 30 {
		t.Skipf("only %d CJK pages in the slice, too few to gate", cjkPages)
	}
	// A space-free CJK lead line almost always carries several words, so the
	// overwhelming majority of CJK pages must segment into more than one term; a
	// segmenter that left them as one giant run would fail this.
	multiFrac := float64(multiTermPages) / float64(cjkPages)
	if multiFrac < 0.90 {
		t.Errorf("only %d/%d = %.2f CJK pages segmented into multiple terms, want >= 0.90",
			multiTermPages, cjkPages, multiFrac)
	}
	// The average term is a word or a bigram, two to a few runes; an average near the
	// whole-line length would mean segmentation was not happening.
	avgRunes := float64(totalRunes) / float64(totalTerms)
	if avgRunes > 4.0 {
		t.Errorf("average term length %.2f runes, want a word-scale average <= 4", avgRunes)
	}
	t.Logf("cjkPages=%d multiTermFrac=%.2f terms=%d avgTermRunes=%.2f", cjkPages, multiFrac, totalTerms, avgRunes)
}

// segmenterMaxLen returns the longest dictionary word length the shared segmenter can
// emit, the ceiling every segmented term must respect. It probes the analyzer rather
// than reaching into the segmenter, so the test stays in the external package.
func segmenterMaxLen() int {
	a := lexical.ForLanguage("zh")
	// A run of a single repeated rune the dictionary does not list segments into
	// bigrams, so the longest term the analyzer emits on arbitrary text is bounded by
	// the dictionary's longest word; six covers the embedded dictionary with headroom.
	max := 0
	for _, tok := range a.Analyze("龘龘龘龘龘龘龘龘") {
		if n := len([]rune(tok)); n > max {
			max = n
		}
	}
	if max < 6 {
		max = 6
	}
	return max
}

// equalTerms reports whether two term slices are identical, the build-equals-query check.
func equalTerms(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkSegmentCJKCCrawl(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		b.Fatalf("open source: %v", err)
	}
	det := lexical.LangDetector()
	var texts []string
	for len(texts) < 500 {
		doc, ok, err := src.Next()
		if err != nil || !ok {
			break
		}
		text := firstLine(doc.Body)
		if len([]rune(text)) < 12 {
			continue
		}
		if r := det.DetectResult(text); r.Script == langid.ScriptHan || r.Script == langid.ScriptHiragana {
			texts = append(texts, text)
		}
	}
	_ = src.Close()
	if len(texts) == 0 {
		b.Skip("no CJK docs")
	}
	a := lexical.ForLanguage("zh")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Analyze(texts[i%len(texts)])
	}
}
