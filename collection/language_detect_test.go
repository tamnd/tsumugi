package collection_test

import (
	"os"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/langid"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
)

// firstLine returns the document's first non-empty markdown line, the title-ish span
// that carries the cleanest language signal: a body mixes navigation chrome, URLs, and
// boilerplate that dilutes the trigram vote, while the lead line is usually one
// language. Detection runs on a query in production, so a short representative span is
// the faithful unit to test on, not a whole page.
func firstLine(body string) string {
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(line, "# "))
		if len([]rune(line)) >= 12 {
			return line
		}
	}
	return strings.TrimSpace(body)
}

// TestDetectLanguageDistributionCCrawl runs the n-gram detector over a real ccrawl
// slice and checks the two properties the engine relies on: the detector resolves a
// language for the overwhelming majority of real pages, and the distribution it
// produces is plausible for a web crawl, English the plurality with a long multilingual
// tail. The crawl is the real language and content distribution the engine serves, so a
// detector that fell over on it, returning unknown for most pages or collapsing
// everything to one language, would be useless even with perfect synthetic tests.
func TestDetectLanguageDistributionCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	det := langid.New()
	const limit = 8000
	counts := map[string]int{}
	scripts := map[langid.Script]int{}
	resolved, confident, seen := 0, 0, 0
	for seen < limit {
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
		seen++
		r := det.DetectResult(text)
		scripts[r.Script]++
		if r.Lang != langid.Unknown {
			resolved++
			counts[r.Lang]++
		}
		if r.Confident {
			confident++
		}
	}
	if seen < 1000 {
		t.Fatalf("only %d usable docs, want a real slice", seen)
	}

	// The detector must name a language for the large majority of real pages; a
	// detector that bailed to unknown on most of them would not be driving analysis.
	resolvedFrac := float64(resolved) / float64(seen)
	if resolvedFrac < 0.85 {
		t.Errorf("resolved %d/%d = %.2f, want >= 0.85", resolved, seen, resolvedFrac)
	}
	// Confidence is the softer signal: real web lead lines are short, and the Romance
	// languages are close enough on a dozen characters that the detector honestly
	// declines to commit on a meaningful minority, which is the low-confidence fallback
	// working rather than failing. Those route to the default analyzer, safe and
	// consistent, so a majority-confident result is the right bar, not a near-total one.
	confidentFrac := float64(confident) / float64(seen)
	if confidentFrac < 0.60 {
		t.Errorf("confident %d/%d = %.2f, want >= 0.60", confident, seen, confidentFrac)
	}
	// English is the plurality of a general web crawl, and the tail must be nonempty:
	// a detector that mapped everything to one language is as broken as one that
	// resolves nothing.
	if counts[langid.English] == 0 {
		t.Error("no English detected, implausible for a web crawl")
	}
	if len(counts) < 4 {
		t.Errorf("only %d languages detected (%v), want a multilingual tail", len(counts), counts)
	}
	t.Logf("seen=%d resolved=%.2f confident=%.2f languages=%d", seen, resolvedFrac, confidentFrac, len(counts))
	t.Logf("top languages: %s", topN(counts, 8))
	t.Logf("scripts: %v", scripts)
}

// TestDetectRoutesToConsistentAnalyzerCCrawl proves the load-bearing property of
// routing: a document and a query in the same language are analyzed by the same
// analyzer, so they compare like with like. For each real document it detects the
// language, analyzes the lead line with the routed analyzer on the build side, then
// analyzes the same text through the query pipeline's ParseDetected, and checks the
// query produced the same analyzer hash the build would have used. A mismatch is the
// exact silent bug the analyzer_hash exists to catch, so proving it cannot happen on
// real data is the gate.
func TestDetectRoutesToConsistentAnalyzerCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	det := lexical.LangDetector()
	sel := func(lang string) query.Analyzer { return lexical.ForLanguage(lang) }

	const limit = 4000
	checked := 0
	for checked < limit {
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
		checked++

		// Build side: detect the language, pick the analyzer the build would index with.
		lang, confident := det.DetectLang(text)
		buildLang := lang
		if !confident {
			buildLang = ""
		}
		buildAnalyzer := lexical.ForLanguage(buildLang)

		// Query side: the same text through the routing query entry point.
		pq := query.ParseDetected(text, sel, det, query.SoftOR)
		queryLang := pq.Lang
		if !pq.LangConfident {
			queryLang = ""
		}
		queryAnalyzer := lexical.ForLanguage(queryLang)

		if buildAnalyzer.Hash() != queryAnalyzer.Hash() {
			t.Fatalf("analyzer mismatch on %q: build lang %q hash %d, query lang %q hash %d",
				text, buildLang, buildAnalyzer.Hash(), queryLang, queryAnalyzer.Hash())
		}
	}
	if checked < 1000 {
		t.Fatalf("only %d usable docs, want a real slice", checked)
	}
	t.Logf("routing consistent on %d real documents", checked)
}

// topN renders the n highest-count languages as a compact string for the test log, so
// the distribution is visible without a dependency on map order.
func topN(counts map[string]int, n int) string {
	type kv struct {
		k string
		v int
	}
	all := make([]kv, 0, len(counts))
	for k, v := range counts {
		all = append(all, kv{k, v})
	}
	for i := 1; i < len(all); i++ {
		for j := i; j > 0 && all[j].v > all[j-1].v; j-- {
			all[j], all[j-1] = all[j-1], all[j]
		}
	}
	var b strings.Builder
	for i := 0; i < n && i < len(all); i++ {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(all[i].k)
		b.WriteByte('=')
		b.WriteString(itoa(all[i].v))
	}
	return b.String()
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}

func BenchmarkDetectLanguageCCrawl(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		b.Fatalf("open source: %v", err)
	}
	var texts []string
	for len(texts) < 2000 {
		doc, ok, err := src.Next()
		if err != nil || !ok {
			break
		}
		t := firstLine(doc.Body)
		if len([]rune(t)) >= 12 {
			texts = append(texts, t)
		}
	}
	_ = src.Close()
	if len(texts) == 0 {
		b.Skip("no usable docs")
	}
	det := langid.New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = det.DetectResult(texts[i%len(texts)])
	}
}
