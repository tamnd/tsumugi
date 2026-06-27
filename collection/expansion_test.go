package collection_test

import (
	"os"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/expand"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
)

// These tests prove the curated query expansion on real Common Crawl data. The point of
// query-side expansion is high precision: the table is curated, not learned, so its
// alternatives should be real vocabulary the index actually holds, and expanding a query
// term should broaden it toward terms a document in this corpus really uses. Driving the
// checks from the corpus's own document frequencies keeps the test honest about what the
// expansion reaches on the real distribution rather than asserting a synthetic word list.

// presentTokens reports whether every analyzed token of a form occurs in the corpus, the
// test for whether an expansion alternative points at vocabulary the index holds.
func presentTokens(df map[string]uint32, form string) bool {
	toks := strings.Fields(form)
	if len(toks) == 0 {
		return false
	}
	for _, tok := range toks {
		if df[tok] == 0 {
			return false
		}
	}
	return true
}

// TestExpansionAlternativesAreRealCorpusTermsCCrawl builds the curated table and checks
// that its expansions reach the real corpus vocabulary: for the curated keys present in
// the corpus, a healthy fraction expand to an alternative whose every token is also a
// real corpus term, so the expansion broadens a query toward words documents here use,
// not toward phantom vocabulary.
func TestExpansionAlternativesAreRealCorpusTermsCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}

	_, combined, _ := buildMultiAndCombined(t, 8000, 1000)
	defer func() { _ = combined.Close() }()

	tab := expand.Default(lexical.Analyze)

	// Gather the corpus df for every token that appears in any curated group, in one
	// lookup, so presence checks are a map probe.
	var probe []string
	for _, g := range expand.DefaultGroups {
		for _, raw := range g {
			probe = append(probe, lexical.Analyze(raw)...)
		}
	}
	df := combined.LexDocFreqs(probe)

	keysPresent, expandedToReal := 0, 0
	for _, g := range expand.DefaultGroups {
		for _, raw := range g {
			toks := lexical.Analyze(raw)
			if len(toks) != 1 { // only single-token forms are table keys
				continue
			}
			key := toks[0]
			if df[key] == 0 {
				continue // the key itself is not in this corpus slice
			}
			keysPresent++
			for _, alt := range tab.Expand(key) {
				if presentTokens(df, alt) {
					expandedToReal++
					t.Logf("expansion reaches corpus vocab: %q -> %q", key, alt)
					break
				}
			}
		}
	}

	if keysPresent == 0 {
		t.Fatal("no curated keys are present in the corpus slice")
	}
	// The curated table is general web vocabulary, so a clear majority of its keys that
	// occur in the corpus should expand to an alternative the corpus also holds.
	if r := float64(expandedToReal) / float64(keysPresent); r < 0.5 {
		t.Errorf("only %d/%d present keys expand to real corpus vocabulary (%.0f%%), want >= 50%%",
			expandedToReal, keysPresent, r*100)
	}
	t.Logf("curated keys present in corpus %d, of which %d expand to real corpus vocabulary", keysPresent, expandedToReal)
}

// TestExpansionBroadensQueryCCrawl runs the expansion through the query pipeline on the
// real table and proves it broadens a parsed query: a term that expands carries an
// alternative whose tokens are real corpus terms, so the OR-expansion the traversal
// applies reaches documents the bare term would miss. It picks a curated key that is
// present in the corpus and whose alternative is also present, the case where expansion
// has something real to add.
func TestExpansionBroadensQueryCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}

	_, combined, _ := buildMultiAndCombined(t, 8000, 1000)
	defer func() { _ = combined.Close() }()

	tab := expand.Default(lexical.Analyze)

	var probe []string
	for _, g := range expand.DefaultGroups {
		for _, raw := range g {
			probe = append(probe, lexical.Analyze(raw)...)
		}
	}
	df := combined.LexDocFreqs(probe)

	// Find a curated key present in the corpus with an alternative also present.
	var key, alt string
	for _, g := range expand.DefaultGroups {
		for _, raw := range g {
			toks := lexical.Analyze(raw)
			if len(toks) != 1 || df[toks[0]] == 0 {
				continue
			}
			for _, a := range tab.Expand(toks[0]) {
				if presentTokens(df, a) {
					key, alt = toks[0], a
					break
				}
			}
			if key != "" {
				break
			}
		}
		if key != "" {
			break
		}
	}
	if key == "" {
		t.Skip("no curated key with a corpus-present alternative in this slice")
	}

	pq := query.Parse(key, lexical.DefaultAnalyzer, query.SoftOR)
	pq.ApplyExpansion(tab)

	var qt *query.QueryTerm
	for i := range pq.Terms {
		if pq.Terms[i].Term == key {
			qt = &pq.Terms[i]
		}
	}
	if qt == nil || len(qt.Alts) == 0 {
		t.Fatalf("expansion did not fill Alts for %q: %+v", key, pq.Terms)
	}
	found := false
	for _, a := range qt.Alts {
		if a == alt {
			found = true
		}
	}
	if !found {
		t.Errorf("expected %q among %q.Alts %v", alt, key, qt.Alts)
	}
	t.Logf("query %q expands to alternative %q, both real corpus vocabulary", key, alt)
}

// BenchmarkExpandQueryCCrawl measures the cost of expanding a parsed multi-term query
// against the curated table, the spec's roughly-one-microsecond, mostly-no-op budget.
func BenchmarkExpandQueryCCrawl(b *testing.B) {
	tab := expand.Default(lexical.Analyze)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pq := query.Parse("nyc gray pizza javascript database", lexical.DefaultAnalyzer, query.SoftOR)
		pq.ApplyExpansion(tab)
	}
}
