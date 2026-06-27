package collection_test

import (
	"os"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/spell"
)

// These tests prove the SymSpell corrector on real Common Crawl data. The correction
// dictionary is built from the corpus's own terms and their fleet-wide frequencies, by
// merging every shard's term enumeration, exactly as the broker builds it. The proofs
// are that the merged frequencies reconstruct the combined index's document frequencies,
// and that deliberately corrupting the corpus's own common terms recovers the original,
// which is the corrector doing its job on the real vocabulary distribution rather than a
// synthetic word list.

// buildCorrector merges every shard's term dictionary into one collection-wide SymSpell
// index, summing each term's per-shard document frequency into the fleet-wide df, the
// same merge the broker performs at startup.
func buildCorrector(t testing.TB, shards []shardEnumerator, maxEdit int) *spell.Index {
	t.Helper()
	b := spell.NewBuilder()
	for _, s := range shards {
		s.ForEachTerm(b.Add)
	}
	if b.Len() == 0 {
		t.Fatal("correction dictionary is empty, the shards enumerated no terms")
	}
	// Bound the build the way a real collection does: drop the long tail of one-off
	// crawl noise and over-long non-words so the delete index stays in budget, the
	// web-scale default cutoffs.
	o := spell.DefaultOptions()
	o.MaxEdit = maxEdit
	return b.BuildWithOptions(o)
}

// shardEnumerator is the slice of a shard the corrector build needs.
type shardEnumerator interface {
	ForEachTerm(fn func(term string, docFreq uint32))
}

// TestSpellCorrectorMatchesFleetFrequenciesCCrawl checks that the corrector's per-term
// frequency, summed across the shards, equals the combined single-shard index's document
// frequency for a spread of common terms, the fleet-wide-statistics discipline applied to
// spell correction.
func TestSpellCorrectorMatchesFleetFrequenciesCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}

	multiDir, combined, _ := buildMultiAndCombined(t, 8000, 1000)
	defer func() { _ = combined.Close() }()

	ix, err := collection.LoadIndex(multiDir)
	if err != nil {
		t.Fatalf("load index: %v", err)
	}
	shards := openInIndexOrder(t, multiDir, ix)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()

	enums := make([]shardEnumerator, len(shards))
	for i, s := range shards {
		enums[i] = s
	}
	corr := buildCorrector(t, enums, 2)

	// Cross-check the merged df against the combined index for common terms.
	probes := []string{"the", "time", "data", "page", "information", "world", "news", "contact", "privacy", "search"}
	checked := 0
	for _, term := range probes {
		analyzed := lexical.Analyze(term)
		if len(analyzed) != 1 {
			continue
		}
		a := analyzed[0]
		want := combined.LexDocFreqs([]string{a})[a]
		if want == 0 {
			continue
		}
		got, ok := corr.Freq(a)
		if !ok {
			t.Errorf("term %q absent from the corrector but present in the combined index", a)
			continue
		}
		if got != uint64(want) {
			t.Errorf("term %q: corrector fleet df %d, combined df %d", a, got, want)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no probe terms were checked against the corpus")
	}
	t.Logf("corrector: %d terms, %d delete-variants, %d frequencies cross-checked",
		corr.Len(), corr.Variants(), checked)
}

// TestSpellCorrectorRecoversCorruptedCorpusTermsCCrawl takes the corpus's own most
// frequent long terms, corrupts each with a single realistic edit, and asserts the
// corrector recovers the original. Driving the corruption from the corpus's actual top
// terms makes the test reflect the real vocabulary rather than a guessed word list.
func TestSpellCorrectorRecoversCorruptedCorpusTermsCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}

	multiDir, combined, _ := buildMultiAndCombined(t, 8000, 1000)
	_ = combined.Close()

	ix, err := collection.LoadIndex(multiDir)
	if err != nil {
		t.Fatalf("load index: %v", err)
	}
	shards := openInIndexOrder(t, multiDir, ix)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()
	enums := make([]shardEnumerator, len(shards))
	for i, s := range shards {
		enums[i] = s
	}
	corr := buildCorrector(t, enums, 2)

	top := topAsciiTerms(t, enums, 6, 200)
	if len(top) < 20 {
		t.Fatalf("only %d usable top terms, the corpus is too small for the test", len(top))
	}

	transposeRecovered, deleteRecovered, transposeTried, deleteTried := 0, 0, 0, 0
	for _, term := range top {
		// A transposition of two adjacent middle characters, the "teh" for "the" error.
		if tp := transposeMid(term); tp != term {
			transposeTried++
			if best, ok := corr.Best(tp); ok && best.Term == term {
				transposeRecovered++
			}
		}
		// A single dropped middle character, the most common typo.
		if dl := dropMid(term); dl != term {
			deleteTried++
			if best, ok := corr.Best(dl); ok && best.Term == term {
				deleteRecovered++
			}
		}
	}

	// The corrector should recover the large majority of single-edit corruptions of the
	// corpus's own common terms. It is not a guarantee of every term, because a corrupted
	// term can land closer to a different, more frequent real word, which is correct
	// behavior, so the bar is a high fraction rather than all.
	if r := float64(transposeRecovered) / float64(transposeTried); r < 0.75 {
		t.Errorf("transposition recovery %.0f%% (%d/%d), want >= 75%%", r*100, transposeRecovered, transposeTried)
	}
	if r := float64(deleteRecovered) / float64(deleteTried); r < 0.75 {
		t.Errorf("deletion recovery %.0f%% (%d/%d), want >= 75%%", r*100, deleteRecovered, deleteTried)
	}
	t.Logf("recovered transpositions %d/%d, deletions %d/%d over %d top terms",
		transposeRecovered, transposeTried, deleteRecovered, deleteTried, len(top))
}

// BenchmarkSpellCorrectCCrawl measures one correction against the real corpus vocabulary,
// the hot-path cost the spec budgets at single-digit microseconds. It corrupts a common
// corpus term and times the recovery, the work a real misspelled query term triggers.
func BenchmarkSpellCorrectCCrawl(b *testing.B) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	multiDir, combined, _ := buildMultiAndCombined(b, 8000, 1000)
	_ = combined.Close()

	ix, err := collection.LoadIndex(multiDir)
	if err != nil {
		b.Fatalf("load index: %v", err)
	}
	shards := openInIndexOrder(b, multiDir, ix)
	defer func() {
		for _, s := range shards {
			_ = s.Close()
		}
	}()
	enums := make([]shardEnumerator, len(shards))
	for i, s := range shards {
		enums[i] = s
	}
	corr := buildCorrector(b, enums, 2)
	top := topAsciiTerms(b, enums, 6, 50)
	if len(top) == 0 {
		b.Fatal("no usable top terms")
	}
	probe := transposeMid(top[0])
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := corr.Best(probe); !ok {
			b.Fatal("no correction")
		}
	}
}

// topAsciiTerms returns the highest-frequency pure-ASCII terms of at least minLen runes,
// limited to the top n, so the corruption test runs over the corpus's real common
// vocabulary. Non-ASCII terms are skipped because the corruption helpers operate on
// ASCII letters.
func topAsciiTerms(t testing.TB, shards []shardEnumerator, minLen, n int) []string {
	t.Helper()
	freq := map[string]uint64{}
	for _, s := range shards {
		s.ForEachTerm(func(term string, df uint32) {
			if len([]rune(term)) < minLen || !isAsciiLetters(term) {
				return
			}
			freq[term] += uint64(df)
		})
	}
	terms := make([]string, 0, len(freq))
	for term := range freq {
		terms = append(terms, term)
	}
	sort.Slice(terms, func(i, j int) bool {
		if freq[terms[i]] != freq[terms[j]] {
			return freq[terms[i]] > freq[terms[j]]
		}
		return terms[i] < terms[j]
	})
	if len(terms) > n {
		terms = terms[:n]
	}
	return terms
}

func isAsciiLetters(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < 'a' || c > 'z' {
			return false
		}
	}
	return len(s) > 0
}

// transposeMid swaps the two characters either side of the midpoint.
func transposeMid(s string) string {
	b := []byte(s)
	i := len(b) / 2
	if i == 0 || i >= len(b) {
		return s
	}
	b[i-1], b[i] = b[i], b[i-1]
	return string(b)
}

// dropMid removes the middle character.
func dropMid(s string) string {
	b := []byte(s)
	i := len(b) / 2
	if len(b) <= 1 {
		return s
	}
	return string(b[:i]) + string(b[i+1:])
}
