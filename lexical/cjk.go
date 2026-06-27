package lexical

import (
	"sort"
	"strings"
	"unicode"

	"github.com/cespare/xxhash/v2"
)

// CJK languages write without spaces between words, so the letter-run tokenizer that
// works for the alphabetic scripts would emit one whole sentence as a single term: a
// Chinese query for one word would never match a document that buried it inside a
// longer run. Doc 10 specifies the fix, dictionary maximum matching with a character
// bigram fallback, and this file implements it. The detection that routes a CJK query
// here landed with the language identifier; this slice is the segmenter it routes to.
//
// The segmenter is shared, immutable, and built once from the embedded dictionary, so
// the build side and the query side segment identical text into identical terms, the
// same byte-for-byte agreement every other analysis step depends on. A run the
// dictionary covers is split into its dictionary words; a run it does not is split into
// overlapping character bigrams, which keeps recall without a dictionary entry for
// every name and number a real crawl carries.

// IsCJKRune reports whether r is a Han ideograph or a Japanese kana, the scripts that
// write without word spaces and so need segmentation. Hangul is left out on purpose:
// Korean separates its words with spaces, so the letter-run tokenizer already splits
// it correctly and a bigram pass would only fragment words that are already whole.
func IsCJKRune(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r)
}

// cjkSegmenter holds the segmentation dictionary as a word set keyed by string and the
// longest word length in runes, the bound the forward scan starts from. It is built
// once and never mutated, so it is safe to share across every concurrent query.
type cjkSegmenter struct {
	words  map[string]struct{}
	maxLen int
}

// newCJKSegmenter builds a segmenter from a word list. Single-rune entries are dropped
// because the fallback already emits lone characters and a one-rune dictionary entry
// would only shadow the bigram path. The maximum word length bounds the forward scan.
func newCJKSegmenter(words []string) *cjkSegmenter {
	set := make(map[string]struct{}, len(words))
	maxLen := 2
	for _, w := range words {
		n := len([]rune(w))
		if n < 2 {
			continue
		}
		set[w] = struct{}{}
		if n > maxLen {
			maxLen = n
		}
	}
	return &cjkSegmenter{words: set, maxLen: maxLen}
}

// split walks a token run, segmenting the CJK spans and leaving the non-CJK spans, the
// Latin letters and digits a mixed run like "iphone手機" carries, as one token each.
// The non-CJK pieces are returned for the caller's fold and stem steps; the CJK terms
// need neither, since accent folding and English stemming are no-ops on them.
func (s *cjkSegmenter) split(tok string) []string {
	runes := []rune(tok)
	n := len(runes)
	var out []string
	for i := 0; i < n; {
		if IsCJKRune(runes[i]) {
			j := i
			for j < n && IsCJKRune(runes[j]) {
				j++
			}
			out = append(out, s.segment(runes[i:j])...)
			i = j
			continue
		}
		j := i
		for j < n && !IsCJKRune(runes[j]) {
			j++
		}
		out = append(out, string(runes[i:j]))
		i = j
	}
	return out
}

// segment splits one run of CJK runes into terms. At each position it takes the longest
// dictionary word that starts there, two runes or more, and advances past it; where no
// dictionary word starts it emits the overlapping bigram of this rune and the next and
// advances one rune, so the next rune still gets its own chance to start a dictionary
// word or its own bigram. A lone final rune the dictionary does not cover is emitted on
// its own. This is forward maximum matching with the bigram fallback doc 10 names: the
// dictionary gives precision where a word is known, the bigrams give recall everywhere
// else, and the whole pass is deterministic so the build and the query agree.
func (s *cjkSegmenter) segment(runes []rune) []string {
	n := len(runes)
	out := make([]string, 0, n)
	lastWasBigram := false
	for i := 0; i < n; {
		max := s.maxLen
		if i+max > n {
			max = n - i
		}
		matched := 0
		for l := max; l >= 2; l-- {
			if _, ok := s.words[string(runes[i:i+l])]; ok {
				matched = l
				break
			}
		}
		if matched >= 2 {
			out = append(out, string(runes[i:i+matched]))
			i += matched
			lastWasBigram = false
			continue
		}
		if i+1 < n {
			out = append(out, string(runes[i:i+2]))
			i++
			lastWasBigram = true
			continue
		}
		// A lone final rune the dictionary does not cover is only its own term when it
		// was not already the tail of the bigram just emitted; otherwise it is redundant.
		if !lastWasBigram {
			out = append(out, string(runes[i:i+1]))
		}
		i++
	}
	return out
}

// hash is the dictionary's identity, folded into the analyzer_hash so a change to the
// segmentation dictionary forces the same rebuild signal a change to any other analysis
// policy does. It hashes the sorted word set, so two segmenters hash equal exactly when
// they segment identically.
func (s *cjkSegmenter) hash() uint64 {
	words := make([]string, 0, len(s.words))
	for w := range s.words {
		words = append(words, w)
	}
	sort.Strings(words)
	return xxhash.Sum64String(strings.Join(words, " "))
}

// sharedCJK is the one segmenter the process holds, built from the embedded dictionary.
// Every analyzer with segmentation turned on uses it, so build and query share one
// dictionary and one segmentation, and it is immutable after construction.
var sharedCJK = newCJKSegmenter(cjkDictionary)
