package langid

import (
	"sort"
	"strings"
)

// The Latin-script languages the trigram model separates. Script alone routes the
// non-Latin writing systems; inside Latin only the n-gram statistics tell English
// from Spanish from German, so this is where the Cavnar-Trenkle profiles earn their
// keep. The set is the common web spread plus the languages whose function words and
// diacritics give a short query a clear signal.
const (
	English    = "en"
	Spanish    = "es"
	French     = "fr"
	German     = "de"
	Italian    = "it"
	Portuguese = "pt"
	Dutch      = "nl"

	// Non-Latin languages the script vote resolves directly, named here so the
	// caller routes on a single language constant rather than a Script.
	Chinese  = "zh"
	Japanese = "ja"
	Korean   = "ko"
	Russian  = "ru"
	Arabic   = "ar"
	Hebrew   = "he"
	Hindi    = "hi"
	Thai     = "th"
	Greek    = "el"

	// Unknown is returned when no script-bearing text is present at all, the signal
	// for the caller to fall back to the script-based default analyzer.
	Unknown = ""
)

// scriptDefault maps a resolved script to the language the caller routes on when the
// script vote is decisive. The CJK and other non-Latin scripts skip the trigram model
// entirely because the script already names the language closely enough for analysis.
var scriptDefault = map[Script]string{
	ScriptHan:        Chinese,
	ScriptHiragana:   Japanese,
	ScriptKatakana:   Japanese,
	ScriptHangul:     Korean,
	ScriptCyrillic:   Russian,
	ScriptArabic:     Arabic,
	ScriptHebrew:     Hebrew,
	ScriptDevanagari: Hindi,
	ScriptThai:       Thai,
	ScriptGreek:      Greek,
}

// profileSize is the number of top-ranked trigrams kept per language profile, the N
// of the Cavnar-Trenkle out-of-place measure. A few hundred captures the function
// words and frequent affixes that carry the language signal without overfitting the
// sample text the profile is trained on.
const profileSize = 300

// profile is a language's trigram fingerprint: the rank of each of its most frequent
// trigrams, rank zero being the most frequent. The detector scores a query by summing
// how far each of the query's trigrams sits from its rank here, the out-of-place
// distance, so a small profile map and an integer sum are the whole hot path.
type profile struct {
	lang string
	rank map[string]int
}

// trigramCounts walks text and tallies every character trigram, padding token
// boundaries with a single space so a short word still contributes the affix trigrams
// that carry much of the language signal. Only script-bearing runs feed the count;
// digits and punctuation are boundaries, matching how the language signal actually
// lives in letters, not symbols.
func trigramCounts(text string) map[string]int {
	counts := make(map[string]int)
	var token []rune
	flush := func() {
		if len(token) == 0 {
			return
		}
		padded := append([]rune{' '}, token...)
		padded = append(padded, ' ')
		for i := 0; i+3 <= len(padded); i++ {
			counts[string(padded[i:i+3])]++
		}
		token = token[:0]
	}
	for _, r := range strings.ToLower(text) {
		if scriptOf(r) == ScriptLatin {
			token = append(token, r)
		} else {
			flush()
		}
	}
	flush()
	return counts
}

// buildProfile turns a language's training text into its ranked trigram profile,
// keeping the profileSize most frequent trigrams and recording each one's rank. Ties
// are broken on the trigram string so the same training text always builds the same
// profile, which keeps detection deterministic across runs.
func buildProfile(lang, training string) *profile {
	counts := trigramCounts(training)
	type tc struct {
		gram string
		n    int
	}
	all := make([]tc, 0, len(counts))
	for g, n := range counts {
		all = append(all, tc{g, n})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].n != all[j].n {
			return all[i].n > all[j].n
		}
		return all[i].gram < all[j].gram
	})
	if len(all) > profileSize {
		all = all[:profileSize]
	}
	rank := make(map[string]int, len(all))
	for i, t := range all {
		rank[t.gram] = i
	}
	return &profile{lang: lang, rank: rank}
}
