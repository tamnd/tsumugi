package langid

import "unicode"

// Script names the writing system a run of text is in. The detector's first decision
// is script, not language, because for most non-Latin writing systems the script
// alone pins the language family closely enough to route analysis: Han means CJK,
// Hangul means Korean, Hiragana or Katakana means Japanese. The Cavnar-Trenkle
// trigram model only earns its keep inside the Latin script, where a dozen languages
// share an alphabet and only the n-gram statistics tell them apart.
type Script string

const (
	ScriptLatin      Script = "latin"
	ScriptHan        Script = "han"
	ScriptHiragana   Script = "hiragana"
	ScriptKatakana   Script = "katakana"
	ScriptHangul     Script = "hangul"
	ScriptCyrillic   Script = "cyrillic"
	ScriptArabic     Script = "arabic"
	ScriptHebrew     Script = "hebrew"
	ScriptDevanagari Script = "devanagari"
	ScriptThai       Script = "thai"
	ScriptGreek      Script = "greek"
	ScriptUnknown    Script = "unknown"
)

// scriptOf classifies a single rune into the script it belongs to, or the empty
// Script for runes that carry no script signal (spaces, digits, punctuation), which
// the caller skips so they do not dilute the vote. The order is the common web case
// first: Latin, then the CJK scripts, then the rest.
func scriptOf(r rune) Script {
	switch {
	case r < unicode.MaxASCII:
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return ScriptLatin
		}
		return ""
	case unicode.Is(unicode.Latin, r):
		return ScriptLatin
	case unicode.Is(unicode.Han, r):
		return ScriptHan
	case unicode.Is(unicode.Hiragana, r):
		return ScriptHiragana
	case unicode.Is(unicode.Katakana, r):
		return ScriptKatakana
	case unicode.Is(unicode.Hangul, r):
		return ScriptHangul
	case unicode.Is(unicode.Cyrillic, r):
		return ScriptCyrillic
	case unicode.Is(unicode.Arabic, r):
		return ScriptArabic
	case unicode.Is(unicode.Hebrew, r):
		return ScriptHebrew
	case unicode.Is(unicode.Devanagari, r):
		return ScriptDevanagari
	case unicode.Is(unicode.Thai, r):
		return ScriptThai
	case unicode.Is(unicode.Greek, r):
		return ScriptGreek
	}
	return ""
}

// dominantScript counts the script of every script-bearing rune and returns the most
// common one with its share of the counted runes. Japanese mixes Han with the kana
// syllabaries, so any Hiragana or Katakana presence overrides a Han plurality: a
// sentence that is mostly kanji but carries kana particles is Japanese, not Chinese,
// and that single rule separates the two CJK languages a script vote would otherwise
// confuse. The share is what the caller turns into a confidence: a run that is purely
// one script is decisive, a run split across scripts is not.
func dominantScript(text string) (Script, float64) {
	var counts map[Script]int
	total := 0
	var kana int
	for _, r := range text {
		s := scriptOf(r)
		if s == "" {
			continue
		}
		if counts == nil {
			counts = make(map[Script]int, 4)
		}
		counts[s]++
		total++
		if s == ScriptHiragana || s == ScriptKatakana {
			kana++
		}
	}
	if total == 0 {
		return ScriptUnknown, 0
	}
	// Any kana presence in a CJK run resolves it to Japanese, the kana syllabaries
	// being unique to Japanese among the Han-using languages.
	if kana > 0 && (counts[ScriptHan] > 0 || counts[ScriptHiragana] > 0 || counts[ScriptKatakana] > 0) {
		return ScriptHiragana, float64(counts[ScriptHan]+kana) / float64(total)
	}
	// The winner is the most common script, with ties broken by a fixed priority order
	// rather than map iteration, so a string that is exactly half Latin and half Han
	// resolves the same way every call. Without this, mixed-script text like a Chinese
	// product page with Latin model codes would flap between the script default and the
	// Latin trigram model from one call to the next, and a document would index under a
	// different analyzer than the query that looks for it. Non-Latin scripts win the tie:
	// Latin is the fallback the trigram model and the default analyzer already cover, so
	// when a run is equally Latin and a non-Latin script, the non-Latin script is the
	// more specific, more useful route.
	best := ScriptUnknown
	bestN := 0
	for _, s := range scriptPriority {
		if n := counts[s]; n > bestN {
			best, bestN = s, n
		}
	}
	return best, float64(bestN) / float64(total)
}

// scriptPriority is the fixed order ties are resolved in: every non-Latin script ahead
// of Latin, so a tie between Latin and a non-Latin script routes to the non-Latin one,
// and the order among the non-Latin scripts is fixed so any tie is deterministic. The
// kana scripts sit ahead of Han because a tie there leans Japanese, consistent with the
// kana-override rule above.
var scriptPriority = []Script{
	ScriptHiragana,
	ScriptKatakana,
	ScriptHan,
	ScriptHangul,
	ScriptCyrillic,
	ScriptArabic,
	ScriptHebrew,
	ScriptDevanagari,
	ScriptThai,
	ScriptGreek,
	ScriptLatin,
}
