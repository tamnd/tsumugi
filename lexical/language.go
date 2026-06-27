package lexical

import "github.com/tamnd/tsumugi/langid"

// Per-language analyzer configuration is the second half of query-language routing:
// the n-gram detector names the language, and this maps that name to the analysis
// chain the language wants. The spec's rule is that routing falls out of term
// routing, so there is no separate router object; the caller detects the language and
// asks ForLanguage for the analyzer, and the same call on the build side and the query
// side guarantees a document and a query in the same language are analyzed identically.
//
// The configurations are deliberately conservative. Every entry shares NFKC for the
// non-ASCII scripts and lowercase tokenization, the steps that are safe in any
// language. They differ where the language differs: English gets the Snowball stemmer,
// the accented Latin languages fold diacritics so a query without accents still
// matches a document with them, and the non-Latin scripts get NFKC to normalize the
// compatibility and width variants their text carries. Stemmers for languages past
// English and curated stopword lists are configuration a deployment layers on; the
// shape here is the analysis-policy difference that detection exists to drive.
var languageAnalyzers = map[string]*Analyzer{
	// English: Snowball stemming, no accent folding (English carries few diacritics).
	"en": {Stemmer: "english"},

	// Accented Latin languages: fold diacritics so an unaccented query reaches an
	// accented document, and NFKC-normalize so composed and decomposed forms agree.
	"es": {NFKC: true, FoldAccents: true},
	"fr": {NFKC: true, FoldAccents: true},
	"it": {NFKC: true, FoldAccents: true},
	"pt": {NFKC: true, FoldAccents: true},
	"de": {NFKC: true, FoldAccents: true},
	"nl": {NFKC: true, FoldAccents: true},

	// Non-Latin scripts: NFKC folds the width and compatibility variants their text
	// carries so half-width and full-width forms match. Word segmentation for the CJK
	// languages is the segmenting analyzer a later slice records here; until then the
	// run-based tokenizer indexes each script run, and NFKC is the safe normalization.
	"zh": {NFKC: true},
	"ja": {NFKC: true},
	"ko": {NFKC: true},
	"ru": {NFKC: true},
	"ar": {NFKC: true},
	"he": {NFKC: true},
	"hi": {NFKC: true},
	"th": {NFKC: true},
	"el": {NFKC: true},
}

// sharedDetector is the one n-gram language detector the process holds, built once
// from the embedded trigram profiles. It is immutable after construction and safe for
// concurrent use, so a single instance serves every query rather than each broker
// rebuilding the profiles.
var sharedDetector = langid.New()

// LangDetector returns the shared n-gram language detector, the one a broker hands to
// query.ParseDetected so the same model identifies every query. It satisfies the
// query.Detector seam through its DetectLang method, which is why the query package
// routes languages without importing langid. The returned detector is shared and
// read-only; callers detect with it, they do not mutate it.
func LangDetector() *langid.Detector {
	return sharedDetector
}

// ForLanguage returns the analyzer a given detected language is analyzed with, on both
// the build side and the query side. An unknown or empty language, the detector's
// low-confidence signal, falls back to DefaultAnalyzer, the script-based default the
// spec calls for: lowercase and tokenize, no language-specific stemming or folding
// that a wrong language guess would apply incorrectly. The returned analyzer is shared
// and immutable; callers read it, they do not mutate it.
func ForLanguage(lang string) *Analyzer {
	if a, ok := languageAnalyzers[lang]; ok {
		return a
	}
	return DefaultAnalyzer
}

// Languages returns the language codes the registry configures, sorted, so a build can
// record the set it routes over and a test can enumerate it. It excludes the empty
// fallback, which is DefaultAnalyzer rather than a configured language.
func Languages() []string {
	out := make([]string, 0, len(languageAnalyzers))
	for lang := range languageAnalyzers {
		out = append(out, lang)
	}
	sortStrings(out)
	return out
}

// sortStrings is an insertion sort kept local so this file adds no import for a list
// that is a couple of dozen short codes; the registry is small and built once.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
