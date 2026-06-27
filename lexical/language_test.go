package lexical

import "testing"

// English routes to the Snowball stemmer so a query for "running" reaches a document
// that only says "run", the policy difference detection exists to drive.
func TestForLanguageEnglishStems(t *testing.T) {
	a := ForLanguage("en")
	got := a.Analyze("running runs runned")
	for _, tok := range got {
		if tok != "run" {
			t.Errorf("english analyzer = %v, want every token stemmed to run", got)
			break
		}
	}
}

// The accented Latin languages fold diacritics so an unaccented query matches an
// accented document. Spanish "informacion" must analyze to the same token as
// "información", which is what makes the locale-insensitive match work.
func TestForLanguageFoldsAccents(t *testing.T) {
	for _, lang := range []string{"es", "fr", "it", "pt", "de", "nl"} {
		a := ForLanguage(lang)
		accented := a.Analyze("información")
		plain := a.Analyze("informacion")
		if len(accented) != 1 || len(plain) != 1 || accented[0] != plain[0] {
			t.Errorf("%s: accented %v != plain %v, want equal after folding", lang, accented, plain)
		}
	}
}

// An unknown or empty language falls back to DefaultAnalyzer, the script-based default
// the spec calls for when detection is not confident. The fallback must be the exact
// default, not a fresh analyzer, so its hash matches the shipped index.
func TestForLanguageUnknownIsDefault(t *testing.T) {
	for _, lang := range []string{"", "xx", "klingon"} {
		if got := ForLanguage(lang); got != DefaultAnalyzer {
			t.Errorf("ForLanguage(%q) = %p, want DefaultAnalyzer %p", lang, got, DefaultAnalyzer)
		}
	}
}

// Every configured language must hash distinctly from the default where its policy
// differs, because the analyzer_hash is the consistency guard: if a per-language
// analyzer hashed equal to the default, the broker could not tell it apart and a
// language mismatch would go silent. The non-Latin entries differ only by NFKC, which
// still changes the hash.
func TestForLanguageHashesDifferFromDefault(t *testing.T) {
	def := DefaultAnalyzer.Hash()
	for _, lang := range Languages() {
		if ForLanguage(lang).Hash() == def {
			t.Errorf("ForLanguage(%q) hashes equal to default, want distinct", lang)
		}
	}
}

// The same language code must always return an analyzer that hashes the same, on the
// build side and the query side, which is the whole point of routing through one
// registry: a document and a query in one language analyze identically.
func TestForLanguageStableHash(t *testing.T) {
	for _, lang := range Languages() {
		if ForLanguage(lang).Hash() != ForLanguage(lang).Hash() {
			t.Errorf("ForLanguage(%q) hash not stable", lang)
		}
	}
}

// Languages returns the configured codes sorted, so a build records a stable set.
func TestLanguagesSorted(t *testing.T) {
	langs := Languages()
	if len(langs) == 0 {
		t.Fatal("no languages configured")
	}
	for i := 1; i < len(langs); i++ {
		if langs[i] < langs[i-1] {
			t.Errorf("Languages not sorted: %v", langs)
			break
		}
	}
}
