package collection

import (
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/langid"
)

// langSample returns prose long enough for the trigram detector to place confidently in
// each language the synthetic tests use. The detector needs a few hundred characters of
// natural text to score above its confidence floor, so each sample is a paragraph
// repeated rather than a single sentence.
func langSample(lang string) string {
	switch lang {
	case "en":
		return strings.Repeat("The quick brown fox jumps over the lazy dog while the morning sun rises over the quiet village and the people walk along the river path. ", 6)
	case "fr":
		return strings.Repeat("Le renard brun rapide saute par-dessus le chien paresseux pendant que le soleil du matin se leve sur le village tranquille et les gens marchent. ", 6)
	case "de":
		return strings.Repeat("Der schnelle braune Fuchs springt ueber den faulen Hund waehrend die Morgensonne ueber dem ruhigen Dorf aufgeht und die Leute am Fluss entlang gehen. ", 6)
	default:
		return strings.Repeat("the and of to in a is that it for as with on ", 10)
	}
}

func langDoc(host, lang string) convert.Document {
	return convert.Document{URL: "https://" + host + "/" + lang, Host: host, Body: langSample(lang)}
}

// TestLanguageConsistencyAgreesAndConflicts is the core gate: on a host whose pages are
// mostly one language, a page in that language scores high and a page in another language
// scores low, the agreement-versus-mismatch the signal exists to capture.
func TestLanguageConsistencyAgreesAndConflicts(t *testing.T) {
	det := langid.New()
	docs := []convert.Document{
		langDoc("a.test", "en"), // 0: host-dominant English, consistent
		langDoc("a.test", "en"), // 1: consistent
		langDoc("a.test", "en"), // 2: consistent
		langDoc("a.test", "fr"), // 3: French page on an English host, the mismatch
	}
	c := languageConsistency(docs, det)

	// The three English pages must be detected and agree with their host.
	for i := 0; i < 3; i++ {
		if c[i] != 1 {
			t.Fatalf("doc %d: English page on English host scored %g, want 1", i, c[i])
		}
	}
	if c[3] != 0 {
		t.Fatalf("doc 3: French page on an English host scored %g, want 0 (mismatch)", c[3])
	}
}

// TestLanguageConsistencyNeutralWhenUnsure checks the not-evidence stance: a body the
// detector cannot place confidently scores neutral rather than penalized, and a page on a
// host with no confident detection (so no dominant language) scores neutral too.
func TestLanguageConsistencyNeutralWhenUnsure(t *testing.T) {
	det := langid.New()
	docs := []convert.Document{
		{URL: "https://b.test/x", Host: "b.test", Body: "ok"}, // 0: too short to place
		{URL: "https://b.test/y", Host: "b.test", Body: "hi"}, // 1: too short to place
	}
	c := languageConsistency(docs, det)
	for i, v := range c {
		if v != languageNeutral {
			t.Fatalf("doc %d: unplaceable page scored %g, want neutral %g", i, v, languageNeutral)
		}
	}
}

// TestLanguageConsistencyPerHost checks the host-dominant language is computed per host,
// not across the corpus: an English page on an English host and a French page on a French
// host both score consistent even though the two hosts disagree, while a French page on
// the English host scores a mismatch.
func TestLanguageConsistencyPerHost(t *testing.T) {
	det := langid.New()
	docs := []convert.Document{
		langDoc("en.test", "en"), // 0 consistent
		langDoc("en.test", "en"), // 1 consistent
		langDoc("en.test", "fr"), // 2 mismatch on the English host
		langDoc("fr.test", "fr"), // 3 consistent on its own host
		langDoc("fr.test", "fr"), // 4 consistent
	}
	c := languageConsistency(docs, det)
	want := []float64{1, 1, 0, 1, 1}
	for i := range want {
		if c[i] != want[i] {
			t.Fatalf("doc %d on its host scored %g, want %g", i, c[i], want[i])
		}
	}
}

// TestLanguageConsistencyEmpty covers the empty input.
func TestLanguageConsistencyEmpty(t *testing.T) {
	if got := languageConsistency(nil, langid.New()); len(got) != 0 {
		t.Fatalf("nil docs: got len %d, want 0", len(got))
	}
}

// TestLanguageConsistencyFeedsStaticRank checks the signal flows into the composite static
// rank's quality term: holding everything else equal, a language-consistent page outranks
// an otherwise identical page whose language conflicts with its host.
func TestLanguageConsistencyFeedsStaticRank(t *testing.T) {
	const n = 4
	docs := neutralDocs(n)
	base := baseSignals(n)

	consistent := cloneSignals(base)
	consistent.langConsist = make([]float64, n)
	for i := range consistent.langConsist {
		consistent.langConsist[i] = 1
	}
	conflict := cloneSignals(base)
	conflict.langConsist = make([]float64, n)
	for i := range conflict.langConsist {
		conflict.langConsist[i] = 1
	}
	conflict.langConsist[2] = 0 // one page's language conflicts with its host

	rc := compositeStaticRank(docs, consistent)
	rx := compositeStaticRank(docs, conflict)
	if rx[2] >= rc[2] {
		t.Fatalf("language-conflict page %g not below consistent page %g for equal authority", rx[2], rc[2])
	}
	// The pages that did not change keep their rank, so the term is local to the page.
	for _, i := range []int{0, 1, 3} {
		if rx[i] != rc[i] {
			t.Fatalf("doc %d changed rank with no change to its language: %g vs %g", i, rx[i], rc[i])
		}
	}
}
