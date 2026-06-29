package collection

import (
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/langid"
)

// TestLanguageIDsFromDetection checks the categorical id column records the real detected
// language: a confidently-detected page gets its language's stable id, and a page the
// detector cannot place gets zero, the unknown id, rather than a guessed one. The ids are
// the ones the langid table assigns, so a page detected English reads langid.LanguageID("en").
func TestLanguageIDsFromDetection(t *testing.T) {
	det := langid.New()
	docs := []convert.Document{
		langDoc("a.test", "en"),                               // 0: English
		langDoc("b.test", "fr"),                               // 1: French
		langDoc("c.test", "de"),                               // 2: German
		{URL: "https://d.test/x", Host: "d.test", Body: "ok"}, // 3: too short to place
	}
	lang, confident := detectLanguages(docs, det)
	ids := languageIDsFrom(lang, confident)

	want := []uint32{
		uint32(langid.LanguageID("en")),
		uint32(langid.LanguageID("fr")),
		uint32(langid.LanguageID("de")),
		0,
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("doc %d language id = %d, want %d (detected %q confident %v)", i, ids[i], want[i], lang[i], confident[i])
		}
	}
	// The placed pages must carry three different nonzero ids, so the column is reading the
	// language rather than collapsing everything to one code.
	if want[0] == 0 || want[1] == 0 || want[2] == 0 {
		t.Fatal("a known language mapped to the unknown id")
	}
	if want[0] == want[1] || want[1] == want[2] || want[0] == want[2] {
		t.Fatalf("distinct languages share an id: en=%d fr=%d de=%d", want[0], want[1], want[2])
	}
}

// TestLanguageIDsSharedDetectionPass checks the consistency signal and the id column agree
// on which pages were placed: the build runs the identifier once and feeds both, so a page
// that scores neutral for consistency (unplaceable) must also carry the unknown id, and a
// page with a real id must not be neutral. A drift between the two would mean two
// detection passes disagreeing, the double work this split exists to avoid.
func TestLanguageIDsSharedDetectionPass(t *testing.T) {
	det := langid.New()
	docs := []convert.Document{
		langDoc("a.test", "en"),                               // placed
		langDoc("a.test", "fr"),                               // placed (mismatch on host)
		{URL: "https://a.test/z", Host: "a.test", Body: "hi"}, // unplaceable
	}
	lang, confident := detectLanguages(docs, det)
	ids := languageIDsFrom(lang, confident)
	cons := languageConsistencyFrom(docs, lang, confident)

	for i := range docs {
		placed := ids[i] != 0
		neutral := cons[i] == languageNeutral
		// An unplaceable page is both id-unknown and consistency-neutral; a placed page is
		// neither (its consistency is a hard 0 or 1, never the neutral middle).
		if placed && neutral {
			t.Fatalf("doc %d carries a real id %d but scored neutral consistency", i, ids[i])
		}
		if !placed && !neutral {
			t.Fatalf("doc %d is id-unknown but did not score neutral consistency (%g)", i, cons[i])
		}
	}
}
