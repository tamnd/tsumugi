package lexical

import "testing"

func TestStemEnglish(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		// consign family.
		{"consign", "consign"},
		{"consigned", "consign"},
		{"consigning", "consign"},
		{"consignment", "consign"},
		// consist family.
		{"consist", "consist"},
		{"consisted", "consist"},
		{"consistency", "consist"},
		{"consistent", "consist"},
		{"consistently", "consist"},
		{"consisting", "consist"},
		{"consists", "consist"},
		// exceptional die/lie/tie forms.
		{"died", "die"},
		{"dying", "die"},
		{"lying", "lie"},
		{"tying", "tie"},
		// gener prefix region special case.
		{"generate", "generat"},
		{"generates", "generat"},
		{"generated", "generat"},
		{"generating", "generat"},
		{"general", "general"},
		{"generally", "general"},
		{"generic", "generic"},
		{"generically", "generic"},
		{"generous", "generous"},
		{"generously", "generous"},
		// short-word / 1b post-processing.
		{"agreed", "agre"},
		{"owed", "owe"},
		{"fished", "fish"},
		{"pirating", "pirat"},
		{"pirate", "pirat"},
		// 1c and the invariant/exceptional words.
		{"happy", "happi"},
		{"sky", "sky"},
		{"skis", "ski"},
		{"skies", "sky"},
		{"news", "news"},
		{"howe", "howe"},
		{"atlas", "atlas"},
		{"cosmos", "cosmos"},
		{"bias", "bias"},
		{"andes", "andes"},
		// exceptional li forms.
		{"ugly", "ugli"},
		{"early", "earli"},
		{"only", "onli"},
		{"singly", "singl"},
		{"gently", "gentl"},
		{"idly", "idl"},
		// step 2 and step 3 derivational suffixes.
		{"national", "nation"},
		{"nationalism", "nation"},
		{"rational", "ration"},
		{"sensational", "sensat"},
		{"relational", "relat"},
		{"conditional", "condit"},
		{"conditionally", "condit"},
		{"valenci", "valenc"},
		{"hopefulness", "hope"},
		{"goodness", "good"},
		{"fulness", "ful"},
		{"lovingly", "love"},
		{"analogousli", "analog"},
		{"angularities", "angular"},
		// double consonant and y->i.
		{"plotted", "plot"},
		{"running", "run"},
		{"runs", "run"},
		{"ran", "ran"},
		{"easily", "easili"},
		{"fly", "fli"},
		{"cried", "cri"},
		{"troubled", "troubl"},
		{"troubling", "troubl"},
		{"sized", "size"},
		{"sizing", "size"},
		{"meeting", "meet"},
		// Official Porter2 leaves agreement unchanged: ement deletes only in R2, and here ement is not in R2.
		{"agreement", "agreement"},
		// eed forms that must not be touched outside R1.
		{"succeed", "succeed"},
		{"proceed", "proceed"},
		{"exceed", "exceed"},
	}

	for _, c := range cases {
		if got := StemEnglish(c.in); got != c.want {
			t.Errorf("StemEnglish(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestStemEnglishStable(t *testing.T) {
	// An already-stemmed word should stem to itself.
	for _, w := range []string{"consign", "consist", "generat", "nation", "happi"} {
		if got := StemEnglish(w); got != w {
			t.Errorf("StemEnglish(%q) = %q, want it unchanged", w, got)
		}
	}

	// Non-ASCII words pass through untouched.
	for _, w := range []string{"café", "naïve", "日本語"} {
		if got := StemEnglish(w); got != w {
			t.Errorf("StemEnglish(%q) = %q, want it unchanged", w, got)
		}
	}

	// One and two letter words are left alone.
	for _, w := range []string{"a", "an", "is", "by"} {
		want := w
		if got := StemEnglish(w); got != want {
			t.Errorf("StemEnglish(%q) = %q, want %q", w, got, want)
		}
	}
}
