package langid

import "testing"

// TestLanguageIDStableAndUnique pins the categorical id table the feature matrix stores:
// the unknown code maps to zero, every known language maps to a distinct nonzero id, and
// the ids fit a one-byte column. A renumbering or a collision would silently relabel
// documents under a model trained against the old table, so the test guards the table the
// way the FeatureID space is guarded.
func TestLanguageIDStableAndUnique(t *testing.T) {
	if LanguageID(Unknown) != 0 {
		t.Fatalf("unknown id = %d, want 0", LanguageID(Unknown))
	}
	if LanguageID("xx") != 0 {
		t.Fatalf("unrecognized code id = %d, want 0 (unknown)", LanguageID("xx"))
	}

	langs := []string{
		English, Spanish, French, German, Italian, Portuguese, Dutch,
		Chinese, Japanese, Korean, Russian, Arabic, Hebrew, Hindi, Thai, Greek,
	}
	seen := map[uint16]string{0: Unknown}
	for _, l := range langs {
		id := LanguageID(l)
		if id == 0 {
			t.Errorf("known language %q maps to the unknown id 0", l)
		}
		if id > 255 {
			t.Errorf("language %q id %d does not fit a one-byte column", l, id)
		}
		if prev, ok := seen[id]; ok {
			t.Errorf("language %q and %q share id %d", l, prev, id)
		}
		seen[id] = l
	}
}

// TestLanguageIDPinned pins the exact id of a few codes, so an accidental reorder of the
// table (which keeps uniqueness but breaks compatibility) is still caught.
func TestLanguageIDPinned(t *testing.T) {
	cases := map[string]uint16{
		English:  1,
		French:   3,
		Dutch:    7,
		Chinese:  8,
		Japanese: 9,
		Greek:    16,
	}
	for code, want := range cases {
		if got := LanguageID(code); got != want {
			t.Errorf("LanguageID(%q) = %d, want %d", code, got, want)
		}
	}
}
