package lexical

import "strings"

// StemEnglish reduces an English word to its Porter2 (Snowball "English") stem.
// The algorithm follows the reference at https://snowballstem.org/algorithms/english/stemmer.html.
// It operates on ASCII only.
// Words that contain a non-ASCII byte, or that are shorter than three letters, are returned unchanged.
func StemEnglish(word string) string {
	if word == "" {
		return word
	}
	// Bail out on any non-ASCII byte.
	// The Porter2 algorithm is defined over the ASCII English alphabet only.
	for i := 0; i < len(word); i++ {
		if word[i] >= 0x80 {
			return word
		}
	}

	lower := strings.ToLower(word)

	// Porter2 leaves words of two letters or fewer untouched.
	if len(lower) <= 2 {
		return lower
	}

	// Exceptional forms that are stemmed by direct lookup, before anything else runs.
	if s, ok := exceptional1[lower]; ok {
		return s
	}
	// Invariant words that are returned unchanged and skip the whole algorithm.
	if invariant[lower] {
		return lower
	}

	w := []byte(lower)

	// Mark every y that should behave as a consonant by upper-casing it.
	// A y at the start of the word, or a y right after a vowel, is a consonant here.
	w = markConsonantY(w)

	// step 0: trailing apostrophe forms.
	w = step0(w)

	r1, r2 := regions(w)

	w = step1a(w)

	// Recheck the post-1a exceptional forms.
	// These are words that should not be stemmed further once 1a has run.
	if exceptional2[string(w)] {
		return strings.ToLower(string(w))
	}

	w = step1b(w, r1)
	w = step1c(w)
	w = step2(w, r1)
	w = step3(w, r1, r2)
	w = step4(w, r2)
	w = step5(w, r1, r2)

	// Turn the consonant Y markers back into ordinary lowercase y.
	return strings.ToLower(string(w))
}

// exceptional1 maps words handled by direct lookup at the very start of the algorithm.
var exceptional1 = map[string]string{
	"skis":   "ski",
	"skies":  "sky",
	"dying":  "die",
	"lying":  "lie",
	"tying":  "tie",
	"idly":   "idl",
	"gently": "gentl",
	"ugly":   "ugli",
	"early":  "earli",
	"only":   "onli",
	"singly": "singl",
}

// invariant is the set of words returned unchanged, skipping the algorithm entirely.
var invariant = map[string]bool{
	"sky":    true,
	"news":   true,
	"howe":   true,
	"atlas":  true,
	"cosmos": true,
	"bias":   true,
	"andes":  true,
}

// exceptional2 holds words that must not be stemmed past step 1a.
var exceptional2 = map[string]bool{
	"inning":  true,
	"outing":  true,
	"canning": true,
	"herring": true,
	"earring": true,
	"proceed": true,
	"exceed":  true,
	"succeed": true,
}

func isVowel(c byte) bool {
	switch c {
	case 'a', 'e', 'i', 'o', 'u', 'y':
		return true
	}
	return false
}

// markConsonantY upper-cases each y that acts as a consonant.
// That is a y at the very start of the word, or a y immediately after a vowel.
func markConsonantY(w []byte) []byte {
	for i := 0; i < len(w); i++ {
		if w[i] != 'y' {
			continue
		}
		if i == 0 || isVowel(w[i-1]) {
			w[i] = 'Y'
		}
	}
	return w
}

// regions computes the start indexes of R1 and R2.
// R1 is the region after the first non-vowel following a vowel.
// R2 is the region after the first non-vowel following a vowel inside R1.
// Words starting with gener, commun, or arsen get a fixed R1 per the spec.
func regions(w []byte) (int, int) {
	s := string(w)
	var r1 int
	switch {
	case strings.HasPrefix(s, "gener"):
		r1 = 5
	case strings.HasPrefix(s, "commun"):
		r1 = 6
	case strings.HasPrefix(s, "arsen"):
		r1 = 5
	default:
		r1 = regionAfter(w, 0)
	}
	r2 := regionAfter(w, r1)
	return r1, r2
}

// regionAfter returns the index after the first vowel-then-consonant pair found at or after start.
// If no such pair exists the region begins at the end of the word.
func regionAfter(w []byte, start int) int {
	i := start
	// Find a vowel.
	for i < len(w) && !isVowel(w[i]) {
		i++
	}
	// Find the following non-vowel.
	for i < len(w) && isVowel(w[i]) {
		i++
	}
	if i < len(w) {
		return i + 1
	}
	return len(w)
}

// step0 strips the trailing apostrophe forms 's', 's, and '.
func step0(w []byte) []byte {
	s := string(w)
	switch {
	case strings.HasSuffix(s, "'s'"):
		return w[:len(w)-3]
	case strings.HasSuffix(s, "'s"):
		return w[:len(w)-2]
	case strings.HasSuffix(s, "'"):
		return w[:len(w)-1]
	}
	return w
}

// step1a handles plural and possessive suffixes.
func step1a(w []byte) []byte {
	s := string(w)
	switch {
	case strings.HasSuffix(s, "sses"):
		return w[:len(w)-2] // sses -> ss
	case strings.HasSuffix(s, "ied") || strings.HasSuffix(s, "ies"):
		// Replace by i if the preceding part is longer than one letter, by ie otherwise.
		if len(w)-3 > 1 {
			return append(w[:len(w)-3], 'i')
		}
		return append(w[:len(w)-3], 'i', 'e')
	case strings.HasSuffix(s, "us") || strings.HasSuffix(s, "ss"):
		return w
	case strings.HasSuffix(s, "s"):
		// Delete the final s if the part before it contains a vowel not immediately before the s.
		for i := 0; i < len(w)-2; i++ {
			if isVowel(w[i]) {
				return w[:len(w)-1]
			}
		}
		return w
	}
	return w
}

// step1b handles the ed and ing family of suffixes.
func step1b(w []byte, r1 int) []byte {
	s := string(w)
	switch {
	case strings.HasSuffix(s, "eedly"):
		if len(w)-5 >= r1 {
			return w[:len(w)-3] // eedly -> ee
		}
		return w
	case strings.HasSuffix(s, "eed"):
		if len(w)-3 >= r1 {
			return w[:len(w)-1] // eed -> ee
		}
		return w
	}

	var suffixLen int
	switch {
	case strings.HasSuffix(s, "ingly"):
		suffixLen = 5
	case strings.HasSuffix(s, "edly"):
		suffixLen = 4
	case strings.HasSuffix(s, "ing"):
		suffixLen = 3
	case strings.HasSuffix(s, "ed"):
		suffixLen = 2
	default:
		return w
	}

	stem := w[:len(w)-suffixLen]
	// Delete only if the preceding word part contains a vowel.
	if !containsVowel(stem) {
		return w
	}
	w = stem

	// Post-processing on the freshly cut stem.
	s = string(w)
	switch {
	case strings.HasSuffix(s, "at") || strings.HasSuffix(s, "bl") || strings.HasSuffix(s, "iz"):
		return append(w, 'e')
	case endsDoubleConsonant(w):
		return w[:len(w)-1]
	case isShortWord(w, r1):
		return append(w, 'e')
	}
	return w
}

// containsVowel reports whether the slice holds at least one vowel.
func containsVowel(w []byte) bool {
	for i := 0; i < len(w); i++ {
		if isVowel(w[i]) {
			return true
		}
	}
	return false
}

// endsDoubleConsonant reports whether the word ends in a double of one of the relevant consonants.
func endsDoubleConsonant(w []byte) bool {
	if len(w) < 2 {
		return false
	}
	a, b := w[len(w)-2], w[len(w)-1]
	if a != b {
		return false
	}
	switch a {
	case 'b', 'd', 'f', 'g', 'm', 'n', 'p', 'r', 't':
		return true
	}
	return false
}

// isShortSyllable reports whether the word ends in a short syllable.
// That is a vowel followed by a non-vowel other than w, x, or Y, with the vowel not at the word start,
// or a vowel-consonant pair at the very start of the word.
func isShortSyllable(w []byte) bool {
	n := len(w)
	if n == 2 {
		// Vowel then non-vowel at the start of the word.
		return isVowel(w[0]) && !isVowel(w[1])
	}
	if n >= 3 {
		a, b, c := w[n-3], w[n-2], w[n-1]
		if !isVowel(a) && isVowel(b) && !isVowel(c) {
			switch c {
			case 'w', 'x', 'Y':
				return false
			}
			return true
		}
	}
	return false
}

// isShortWord reports whether the word is short.
// A short word ends in a short syllable and has an empty R1.
func isShortWord(w []byte, r1 int) bool {
	return r1 >= len(w) && isShortSyllable(w)
}

// step1c turns a trailing y or Y into i when it follows a consonant that is not the first letter.
func step1c(w []byte) []byte {
	n := len(w)
	if n < 3 {
		return w
	}
	last := w[n-1]
	if last != 'y' && last != 'Y' {
		return w
	}
	if !isVowel(w[n-2]) {
		w[n-1] = 'i'
	}
	return w
}

// step2 rewrites a set of derivational suffixes when they sit in R1.
func step2(w []byte, r1 int) []byte {
	type rule struct {
		suffix string
		repl   string
	}
	// Order matters: longer and more specific suffixes come first so the longest match wins.
	rules := []rule{
		{"ization", "ize"},
		{"ational", "ate"},
		{"fulness", "ful"},
		{"ousness", "ous"},
		{"iveness", "ive"},
		{"tional", "tion"},
		{"biliti", "ble"},
		{"lessli", "less"},
		{"entli", "ent"},
		{"ation", "ate"},
		{"alism", "al"},
		{"aliti", "al"},
		{"ousli", "ous"},
		{"iviti", "ive"},
		{"fulli", "ful"},
		{"enci", "ence"},
		{"anci", "ance"},
		{"abli", "able"},
		{"izer", "ize"},
		{"ator", "ate"},
		{"alli", "al"},
		{"bli", "ble"},
	}
	s := string(w)
	for _, r := range rules {
		if strings.HasSuffix(s, r.suffix) {
			if len(w)-len(r.suffix) >= r1 {
				return append(w[:len(w)-len(r.suffix)], []byte(r.repl)...)
			}
			return w
		}
	}
	// ogi -> og, but only after an l.
	if strings.HasSuffix(s, "ogi") {
		if len(w)-3 >= r1 && len(w) >= 4 && w[len(w)-4] == 'l' {
			return w[:len(w)-1]
		}
		return w
	}
	// li deletion: drop a trailing li when it sits in R1 and follows a valid li-ending.
	if strings.HasSuffix(s, "li") {
		if len(w)-2 >= r1 && len(w) >= 3 && isLiEnding(w[len(w)-3]) {
			return w[:len(w)-2]
		}
	}
	return w
}

// isLiEnding reports whether c is one of the letters that may precede a deletable li.
func isLiEnding(c byte) bool {
	switch c {
	case 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't':
		return true
	}
	return false
}

// step3 rewrites another set of suffixes when they sit in R1, plus ative which needs R2.
func step3(w []byte, r1, r2 int) []byte {
	type rule struct {
		suffix string
		repl   string
	}
	rules := []rule{
		{"ational", "ate"},
		{"tional", "tion"},
		{"alize", "al"},
		{"icate", "ic"},
		{"iciti", "ic"},
		{"ical", "ic"},
		{"ful", ""},
		{"ness", ""},
	}
	s := string(w)
	for _, r := range rules {
		if strings.HasSuffix(s, r.suffix) {
			if len(w)-len(r.suffix) >= r1 {
				return append(w[:len(w)-len(r.suffix)], []byte(r.repl)...)
			}
			return w
		}
	}
	// ative is deleted only when it lies in R2.
	if strings.HasSuffix(s, "ative") {
		if len(w)-5 >= r2 {
			return w[:len(w)-5]
		}
	}
	return w
}

// step4 deletes a set of suffixes when they sit in R2.
func step4(w []byte, r2 int) []byte {
	// Try the longest matching suffix first.
	// We order explicitly so a short suffix never wins over a longer one.
	ordered := []string{
		"ement", "ation", "ment", "able", "ible", "ance", "ence", "ant",
		"ent", "ism", "ate", "iti", "ous", "ive", "ize", "al", "er", "ic", "ou",
	}
	s := string(w)
	for _, suf := range ordered {
		if strings.HasSuffix(s, suf) {
			if len(w)-len(suf) >= r2 {
				return w[:len(w)-len(suf)]
			}
			return w
		}
	}
	// ion is deleted only when it lies in R2 and follows an s or t.
	if strings.HasSuffix(s, "ion") {
		if len(w)-3 >= r2 && len(w) >= 4 {
			prev := w[len(w)-4]
			if prev == 's' || prev == 't' {
				return w[:len(w)-3]
			}
		}
	}
	return w
}

// step5 removes a trailing e and a trailing l under the region conditions.
func step5(w []byte, r1, r2 int) []byte {
	n := len(w)
	if n == 0 {
		return w
	}
	last := w[n-1]
	if last == 'e' {
		// Delete e if it is in R2, or in R1 and not preceded by a short syllable.
		if n-1 >= r2 {
			return w[:n-1]
		}
		if n-1 >= r1 && !isShortSyllable(w[:n-1]) {
			return w[:n-1]
		}
		return w
	}
	if last == 'l' {
		// Delete l if it is in R2 and preceded by another l.
		if n-1 >= r2 && n >= 2 && w[n-2] == 'l' {
			return w[:n-1]
		}
	}
	return w
}
