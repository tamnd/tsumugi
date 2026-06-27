package lexical

import (
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// NFKC returns the NFKC normalization of s, the spec's first analysis step. Unicode
// lets one visible string be encoded several ways: a precomposed "é" is one code
// point or "e" plus a combining accent, a full-width "Ａ" is a separate code point
// from "A", a ligature "ﬁ" is one glyph. NFKC folds all of these to one canonical
// form so the rest of the chain sees one byte sequence for one piece of text, which
// is the property a byte-for-byte dictionary lookup depends on. The pure-ASCII fast
// path skips the normalizer, because ASCII text is already in normal form.
func NFKC(s string) string {
	if isASCII(s) {
		return s
	}
	return norm.NFKC.String(s)
}

// FoldAccents strips diacritics, mapping "café" to "cafe" and "Müller" to "muller",
// so an accent-careless query finds an accented document and the reverse. It
// decomposes to NFD and drops the combining marks, which is the standard accent
// fold. It is applied per token, after case folding, and only when the analyzer's
// per-language policy turns it on. The ASCII fast path returns the token untouched,
// because ASCII carries no combining marks.
func FoldAccents(tok string) string {
	if isASCII(tok) {
		return tok
	}
	d := norm.NFD.String(tok)
	out := make([]rune, 0, len(d))
	for _, r := range d {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		out = append(out, r)
	}
	return string(out)
}

// isASCII reports whether s is pure ASCII, the common case the fold steps skip.
func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}
