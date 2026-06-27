// Package lexical implements the .tsumugi lexical region: the inverted index and
// the BM25F retrieval plane. It is the primary plane a keyword query touches.
// M1 ships the BM25F-mode index, the front-coded term dictionary with a bloom
// prefilter, the docID-ordered block posting codec with per-field frequencies,
// the per-block max-contribution metadata, the BlockMax-WAND traversal, and the
// exhaustive scan that stands as the correctness oracle. Spec 2067 doc 04 pins
// the design; the impact-mode learned-sparse path lands later as M6.
package lexical

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Field identifies one of the four indexed fields. BM25F weights and
// length-normalizes each field on its own scale, so a hit's field is carried in
// the posting payload rather than in separate lists.
type Field uint8

const (
	FieldTitle  Field = 0
	FieldBody   Field = 1
	FieldURL    Field = 2
	FieldAnchor Field = 3
	numFields         = 4
)

// Analyze turns text into the normalized tokens the dictionary stores. The build
// and the query run the identical chain so a query term matches a dictionary
// term byte for byte. The chain is deliberately cheap: lowercase, then split on
// runs of letters and digits, dropping everything else. Unicode folding and
// stemming are left for later; what matters for M1 is that both sides agree.
func Analyze(text string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			out = append(out, b.String())
			b.Reset()
		}
	}
	for _, r := range text {
		if IsTokenRune(r) {
			b.WriteRune(FoldRune(r))
		} else {
			flush()
		}
	}
	flush()
	return out
}

// IsTokenRune reports whether r belongs to a token: a letter or a digit. Token
// boundaries are the runs of everything else. The ASCII range is decided inline so
// the common case skips the Unicode tables; non-ASCII falls back to them. This is
// the single source of the analyzer's token rule, so every reader of the index,
// the query side and the online L2 feature scanner included, splits identically.
func IsTokenRune(r rune) bool {
	if r < utf8.RuneSelf {
		return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
	}
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

// FoldRune applies the analyzer's one normalization, lowercasing. ASCII upper case
// folds inline; everything else defers to unicode.ToLower. Pairing it with
// IsTokenRune keeps the build, the query, and the online scanner byte for byte in
// agreement on what a term is.
func FoldRune(r rune) rune {
	if r < utf8.RuneSelf {
		if r >= 'A' && r <= 'Z' {
			return r + ('a' - 'A')
		}
		return r
	}
	return unicode.ToLower(r)
}
