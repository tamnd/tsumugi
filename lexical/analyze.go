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
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		} else {
			flush()
		}
	}
	flush()
	return out
}
