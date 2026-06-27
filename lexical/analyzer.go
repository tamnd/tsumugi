package lexical

import (
	"sort"
	"strconv"
	"strings"

	"github.com/cespare/xxhash/v2"
)

// Analyzer is the one shared component the spec's query-understanding doc pins: the
// build calls Analyze over each document field to produce the terms it indexes, and
// the query calls the same Analyze over the query string to produce the terms it
// looks up, so a dictionary lookup compares like with like. The configuration is the
// contract. A document indexed with accent folding on and a query analyzed with it
// off would silently never match, so the config is recorded in the shard and the
// query analyzer is reconstructed from it, never set independently.
//
// The chain is the cheap-on-CPU pipeline doc 10 specifies, in order: NFKC, Unicode
// lowercase, tokenize on letter/digit runs, accent fold by policy, light Snowball
// stem by policy, drop stopwords by configuration. Each step past tokenization is a
// flag so a deployment turns on exactly the policy its corpus wants, and the
// analyzer_hash over the whole config is the one-number consistency guard.
type Analyzer struct {
	// NFKC turns on Unicode NFKC normalization as the first step.
	NFKC bool
	// FoldAccents strips diacritics per the language policy.
	FoldAccents bool
	// Stemmer names the stemming algorithm: "" for none, "english" for Snowball English.
	Stemmer string
	// Stopwords is the drop set; empty keeps every token, the spec's default.
	Stopwords map[string]struct{}
}

// DefaultAnalyzer is the analyzer the package-level Analyze uses and the one the
// shipped index was built with: lowercase and tokenize only, no NFKC, no accent
// folding, no stemming, every stopword kept. Its output is byte-for-byte the
// original Analyze, so it leaves the existing shard format and the L2 online scanner
// untouched while the richer steps become first-class config the build can turn on.
var DefaultAnalyzer = &Analyzer{}

// Analyze runs the analysis chain over text and returns the normalized tokens. With
// the default config (every policy off) it is the lowercase-and-tokenize chain the
// index has always used; turning a policy on adds exactly that step on both sides.
func (a *Analyzer) Analyze(text string) []string {
	if a.NFKC {
		text = NFKC(text)
	}
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		tok := b.String()
		b.Reset()
		if a.FoldAccents {
			tok = FoldAccents(tok)
		}
		switch a.Stemmer {
		case "english":
			tok = StemEnglish(tok)
		}
		if a.Stopwords != nil {
			if _, drop := a.Stopwords[tok]; drop {
				return
			}
		}
		out = append(out, tok)
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

// Hash is the analyzer_hash the spec pins: a hash over the entire configuration, so
// the broker can verify in one comparison that every shard it is about to query was
// built with the analyzer it is about to use, and refuse with a clear error if not.
// This turns the silent query-analyzer-not-equal-index-analyzer bug, the one that
// returns wrong results forever, into a loud mismatch the caller can catch. The
// canonical string lists every field in a fixed order, including the sorted stopword
// set, so two analyzers hash equal if and only if they analyze identically.
func (a *Analyzer) Hash() uint64 {
	var b strings.Builder
	b.WriteString("nfkc=")
	b.WriteString(strconv.FormatBool(a.NFKC))
	b.WriteString(";fold=")
	b.WriteString(strconv.FormatBool(a.FoldAccents))
	b.WriteString(";stem=")
	b.WriteString(a.Stemmer)
	b.WriteString(";stop=")
	if len(a.Stopwords) > 0 {
		words := make([]string, 0, len(a.Stopwords))
		for w := range a.Stopwords {
			words = append(words, w)
		}
		sort.Strings(words)
		b.WriteString(strings.Join(words, ","))
	}
	return xxhash.Sum64String(b.String())
}

// Analyze turns text into the normalized tokens the dictionary stores, using the
// default analyzer. The build and the query run this identical chain so a query term
// matches a dictionary term byte for byte. A build that wants the richer chain
// constructs an Analyzer with the policy flags set and calls its Analyze instead,
// recording the analyzer's Hash in the shard so the query side reconstructs it.
func Analyze(text string) []string {
	return DefaultAnalyzer.Analyze(text)
}
