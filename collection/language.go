package collection

import (
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/langid"
)

// languageConsistency computes the per-document language-consistency content-quality
// signal of spec doc 07. A page whose content language agrees with its host's dominant
// language is more likely legitimate than one where they conflict, because a language
// mismatch (a page that contains one language while its host is otherwise all another) is
// a common signature of scraped, translated, or injected spam content. The signal is one
// value per document in [0,1], higher meaning more consistent, so it enters the composite
// static rank's quality term with a plus and keeps the rank monotone.
//
// The spec defines the signal as the agreement among three inputs: the page's detected
// content language, its declared language (the HTML lang attribute and the Content-Language
// header), and its host's dominant language. This corpus is the ccrawl markdown export,
// whose document carries no declared language (no HTML lang attribute, no response
// headers), so the declared-language term is not available here and the agreement is
// measured between the two inputs the corpus does carry, the detected language and the
// host-dominant language. The declared-language term lands with the raw-HTML crawl path
// that preserves those fields, the same corpus boundary the text-to-HTML ratio sits on.
//
// Detection uses the same n-gram identifier the query path routes on, run over the body,
// so the build-side content language and the query-side language are read by one model.
// The host-dominant language is the most common confident detection among the host's
// pages. The score is the agreement: a page detected confidently in its host's dominant
// language scores high, a page detected confidently in a different language scores low,
// and a page the detector cannot place confidently (a too-short or mixed body) or a page
// on a host with no dominant language (no page detected confidently, so nothing to agree
// with) scores neutral, since absence of a reading is not evidence of a mismatch.
func languageConsistency(docs []convert.Document, det *langid.Detector) []float64 {
	lang, confident := detectLanguages(docs, det)
	return languageConsistencyFrom(docs, lang, confident)
}

// detectLanguages runs the language identifier over each document body once, the single
// detection pass the language signals share so a build does not read every body twice. It
// returns the detected language code per document and whether the detection was confident,
// recording the code only on a confident reading: an unsure page (a too-short or mixed
// body) keeps the empty code and false, so it neither names its own language nor counts
// toward its host's dominant language nor stores a guessed language id.
func detectLanguages(docs []convert.Document, det *langid.Detector) (lang []string, confident []bool) {
	lang = make([]string, len(docs))
	confident = make([]bool, len(docs))
	for i, d := range docs {
		l, ok := det.DetectLang(d.Body)
		if !ok {
			continue
		}
		lang[i] = l
		confident[i] = true
	}
	return lang, confident
}

// languageConsistencyFrom scores the per-document language-consistency signal from a
// detection already computed by detectLanguages: the agreement between each page's
// detected language and its host's dominant language. It is split from the detection so
// the build runs the identifier once and feeds both this signal and the language-id column.
func languageConsistencyFrom(docs []convert.Document, lang []string, confident []bool) []float64 {
	n := len(docs)
	out := make([]float64, n)
	if n == 0 {
		return out
	}

	// The host of each page, and the per-host vote among each host's confident detections.
	hostOf := make([]string, n)
	votes := map[string]map[string]int{}
	for i, d := range docs {
		h := d.Host
		if h == "" {
			h = analyze.HostOf(d.URL)
		}
		hostOf[i] = h
		if !confident[i] {
			continue
		}
		hv := votes[h]
		if hv == nil {
			hv = map[string]int{}
			votes[h] = hv
		}
		hv[lang[i]]++
	}

	// The host-dominant language is the plurality winner among the host's confident
	// detections, ties broken on the language code so the choice is deterministic. A host
	// with no confident detection has no dominant language and leaves its pages neutral.
	dominant := make(map[string]string, len(votes))
	for h, hv := range votes {
		var best string
		var bestN int
		for l, c := range hv {
			if c > bestN || (c == bestN && l < best) {
				best, bestN = l, c
			}
		}
		dominant[h] = best
	}

	for i := range docs {
		if !confident[i] {
			out[i] = languageNeutral
			continue
		}
		dom, ok := dominant[hostOf[i]]
		if !ok {
			out[i] = languageNeutral
			continue
		}
		if lang[i] == dom {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
	return out
}

// languageIDsFrom turns a detection into the categorical language-id column, one stable id
// per document for the feature matrix. A confident detection stores langid.LanguageID of
// its code; an unplaceable page stores zero, the unknown id, since storing a fabricated
// language for a page the detector would not commit to is exactly the mislabel the
// confidence floor exists to avoid. This replaces the latin-script ratio stand-in the
// analyze stage wrote, so the query-time language match reads a real detected language
// rather than a coarse script flag.
func languageIDsFrom(lang []string, confident []bool) []uint32 {
	out := make([]uint32, len(lang))
	for i := range lang {
		if confident[i] {
			out[i] = uint32(langid.LanguageID(lang[i]))
		}
	}
	return out
}

// languageNeutral is the score for a page whose consistency cannot be judged: the
// detector could not place the page, or the host has no dominant language to compare it
// to. It is the middle of the range so an unjudged page neither rewards nor penalizes the
// quality term, the same not-evidence stance the freshness default takes.
const languageNeutral = 0.5
