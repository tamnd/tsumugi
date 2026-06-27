// Package langid is the fast n-gram language identifier doc 10 pins for query
// understanding: a small model over character trigrams that scores the query against
// per-language profiles in microseconds and needs no transformer. Detection picks the
// per-language analyzer, so getting it right is what lets a Japanese query be
// segmented and an English one be stemmed without the caller naming the language.
//
// The model is two layers. The first is script: for the non-Latin writing systems the
// script alone names the language closely enough to route analysis, and a single pass
// counting runes settles it. The second, inside the Latin script where a dozen
// languages share an alphabet, is the Cavnar-Trenkle out-of-place trigram measure,
// scoring the query's trigrams against each language profile and taking the nearest.
//
// The spec's safety rule is the low-confidence fallback: when neither layer is
// confident, Detect reports the script's default language with Confident false, the
// signal for the caller to analyze with a script-based default rather than commit to a
// guessed language and silently mis-segment or mis-stem the query.
package langid

import (
	"math"
	"sort"
)

// Result is what Detect returns: the language code, the script it was resolved on, and
// whether the call is confident enough to route on Lang. When Confident is false the
// caller uses a script-based default analyzer instead of the per-language one, the
// spec's rule that a low-confidence guess must not drive analysis.
type Result struct {
	Lang      string
	Script    Script
	Confident bool
}

// Detector holds the built per-language trigram profiles. It is constructed once at
// broker startup and is read-only and safe for concurrent Detect calls, the profiles
// being immutable maps after Build. A nil *Detector is usable and reports every input
// by script alone, the degraded mode for a deployment that ships no trigram profiles.
type Detector struct {
	profiles []*profile
	// minConfidentLatin is the largest acceptable normalized out-of-place distance for
	// a Latin detection to count as confident. Above it the languages are too close to
	// call and the caller falls back to the script default.
	minScriptShare   float64
	maxLatinDistance float64
	minLatinTrigrams int
}

// New builds the detector from the embedded training text, the constructor the broker
// calls at startup. The profiles are a few hundred ranked trigrams per language,
// kilobytes resident, and Build is the only allocation-heavy step, run once.
func New() *Detector {
	return BuildFrom(trainingText)
}

// BuildFrom builds a detector from a caller-supplied training corpus, the seam a
// deployment with a domain corpus uses to retrain the profiles without editing the
// package. The map is language code to representative text; each becomes one profile.
func BuildFrom(corpus map[string]string) *Detector {
	d := &Detector{
		minScriptShare:   0.60,
		maxLatinDistance: 0.92,
		minLatinTrigrams: 3,
	}
	for lang, text := range corpus {
		d.profiles = append(d.profiles, buildProfile(lang, text))
	}
	return d
}

// Detect identifies the language of text. It resolves the script first; for a
// non-Latin script with a clear majority it returns the script's default language,
// confident, because the script names the language. For Latin script it runs the
// trigram model and returns the nearest language, confident only when the distance is
// small enough and the query had enough trigrams to score. For an unknown or
// too-mixed script, or a too-short Latin query, it returns the best guess with
// Confident false so the caller falls back to a script-based default.
func (d *Detector) DetectResult(text string) Result {
	script, share := dominantScript(text)
	if script == ScriptUnknown {
		return Result{Lang: Unknown, Script: ScriptUnknown, Confident: false}
	}

	// Non-Latin scripts route by script. A clear majority is decisive; a script that
	// is present but mixed below the threshold still names the likely language but
	// without confidence, so the caller uses the script default analyzer either way.
	if script != ScriptLatin {
		return Result{
			Lang:      scriptDefault[script],
			Script:    script,
			Confident: share >= d.minScriptShareOr(),
		}
	}

	// Latin script: the trigram model separates the languages that share the alphabet.
	if d == nil || len(d.profiles) == 0 {
		return Result{Lang: Unknown, Script: ScriptLatin, Confident: false}
	}
	q := trigramCounts(text)
	if len(q) < d.minLatinTrigrams {
		return Result{Lang: Unknown, Script: ScriptLatin, Confident: false}
	}
	grams := rankByFrequency(q)

	bestLang := Unknown
	bestDist := math.MaxFloat64
	secondDist := math.MaxFloat64
	for _, p := range d.profiles {
		dist := p.distance(grams)
		if dist < bestDist {
			secondDist = bestDist
			bestDist, bestLang = dist, p.lang
		} else if dist < secondDist {
			secondDist = dist
		}
	}

	// Confidence has two parts: the winner must be close in absolute terms, and it
	// must be clearly ahead of the runner-up. A query that scores near-equal against
	// two languages (a brand name, a shared loanword) is not confidently either.
	confident := bestDist <= d.maxLatinDistance && bestDist < secondDist*0.97
	return Result{Lang: bestLang, Script: ScriptLatin, Confident: confident}
}

// Detect identifies the language and returns the primitives the query package's
// Detector seam expects: the language code and whether the detection is confident. It
// is the structural method query.Detector matches, so the broker hands a *Detector
// straight to query.ParseDetected without the query package importing langid. A
// low-confidence result returns an empty language so the analyzer selector falls back
// to the script-based default rather than route on a guess. Callers that want the
// resolved Script along with the language use DetectResult.
func (d *Detector) DetectLang(text string) (lang string, confident bool) {
	r := d.DetectResult(text)
	if !r.Confident {
		return "", false
	}
	return r.Lang, true
}

// minScriptShareOr returns the configured non-Latin majority threshold, defaulting for
// a nil detector so the script path works without built profiles.
func (d *Detector) minScriptShareOr() float64 {
	if d == nil || d.minScriptShare == 0 {
		return 0.60
	}
	return d.minScriptShare
}

// rankByFrequency turns the query's trigram counts into the ranked list the
// out-of-place measure needs: the query's own profile, most frequent trigram first.
// Ties break on the trigram string so the ranking is deterministic, which is what
// keeps detection reproducible for the same query.
func rankByFrequency(counts map[string]int) []string {
	grams := make([]string, 0, len(counts))
	for g := range counts {
		grams = append(grams, g)
	}
	sort.Slice(grams, func(i, j int) bool {
		gi, gj := grams[i], grams[j]
		if counts[gi] != counts[gj] {
			return counts[gi] > counts[gj]
		}
		return gi < gj
	})
	return grams
}

// distance is the Cavnar-Trenkle out-of-place measure between a query's ranked
// trigrams and this profile, normalized to the query length so it compares across
// query sizes. For each query trigram at query rank i, the cost is how far that rank
// sits from the trigram's rank in this profile; a trigram the profile does not list at
// all costs the maximum, the profile size. The sum divided by the worst possible sum
// is a number in [0,1] where zero is a perfect match and one is no overlap.
func (p *profile) distance(grams []string) float64 {
	if len(grams) == 0 {
		return 1
	}
	var sum int
	for i, g := range grams {
		if r, ok := p.rank[g]; ok {
			d := i - r
			if d < 0 {
				d = -d
			}
			sum += d
		} else {
			sum += profileSize
		}
	}
	worst := profileSize * len(grams)
	return float64(sum) / float64(worst)
}
