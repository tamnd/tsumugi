package collection

import (
	"math"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
)

// The default composite-static-rank weights. The blend is a weighted sum of
// normalized signals, monotone by construction: quality and authority terms enter
// with a plus, spam and near-dup penalties with a minus, every weight non-negative,
// so increasing any quality signal or decreasing any spam signal can only raise the
// rank. Monotonicity is what makes sorting the postings by this scalar a sound
// upper-bound order for Block-Max Pruning (spec doc 04). The weights lean on the
// robust signals (host rank, domain rank, distinct linking domains, trust) over the
// gameable ones (raw page rank), because the composite orders the postings and a
// gameable composite would let a spammer order itself early, so the weighting is
// itself a soft anti-spam choice. They are the default and are tunable: because
// every raw signal keeps its own column, the composite recomputes as a cheap scan
// over the columns whenever the weights change, with no graph walk.
const (
	wStaticHostRank   = 0.20
	wStaticDomainRank = 0.20
	wStaticTrust      = 0.15
	wStaticLinkDom    = 0.15
	wStaticPageRank   = 0.10
	wStaticFreshness  = 0.10
	wStaticQuality    = 0.10
	wStaticSpamMass   = 0.30
	wStaticNearDup    = 0.20
)

// The content-quality term is itself a blend of the page-local quality signals, both in
// [0,1] and both positive, so the term stays in [0,1] and the composite stays monotone.
// Boilerplate carries most of the weight because it measures the page's own substance
// directly; language consistency is the lighter cross-check that the page's language fits
// its host, the spam-signature signal of spec doc 07.
const (
	wQualityBoiler = 0.70
	wQualityLang   = 0.30
)

// staticRankEps and staticCountEps are the offsets that keep the log scale defined.
// The rank signals (page, host, domain, trust) are tiny positive fractions from the
// power iteration, so a vanishing eps keeps their heavy tail spread under the log;
// the count signals are non-negative integers that include zero, so the count eps is
// one, the same convention the feature package's log quantization uses.
const (
	staticRankEps  = 1e-12
	staticCountEps = 1.0
)

// compositeStaticRank collapses the raw per-document signals into one query-independent
// quality scalar per document, the composite static rank of spec doc 07. It has two
// jobs: it orders the postings so the Block-Max Pruning walk can terminate early, and
// it feeds the ranker as a feature. It is computed once, offline, after the graph
// iteration has written the raw columns, as a thin function of those columns.
//
// Each heavy-tailed authority signal is log-scaled and min-max normalized to [0,1]
// over the corpus so one giant-PageRank page does not swamp the others and the scale
// is stable across shards. The quality term blends 1 minus the boilerplate ratio with
// the language-consistency signal, both already in [0,1]. Freshness defaults to the
// neutral fully-fresh value until the freshness
// family populates a per-page estimate; being uniform it shifts every rank equally
// and does not change the order, but the term is kept so the blend is complete and
// reads the real value once it lands. SpamMass and the near-dup penalty enter
// negatively.
func compositeStaticRank(docs []convert.Document, s graphSignals) []float64 {
	n := len(docs)
	rank := make([]float64, n)
	if n == 0 {
		return rank
	}
	npr := normLog(s.pageRank, staticRankEps)
	nhost := normLog(s.hostRank, staticRankEps)
	ndom := normLog(s.domainRank, staticRankEps)
	ntr := normLog(s.trust, staticRankEps)
	nid := normLogInt(s.linkingDomains, staticCountEps)
	for i := 0; i < n; i++ {
		quality := wQualityBoiler*(1-analyze.BoilerplateRatio(docs[i].Body)) +
			wQualityLang*langConsistencyAt(s, i)
		const freshness = 1.0 // neutral until the freshness family lands
		rank[i] = wStaticPageRank*npr[i] +
			wStaticHostRank*nhost[i] +
			wStaticDomainRank*ndom[i] +
			wStaticTrust*ntr[i] +
			wStaticLinkDom*nid[i] +
			wStaticFreshness*freshness +
			wStaticQuality*quality -
			wStaticSpamMass*s.spamMass[i] -
			wStaticNearDup*s.nearDup[i]
	}
	return rank
}

// impactQuantScale is the top of the one-byte impact range the static rank quantizes into,
// the largest value an impact posting carries. The whole [0,255] byte is used so the rank
// spreads across the full resolution the impact codec stores.
const impactQuantScale = 255

// quantizeImpact maps the composite static rank of each document to the one-byte impact the
// impact-ordered lexical region orders and scores by. The composite rank is an unbounded
// weighted sum (authority terms add, spam and near-dup subtract), so it is min-max
// normalized over the shard and scaled to [0,255], a monotone map that preserves the rank
// order the Block-Max Pruning walk terminates on. A degenerate shard whose ranks are all
// equal maps every document to the top of the range rather than to zero, so query-term
// coverage still orders the results instead of every score collapsing to zero. The build
// and the oracle call this one function, so they agree on the impact byte by construction.
func quantizeImpact(rank []float64) []uint8 {
	out := make([]uint8, len(rank))
	if len(rank) == 0 {
		return out
	}
	lo, hi := rank[0], rank[0]
	for _, v := range rank {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	span := hi - lo
	if span <= 0 {
		for i := range out {
			out[i] = impactQuantScale
		}
		return out
	}
	for i, v := range rank {
		q := (v - lo) / span * impactQuantScale
		out[i] = uint8(math.Round(q))
	}
	return out
}

// langConsistencyAt reads the language-consistency score of document i, defaulting to the
// neutral value when the signal is absent. The full build always populates it, but the
// direct-call tests build a graphSignals from the authority and spam columns alone, and a
// neutral default lets the composite read uniformly for them without a nil check at every
// call site, the same not-evidence stance languageConsistency itself takes.
func langConsistencyAt(s graphSignals, i int) float64 {
	if s.langConsist == nil {
		return languageNeutral
	}
	return s.langConsist[i]
}

// normLog log-scales a float signal with the given offset and min-max normalizes the
// result to [0,1]. A degenerate corpus where every value is equal maps to zero, the
// neutral value that contributes nothing to the order.
func normLog(x []float64, eps float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Log(eps + v)
	}
	return normalize01(out)
}

// normLogInt is normLog for an integer count signal, log-scaled with the count offset
// and normalized the same way.
func normLogInt(x []int, eps float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Log(eps + float64(v))
	}
	return normalize01(out)
}

// normalize01 maps a slice to [0,1] by its min and max in place on a copy. When the
// range is zero (all equal, or empty) every entry becomes zero.
func normalize01(x []float64) []float64 {
	out := make([]float64, len(x))
	if len(x) == 0 {
		return out
	}
	lo, hi := x[0], x[0]
	for _, v := range x {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	span := hi - lo
	if span <= 0 {
		return out
	}
	for i, v := range x {
		out[i] = (v - lo) / span
	}
	return out
}
