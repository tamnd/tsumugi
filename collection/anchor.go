package collection

import (
	"sort"
	"strings"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/mph"
)

// Anchor text inversion. Each document's outbound inline links carry a phrase that
// describes their target; gathered across the corpus and indexed on the target,
// those phrases are the anchor field the spec's doc 02 and doc 04 pin. This file
// does the inversion: it walks every document's AnchorLinks, resolves each target to
// a collection node id through the same directory the link graph uses, and collects
// per target the phrases that point at it, weighted so an off-domain endorsement
// outweighs a within-domain one and a phrase repeated by one source is not counted
// more than once for that source.
//
// The output is one assembled anchor field string per document, in the same node-id
// order the shards number their documents by, so writeShard can index it as the
// document's FieldAnchor. The whole pass is deterministic: phrases sort, weights are
// integer, and the assembly reads them in sorted order, so a build over the same
// corpus yields byte-identical anchor fields and the reproducibility gate holds.

// phraseInfo records, for one (target, phrase) pair, the distinct source domains
// that used the phrase to link the target and whether any of them was off-domain.
type phraseInfo struct {
	domains map[int]bool // distinct source domains that used this phrase for the target
	off     bool         // at least one of those source domains differs from the target's
}

// anchorFields gathers inbound anchor text for every document and returns the
// assembled anchor field string for each, indexed by node id (position in docs).
// A document with no inbound anchors gets an empty string, which AddDoc indexes as
// an empty field. dir is the collection's canonical-URL to node-id directory, the
// same one the link graph resolves targets through, so an anchor edge and a graph
// edge land on the same document.
func anchorFields(docs []convert.Document, dir *mph.Dir) []string {
	// The registered domain of every document, so a source's endorsement of a target
	// can be judged off-domain or within-domain. This is the same domain grouping the
	// link signals bucket by: pages under one domain a farm controls are one voice.
	_, domainOf := groupings(docs)

	// Per target: each phrase and, per phrase, the set of distinct source domains that
	// used it and whether any of those was off-domain. Counting a source domain once
	// per phrase is the down-weighting of a phrase a single source repeats: a page (or
	// a domain) that links a target ten times under the same text still counts once.
	inbound := make(map[int]map[string]*phraseInfo)

	for src, d := range docs {
		srcDom := domainOf[src]
		for _, a := range analyze.AnchorLinks(d) {
			id, ok := dir.Lookup([]byte(a.URL))
			if !ok || int(id) == src {
				continue
			}
			tgt := int(id)
			phrases := inbound[tgt]
			if phrases == nil {
				phrases = make(map[string]*phraseInfo)
				inbound[tgt] = phrases
			}
			pi := phrases[a.Text]
			if pi == nil {
				pi = &phraseInfo{domains: make(map[int]bool)}
				phrases[a.Text] = pi
			}
			pi.domains[srcDom] = true
			if srcDom != domainOf[tgt] {
				pi.off = true
			}
		}
	}

	out := make([]string, len(docs))
	for tgt, phrases := range inbound {
		out[tgt] = assembleAnchorField(phrases)
	}
	return out
}

// assembleAnchorField turns the gathered phrases for one target into its anchor
// field text. Each phrase is weighted by how many distinct source domains used it,
// with an off-domain phrase weighted double a within-domain one, since an outside
// endorsement is the harder-to-manufacture signal the spec values over a site
// linking itself. The weight becomes term frequency: the phrase is emitted that
// many times, so BM25F over the anchor field reads a more-endorsed phrase as the
// stronger match. Weights are capped so no single phrase can dominate the field,
// and phrases are emitted in sorted order so the assembled string is deterministic.
func assembleAnchorField(phrases map[string]*phraseInfo) string {
	if len(phrases) == 0 {
		return ""
	}
	texts := make([]string, 0, len(phrases))
	for t := range phrases {
		texts = append(texts, t)
	}
	sort.Strings(texts)

	var b strings.Builder
	for _, t := range texts {
		pi := phrases[t]
		w := len(pi.domains)
		if pi.off {
			w *= offDomainAnchorWeight
		}
		if w > maxAnchorPhraseWeight {
			w = maxAnchorPhraseWeight
		}
		for i := 0; i < w; i++ {
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(t)
		}
	}
	return b.String()
}

// offDomainAnchorWeight is how much more an off-domain anchor counts than a
// within-domain one. An outside site describing a page is a stronger, less gameable
// signal than a page's own site describing it, so an off-domain phrase enters the
// field at double frequency.
const offDomainAnchorWeight = 2

// maxAnchorPhraseWeight caps the frequency any one phrase reaches in a target's
// anchor field. A wildly popular page can be linked by thousands of domains; without
// a cap one phrase's term frequency would swamp the field and the BM25F saturation
// would stop distinguishing pages at the top. The cap keeps the field bounded in
// size and lets the BM25F length normalization still separate targets.
const maxAnchorPhraseWeight = 16
