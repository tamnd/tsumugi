package cli

import (
	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/expand"
	"github.com/tamnd/tsumugi/langid"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
	"github.com/tamnd/tsumugi/search"
	"github.com/tamnd/tsumugi/spell"
)

// denseNonzero and denseSeed parameterize the broker's dense query encoder, the
// random-indexing table that turns a query's terms into the dense-plane vector. They are
// the canon settings the dense tests pin: eight signed coordinates per token, which keeps
// distinct tokens near-orthogonal while a pooled query still fills enough of the space to
// compare, and a fixed seed so the table is reproducible across broker restarts.
const (
	denseNonzero = 8
	denseSeed    = 1
)

// pipeline is the query-understanding front the broker runs each raw query through once,
// at the broker, before fanning the parsed query out to the shards. It holds the
// components doc 10 places between the search box and retrieval: a language detector and
// a per-language analyzer selector for ParseDetected, a fleet-wide spell corrector, an
// acronym and synonym expander, and an optional dense encoder. Every component is built
// at startup from the broker's config and its open shards, so the corrector's dictionary
// and the dense plane's dimension match the fleet the broker serves.
type pipeline struct {
	sel       query.AnalyzerFor
	detector  query.Detector
	corrector query.Corrector
	expander  query.Expander
	encoder   query.DenseEncoder
}

// buildPipeline constructs the query-understanding pipeline from the broker's open
// shards. The analyzer selector and the detector are config: lexical.ForLanguage routes a
// detected language to its analysis chain, the same selector the build side uses, and
// langid.New loads the embedded n-gram profiles. The corrector is built from the fleet,
// not one shard, because a per-shard corrector would suggest different corrections
// depending on which shard held a term, so every shard's dictionary is merged through one
// builder. The expander is the curated default table over the package analyzer. The dense
// encoder is wired only when the shards carry a vector region to recall against; with no
// region the dense plane stays off and the encoder is left nil, which ApplyDense reads as
// a no-op.
func buildPipeline(shards []*search.Shard) *pipeline {
	p := &pipeline{
		sel:      func(lang string) query.Analyzer { return lexical.ForLanguage(lang) },
		detector: langid.New(),
		expander: expand.Default(lexical.DefaultAnalyzer.Analyze),
	}

	b := spell.NewBuilder()
	for _, s := range shards {
		s.ForEachTerm(func(term string, df uint32) { b.Add(term, df) })
	}
	p.corrector = spell.NewQueryCorrector(b.BuildWithOptions(spell.DefaultOptions()))

	if dim, ok := denseDim(shards); ok {
		p.encoder = dense.NewStatic(dense.NewHashTable(dim, denseNonzero, denseSeed))
	}
	return p
}

// denseDim returns the input dimension the dense query encoder must produce, read from
// the shards' vector regions, and whether the dense plane is on at all. A query vector
// has to carry exactly the dimension the regions rotate and quantize from, so the broker
// only wires the encoder when every shard that carries a vector region agrees on one
// dimension; a fleet with no vector region, or one whose regions disagree, leaves the
// dense plane off rather than encode into a width some shard cannot read.
func denseDim(shards []*search.Shard) (int, bool) {
	dim, ok := 0, false
	for _, s := range shards {
		d, has := s.VectorDim()
		if !has {
			continue
		}
		if !ok {
			dim, ok = d, true
			continue
		}
		if d != dim {
			return 0, false
		}
	}
	return dim, ok
}

// parse runs the full query-understanding pipeline over a raw query string and returns
// the parsed query the broker ships to the shards: detect the language and analyze with
// its chain, correct, expand, then dense-encode, the order doc 10 fixes. The mode is
// soft-OR, the default web-search semantics. Every Apply step is a no-op on a nil
// component, so a pipeline missing the dense encoder simply skips the dense plane.
func (p *pipeline) parse(raw string) *query.ParsedQuery {
	pq := query.ParseDetected(raw, p.sel, p.detector, query.SoftOR)
	pq.ApplyCorrection(p.corrector)
	pq.ApplyExpansion(p.expander)
	pq.ApplyDense(p.encoder)
	return pq
}
