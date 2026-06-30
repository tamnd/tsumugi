package cli

import (
	"context"

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
	dim, ok := denseDim(shards)
	return newPipeline(func(b *spell.Builder) {
		for _, s := range shards {
			s.ForEachTerm(func(term string, df uint32) { b.Add(term, df) })
		}
	}, dim, ok)
}

// newPipeline is the core pipeline builder both the broker and the head node share. The
// config components, the analyzer selector, detector, and expander, are the same wherever
// the pipeline runs, so they are built here once. The two parts that depend on the corpus,
// the corrector's dictionary and the dense encoder's dimension, are passed in: feed walks a
// vocabulary into the spell builder (a broker walks its shards, a head walks its peers over
// the wire), and dim/hasDim carry the agreed dense dimension. Building the corrector and
// encoder in one place keeps a head node's query understanding identical to a broker's, the
// only difference being where the vocabulary and dimension are gathered from.
func newPipeline(feed func(*spell.Builder), dim int, hasDim bool) *pipeline {
	p := &pipeline{
		sel:      func(lang string) query.Analyzer { return lexical.ForLanguage(lang) },
		detector: langid.New(),
		expander: expand.Default(lexical.DefaultAnalyzer.Analyze),
	}

	b := spell.NewBuilder()
	feed(b)
	p.corrector = spell.NewQueryCorrector(b.BuildWithOptions(spell.DefaultOptions()))

	if hasDim {
		p.encoder = dense.NewStatic(dense.NewHashTable(dim, denseNonzero, denseSeed))
	}
	return p
}

// buildHeadPipeline constructs the query-understanding pipeline for a head node from its
// peers over the wire. A head holds no shards of its own, so it gathers the corrector's
// vocabulary by streaming each peer's /vocab and folding every peer's term frequencies into
// one builder, the cross-machine form of buildPipeline walking local shards: a term's df at
// the head is its fleet-wide df, so a correction is suggested the same way wherever a term
// lives. The dense dimension is agreed across the peers' reported dimensions, the same
// agree-or-off rule denseDim applies to shards, so the head encodes into a width every leaf
// can read or leaves the dense plane off. A peer that cannot stream its vocabulary, or that
// is unreachable for it, is skipped with its error returned to the caller so a head over a
// degraded fleet still builds a working corrector from the peers it could read rather than
// failing to start.
func buildHeadPipeline(ctx context.Context, peers []*search.RemoteSearcher) (*pipeline, []error) {
	var errs []error
	dim, ok := headDenseDim(peers)
	p := newPipeline(func(b *spell.Builder) {
		for _, peer := range peers {
			if err := peer.Vocab(ctx, func(term string, df uint32) { b.Add(term, df) }); err != nil {
				errs = append(errs, err)
			}
		}
	}, dim, ok)
	return p, errs
}

// headDenseDim agrees the dense dimension across a head node's peers the way denseDim agrees
// it across one broker's shards: the dense plane is on only when every peer that carries a
// vector region reports the same dimension, and a fleet with no region, or one whose peers
// disagree, leaves the head's encoder off rather than encode into a width some leaf cannot
// read.
func headDenseDim(peers []*search.RemoteSearcher) (int, bool) {
	dim, ok := 0, false
	for _, peer := range peers {
		d, has := peer.VectorDim()
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
