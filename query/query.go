// Package query is the query-understanding front of tsumugi: it turns a raw query
// string into the ParsedQuery the retrieval planes consume. Everything that happens
// to a query between the search box and the Block-Max Pruning walk lives here, the
// short cheap pipeline doc 10 specifies: parse the operators, run the shared analysis
// chain, and assemble the parsed query with its normalized cache key.
//
// The work is done once, at the broker, and the result is shipped to the shards, so a
// shard never re-analyzes the query string. That is the spec's analyze-once rule: the
// analysis chain runs one time per query, not once per shard the fan-out visits, so
// its cost is paid one time even when a query touches fifty shards.
package query

import (
	"encoding/binary"
	"math"
	"sort"
	"strings"
)

// Analyzer is the analysis chain the parser runs over each piece's text. The build
// and the query share one analyzer so a query term matches a dictionary term byte for
// byte; lexical.Analyzer satisfies this interface. Keeping it an interface lets the
// parser be tested without the index and keeps the package free of a build dependency.
type Analyzer interface {
	Analyze(text string) []string
}

// Semantics is the global match mode: soft-OR scoring by default, or hard-AND opt-in.
type Semantics uint8

const (
	// SoftOR scores documents that match any term, the default web-search semantics.
	SoftOR Semantics = iota
	// HardAND requires every free term, the opt-in strict mode.
	HardAND
)

// AnyField is the Field value for a term with no field scope: it may occur in any field.
const AnyField int8 = -1

// fieldIDs maps the field operators a user can type to the field ids the index uses,
// the same ids lexical.Field assigns (title 0, body 1, url 2, anchor 3). A field:
// prefix naming anything not here is treated as free text, the forgiving-parser rule.
var fieldIDs = map[string]int8{
	"title":  0,
	"body":   1,
	"url":    2,
	"anchor": 3,
}

// QueryTerm is one analyzed, dictionary-comparable token with the retrieval metadata
// doc 04's traversal multiplies and matches against: a query-side weight, a field
// scope (AnyField for unscoped), and optional expansion alternatives.
type QueryTerm struct {
	Term   string
	Weight float32
	Field  int8
	Alts   []string
}

// Phrase is a quoted span: an ordered sequence of analyzed terms that must appear
// adjacent and in order, with Slop zero for a strict phrase and greater than zero for
// a proximity window.
type Phrase struct {
	Terms []string
	Slop  int
}

// FilterKind is the stored field a filter narrows against.
type FilterKind uint8

const (
	// FilterHost narrows to documents on a host or domain, the site: operator.
	FilterHost FilterKind = iota
)

// Filter narrows the candidate set against a stored field without contributing score.
type Filter struct {
	Kind  FilterKind
	Value string
}

// ParsedQuery is the output of query understanding and the single thing handed to L0
// retrieval. Every field is the product of a step in the pipeline: Terms from analysis,
// Required and Excluded and Phrases and Filters from parsing, Lang from detection,
// Mode from the query or the default, and NormKey from normalization for the cache.
// DenseVec is the optional dense-plane input, nil when the dense plane is off.
type ParsedQuery struct {
	Terms    []QueryTerm
	Required []QueryTerm
	Excluded []QueryTerm
	Phrases  []Phrase
	Filters  []Filter

	DenseVec []byte

	Lang    string
	Mode    Semantics
	NormKey string

	// LangConfident records whether the language detector was sure of Lang. When it is
	// false the query was analyzed with the script-based default rather than a
	// per-language chain, the spec's rule that a low-confidence guess must not drive
	// analysis; the broker can use it to decide whether to trust Lang for anything else.
	LangConfident bool

	// Corrected is set when spell correction auto-substituted at least one term, so the
	// broker can surface a "showing results for" notice. Suggestion holds the
	// did-you-mean rendering when a correction was offered rather than applied; it is
	// empty when nothing was suggested.
	Corrected  bool
	Suggestion string
}

// Corrector corrects a single analyzed query term, the SymSpell corrector the broker
// holds. It is an interface returning primitives so the query package depends on
// neither the spell package nor a shared type; spell.QueryCorrector satisfies it
// structurally. ok is false to leave a term as typed, the common case for a correctly
// spelled query; auto is true to substitute the replacement, false to only offer it.
type Corrector interface {
	Correct(term string) (replacement string, auto bool, ok bool)
}

// ApplyCorrection runs the corrector over the free and required terms after parsing,
// the broker step doc 10 places between analysis and retrieval. An auto-correction
// substitutes the term in place and marks the query corrected; a did-you-mean
// correction leaves the term as typed and contributes to the Suggestion string the
// broker can offer. The cache key is recomputed when an auto-correction changed the
// terms, so the corrected query caches under its corrected form.
func (pq *ParsedQuery) ApplyCorrection(c Corrector) {
	if c == nil {
		return
	}
	changed := false
	suggested := false
	var sugg strings.Builder
	writeSugg := func(term string) {
		if sugg.Len() > 0 {
			sugg.WriteByte(' ')
		}
		sugg.WriteString(term)
	}
	fix := func(terms []QueryTerm) {
		for i := range terms {
			repl, auto, ok := c.Correct(terms[i].Term)
			if !ok || repl == "" || repl == terms[i].Term {
				writeSugg(terms[i].Term)
				continue
			}
			if auto {
				terms[i].Term = repl
				changed = true
				writeSugg(repl)
			} else {
				suggested = true
				writeSugg(repl)
			}
		}
	}
	fix(pq.Terms)
	fix(pq.Required)
	if changed {
		pq.Corrected = true
		pq.NormKey = pq.normKey()
	}
	if suggested {
		pq.Suggestion = sugg.String()
	}
}

// Expander returns the curated expansion alternatives for one analyzed query term, the
// acronym and synonym table the broker holds. Like Corrector it is an interface
// returning primitives so the query package depends on neither the expand package nor a
// shared type; expand.Table satisfies it structurally. It returns nil for a term with
// no expansion, the common case.
type Expander interface {
	Expand(term string) []string
}

// ApplyExpansion fills each free and required term's Alts with the curated alternatives
// the table holds, the light query-side expansion doc 10 keeps after correction. The
// original term is always kept; the alternatives are optional OR forms the traversal
// matches alongside it, so "nyc" carries "new york city" as an alternative and the
// scoring decides which document matches better. Excluded terms are not expanded,
// because broadening a must-not would drop documents the user did not ask to exclude.
// The cache key is recomputed so the expanded query caches distinctly from the bare one,
// the spec's rule that the key is computed after expansion. A nil expander is a no-op.
func (pq *ParsedQuery) ApplyExpansion(e Expander) {
	if e == nil {
		return
	}
	changed := false
	expand := func(terms []QueryTerm) {
		for i := range terms {
			alts := e.Expand(terms[i].Term)
			if len(alts) == 0 {
				continue
			}
			terms[i].Alts = mergeAlts(terms[i].Alts, alts, terms[i].Term)
			if len(terms[i].Alts) > 0 {
				changed = true
			}
		}
	}
	expand(pq.Terms)
	expand(pq.Required)
	if changed {
		pq.NormKey = pq.normKey()
	}
}

// DenseEncoder turns the query's analyzed terms into the dense-plane query vector, the
// full-precision embedding the shards' ANN index recalls against. Like Corrector and
// Expander it is an interface returning primitives so the query package depends on
// neither the dense package nor a shared type; dense.StaticEncoder satisfies it
// structurally. It returns a vector of the index's kept dimension, or a nil or zero
// vector for a query whose terms carry no dense signal.
type DenseEncoder interface {
	Encode(terms []string) []float32
}

// ApplyDense runs the dense encoder over the query's terms and packs the result into
// DenseVec, the broker step that fills the dense plane's input when the dense plane is
// on. It encodes the free and required terms together, the bag of words the static
// encoder mean-pools, and stores the full-precision float32 vector as its little-endian
// byte form: the shard's vector reader takes a float32 query and rotates and quantizes it
// internally, so the wire form is the lossless float32 vector rather than a pre-quantized
// one. An all-zero vector is the dense plane's no-signal value and is left unstored so the
// cache key still reads the query as dense-off, and a nil encoder is a no-op. The cache
// key is recomputed so a dense query caches distinctly from the same string without the
// dense plane.
func (pq *ParsedQuery) ApplyDense(enc DenseEncoder) {
	if enc == nil {
		return
	}
	terms := make([]string, 0, len(pq.Terms)+len(pq.Required))
	for _, t := range pq.Terms {
		terms = append(terms, t.Term)
	}
	for _, t := range pq.Required {
		terms = append(terms, t.Term)
	}
	vec := enc.Encode(terms)
	if !anyNonzero(vec) {
		return
	}
	pq.DenseVec = EncodeDenseVec(vec)
	pq.NormKey = pq.normKey()
}

// anyNonzero reports whether a vector carries any signal, the test that distinguishes a
// real dense query from the encoder's zero-vector no-signal return.
func anyNonzero(vec []float32) bool {
	for _, x := range vec {
		if x != 0 {
			return true
		}
	}
	return false
}

// EncodeDenseVec packs a full-precision dense vector into its little-endian float32 byte
// form, the wire shape ParsedQuery.DenseVec carries to the shards. It is lossless: each
// float32 becomes four bytes, so the shard decodes exactly the vector the encoder
// produced before the reader rotates and quantizes it.
func EncodeDenseVec(vec []float32) []byte {
	out := make([]byte, len(vec)*4)
	for i, x := range vec {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(x))
	}
	return out
}

// DecodeDenseVec unpacks the little-endian float32 byte form back into a vector, the
// inverse of EncodeDenseVec the shard side uses to recover the query vector from
// ParsedQuery.DenseVec. A byte slice whose length is not a multiple of four is malformed
// and decodes to nil rather than a truncated vector.
func DecodeDenseVec(b []byte) []float32 {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// mergeAlts appends the new alternatives to a term's existing alts, dropping any that
// equal the term itself or a form already present, so expansion is idempotent and never
// lists the original among its own alternatives. The result keeps first-seen order.
func mergeAlts(existing, add []string, term string) []string {
	seen := map[string]bool{term: true}
	out := existing
	for _, a := range existing {
		seen[a] = true
	}
	for _, a := range add {
		if a == "" || seen[a] {
			continue
		}
		seen[a] = true
		out = append(out, a)
	}
	return out
}

// piece is one raw fragment the operator scan produces, before the analysis chain runs
// over its text. The kind records the operator the scan recognized; the text is still
// raw, because operators are recognized on the raw string but the term content inside
// them must still go through the analysis chain.
type piece struct {
	kind  pieceKind
	field int8
	text  string
}

type pieceKind uint8

const (
	pieceFree pieceKind = iota
	pieceRequired
	pieceExcluded
	pieceFieldScoped
	pieceFilterHost
	piecePhrase
)

// scan walks the raw string left to right and splits it into operator pieces, the
// parse step doc 10 pins. It recognizes a quoted span as a phrase, a +token as
// required, a -token as excluded, site:value as a host filter, and field:value as a
// field scope, and everything else as free text. The scan is deliberately forgiving:
// an unclosed quote treats the rest as a phrase, an unknown field falls back to free
// text, and a lone + or - with no term attached is dropped, because a search box
// should answer a slightly-wrong query rather than error on it.
func scan(raw string) []piece {
	var pieces []piece
	i := 0
	n := len(raw)
	for i < n {
		// Skip the whitespace between tokens.
		for i < n && isSpace(raw[i]) {
			i++
		}
		if i >= n {
			break
		}
		if raw[i] == '"' {
			// A quote opens a phrase; read to the closing quote, or to the end if the
			// quote is never closed, the forgiving unclosed-quote rule.
			i++
			start := i
			for i < n && raw[i] != '"' {
				i++
			}
			span := raw[start:i]
			if i < n {
				i++ // consume the closing quote
			}
			if strings.TrimSpace(span) != "" {
				pieces = append(pieces, piece{kind: piecePhrase, field: AnyField, text: span})
			}
			continue
		}
		// An ordinary token runs to the next whitespace.
		start := i
		for i < n && !isSpace(raw[i]) {
			i++
		}
		pieces = append(pieces, classify(raw[start:i]))
	}
	return pieces
}

// classify turns one whitespace-delimited token into its piece, recognizing the +/-
// prefixes and the site:/field: operators and falling back to free text otherwise.
func classify(tok string) piece {
	switch tok[0] {
	case '+':
		rest := tok[1:]
		if rest == "" {
			return piece{kind: pieceFree, field: AnyField, text: ""} // a lone +, dropped downstream
		}
		return scopedOr(rest, pieceRequired)
	case '-':
		rest := tok[1:]
		if rest == "" {
			return piece{kind: pieceFree, field: AnyField, text: ""}
		}
		return scopedOr(rest, pieceExcluded)
	}
	return scopedOr(tok, pieceFree)
}

// scopedOr recognizes the colon operators in a token. site:value becomes a host
// filter, a known field:value becomes a field scope that keeps the base +/- intent and
// carries the field id, so a plain field:x scores in that field, a +field:x is a
// field-scoped must, and a -field:x a field-scoped exclusion; anything else (no colon,
// or an unknown field) is the base kind over the whole token, the forgiving fallback.
func scopedOr(tok string, base pieceKind) piece {
	colon := strings.IndexByte(tok, ':')
	if colon <= 0 || colon == len(tok)-1 {
		return piece{kind: base, field: AnyField, text: tok}
	}
	key := strings.ToLower(tok[:colon])
	val := tok[colon+1:]
	if key == "site" {
		return piece{kind: pieceFilterHost, field: AnyField, text: val}
	}
	if fid, ok := fieldIDs[key]; ok {
		// A free field scope gets its own kind so it lands in Terms with the field set; a
		// required or excluded field scope stays required or excluded and carries the field.
		k := base
		if base == pieceFree {
			k = pieceFieldScoped
		}
		return piece{kind: k, field: fid, text: val}
	}
	// An unknown field is not an operator; treat the whole token as free text.
	return piece{kind: base, field: AnyField, text: tok}
}

// Parse parses a raw query string and runs the analysis chain over the text inside
// each operator, producing the ParsedQuery the retrieval planes consume. Parsing
// before analysis is what preserves the operator structure: analyzing first would
// split a quoted phrase into free terms and lose it, and would not know which terms
// were required. The mode is the caller's default semantics, soft-OR for web search.
func Parse(raw string, a Analyzer, mode Semantics) *ParsedQuery {
	pq := &ParsedQuery{Mode: mode}
	for _, p := range scan(raw) {
		switch p.kind {
		case pieceFree:
			for _, t := range a.Analyze(p.text) {
				pq.Terms = append(pq.Terms, QueryTerm{Term: t, Weight: 1, Field: p.field})
			}
		case pieceRequired:
			for _, t := range a.Analyze(p.text) {
				pq.Required = append(pq.Required, QueryTerm{Term: t, Weight: 1, Field: p.field})
			}
		case pieceExcluded:
			for _, t := range a.Analyze(p.text) {
				pq.Excluded = append(pq.Excluded, QueryTerm{Term: t, Weight: 1, Field: p.field})
			}
		case pieceFieldScoped:
			for _, t := range a.Analyze(p.text) {
				pq.Terms = append(pq.Terms, QueryTerm{Term: t, Weight: 1, Field: p.field})
			}
		case pieceFilterHost:
			if v := canonHost(p.text); v != "" {
				pq.Filters = append(pq.Filters, Filter{Kind: FilterHost, Value: v})
			}
		case piecePhrase:
			terms := a.Analyze(p.text)
			if len(terms) > 0 {
				pq.Phrases = append(pq.Phrases, Phrase{Terms: terms, Slop: 0})
			}
		}
	}
	pq.NormKey = pq.normKey()
	return pq
}

// Detector identifies a query's language, the n-gram identifier the broker holds. It
// is an interface returning primitives so the query package depends on neither the
// langid package nor a shared type; langid.Detector satisfies it structurally. lang is
// the detected language code, empty for no signal or a low-confidence Latin guess;
// confident reports whether the detection is sure enough to drive per-language
// analysis rather than fall back to the script-based default.
type Detector interface {
	DetectLang(text string) (lang string, confident bool)
}

// AnalyzerFor selects the analyzer a detected language is analyzed with, the registry
// the broker passes so the query package never imports the analyzer registry. The same
// selector on the build side and the query side is what guarantees a document and a
// query in one language are analyzed identically; lexical.ForLanguage wrapped in a
// closure is the canonical implementation.
type AnalyzerFor func(lang string) Analyzer

// ParseDetected is the language-routing entry point: it detects the query's language,
// selects that language's analyzer through sel, and parses with it, so language routing
// falls out of analyzer selection rather than a separate router, the shape doc 10
// pins. The detected language and its confidence are recorded on the parsed query. A
// low-confidence detection routes on the empty language, which sel maps to the
// script-based default, so a wrong guess never drives a per-language stem or fold. The
// selector is the analyzer source, the same role the explicit Analyzer plays in Parse,
// so it must be non-nil and return a non-nil analyzer; lexical.ForLanguage wrapped in a
// closure is the canonical sel and never returns nil. A nil detector is safe and means
// no language was detected: the query routes to the default analyzer with an empty Lang.
func ParseDetected(raw string, sel AnalyzerFor, det Detector, mode Semantics) *ParsedQuery {
	lang := ""
	confident := false
	if det != nil {
		lang, confident = det.DetectLang(raw)
	}
	// The analyzer is chosen on the confident language; a low-confidence guess routes
	// to the default by passing the empty language, never the guessed one.
	routeLang := lang
	if !confident {
		routeLang = ""
	}
	pq := Parse(raw, sel(routeLang), mode)
	pq.Lang = lang
	pq.LangConfident = confident
	return pq
}

// LexicalTerms returns the plain term strings the lexical plane retrieves on: the
// soft-OR free terms followed by the required terms, deduplicated in first-seen order.
// Excluded terms are not retrieved on; they filter the candidate set. This is the term
// set the broker analyzes once and ships to the shards.
func (pq *ParsedQuery) LexicalTerms() []string {
	seen := map[string]bool{}
	var out []string
	add := func(qt QueryTerm) {
		if qt.Term == "" || seen[qt.Term] {
			return
		}
		seen[qt.Term] = true
		out = append(out, qt.Term)
	}
	for _, t := range pq.Terms {
		add(t)
	}
	for _, t := range pq.Required {
		add(t)
	}
	return out
}

// RetrievalTerms is the term set the broker actually retrieves on: the lexical terms
// followed by every term's curated expansion alternatives, deduplicated in first-seen
// order. Folding the alternatives in is what makes expansion broaden recall under the
// soft-OR semantics, where an alternative is simply another term that contributes to the
// score, so a document matching only "colour" still answers a query for "color". The base
// terms come first so a query with no expansion returns exactly LexicalTerms.
func (pq *ParsedQuery) RetrievalTerms() []string {
	out := pq.LexicalTerms()
	seen := make(map[string]bool, len(out))
	for _, t := range out {
		seen[t] = true
	}
	addAlts := func(terms []QueryTerm) {
		for _, t := range terms {
			for _, a := range t.Alts {
				if a == "" || seen[a] {
					continue
				}
				seen[a] = true
				out = append(out, a)
			}
		}
	}
	addAlts(pq.Terms)
	addAlts(pq.Required)
	return out
}

// Empty reports whether the query carries nothing to retrieve on: no lexical terms and
// no phrases. The broker short-circuits an empty query to an empty result rather than
// routing it, the spec's empty-query behavior.
func (pq *ParsedQuery) Empty() bool {
	return len(pq.Terms) == 0 && len(pq.Required) == 0 && len(pq.Phrases) == 0
}

// normKey builds the canonical, order-stable cache key doc 10 pins: it captures
// everything that changes the result and nothing that does not, so two raw queries
// differing only in casing, spacing, or unordered term order collide on one entry and
// two different queries never do. The free term set is sorted because soft-OR is
// order-independent; phrases keep their order because a phrase is ordered; filters and
// excluded terms are sorted; the mode is included because it changes the result.
func (pq *ParsedQuery) normKey() string {
	var b strings.Builder
	terms := termStrings(pq.Terms)
	sort.Strings(terms)
	b.WriteString("t:")
	b.WriteString(strings.Join(terms, ","))
	req := termStrings(pq.Required)
	sort.Strings(req)
	b.WriteString(";+:")
	b.WriteString(strings.Join(req, ","))
	exc := termStrings(pq.Excluded)
	sort.Strings(exc)
	b.WriteString(";-:")
	b.WriteString(strings.Join(exc, ","))
	b.WriteString(";ph:")
	for i, ph := range pq.Phrases {
		if i > 0 {
			b.WriteByte('|')
		}
		b.WriteString(strings.Join(ph.Terms, " "))
	}
	filters := make([]string, len(pq.Filters))
	for i, f := range pq.Filters {
		filters[i] = filterKey(f)
	}
	sort.Strings(filters)
	b.WriteString(";f:")
	b.WriteString(strings.Join(filters, ","))
	b.WriteString(";m:")
	if pq.Mode == HardAND {
		b.WriteString("and")
	} else {
		b.WriteString("or")
	}
	b.WriteString(";d:")
	if pq.DenseVec != nil {
		b.WriteString("1")
	} else {
		b.WriteString("0")
	}
	return b.String()
}

// termStrings projects the term strings out of a query-term slice. A field-scoped term
// carries its field into the key so title:rust and a plain rust do not collide, and an
// expanded term carries its alternatives so a query expanded against the table caches
// distinctly from the bare query, the spec's rule that the key is computed after
// expansion. The alternatives are emitted in their stored order, which Build already
// sorted, so the key stays stable.
func termStrings(ts []QueryTerm) []string {
	out := make([]string, len(ts))
	for i, t := range ts {
		s := t.Term
		if t.Field != AnyField {
			s = fieldName(t.Field) + ":" + t.Term
		}
		if len(t.Alts) > 0 {
			s += "=(" + strings.Join(t.Alts, "|") + ")"
		}
		out[i] = s
	}
	return out
}

func filterKey(f Filter) string {
	if f.Kind == FilterHost {
		return "site:" + f.Value
	}
	return f.Value
}

func fieldName(id int8) string {
	for name, fid := range fieldIDs {
		if fid == id {
			return name
		}
	}
	return "?"
}

// canonHost lowercases a host filter value and strips a leading scheme or www. so
// site:Example.com and site:www.example.com narrow to the same host.
func canonHost(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	v = strings.TrimPrefix(v, "http://")
	v = strings.TrimPrefix(v, "https://")
	v = strings.TrimPrefix(v, "www.")
	v = strings.TrimSuffix(v, "/")
	return v
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
