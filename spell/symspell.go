// Package spell is tsumugi's query-side spell corrector, the SymSpell symmetric-delete
// algorithm doc 10 pins. A large fraction of real queries are misspelled, and a
// misspelled term misses the dictionary entirely so its posting list is never opened
// and it contributes nothing; the corrector maps the misspelling to the nearest real
// dictionary term so the query finds the documents the user meant.
//
// The dictionary is built from the corpus's own terms and their document frequencies,
// not an external word list, so the corrector knows the vocabulary the index actually
// holds (proper nouns, brand names, technical terms, the long tail of web vocabulary)
// and ranks corrections by how common they are in this corpus specifically. It is
// collection-wide, built by merging the shards' term dictionaries with their
// fleet-wide frequencies, because correction must be consistent across the fleet.
//
// SymSpell turns the distance computation into a hash lookup by precomputing, offline,
// the deletes of every dictionary term. A term within edit distance two of a query is
// reachable by deleting up to two characters from each side, so the build maps every
// delete-variant back to its originals and a query lookup is a few dozen hash probes
// whose count depends only on the query term's length, not the dictionary's size.
package spell

import (
	"sort"
)

// TermFreq is one dictionary entry the index is built from: the analyzed,
// dictionary-comparable term and its corpus document frequency, the same df the rank
// tie-break uses. Both come straight from the lexical region's term dictionary.
type TermFreq struct {
	Term string
	Freq uint64
}

// Suggestion is one verified, ranked correction candidate: the dictionary term, its
// Damerau-Levenshtein distance from the query term, and its corpus frequency.
type Suggestion struct {
	Term string
	Dist int
	Freq uint64
}

// Options tunes how the index is built over a corpus dictionary. The web's raw
// vocabulary has a long tail of one-off garbage tokens and very long non-words (URLs,
// hashes, run-together text), and indexing the deletes of all of them blows the delete
// index up far past the few hundred megabytes the spec budgets, so two practical
// cutoffs bound what enters the correction dictionary without changing the
// symmetric-delete algorithm itself: a maximum term length and a minimum frequency.
// Both are what real correctors apply, and they fit the spec's own rule that correction
// targets the real vocabulary, not every token the crawl happened to emit.
type Options struct {
	// MaxEdit is the maximum edit distance the index supports, which the canon fixes at
	// two.
	MaxEdit int
	// MaxLen skips terms longer than this many runes; a longer token is not a word a
	// user misspells and only bloats the delete index. Zero means no cap.
	MaxLen int
	// MinFreq skips terms with a fleet-wide document frequency below this, dropping the
	// long tail of one-off tokens so the corrector suggests real words. Zero keeps all.
	MinFreq uint64
}

// DefaultOptions is the web-scale build default: distance two, drop tokens past
// twenty-four runes, and require a term to appear in at least two documents before it
// can be a correction target, which removes the hapax-legomena tail that is mostly
// crawl noise.
func DefaultOptions() Options {
	return Options{MaxEdit: 2, MaxLen: 24, MinFreq: 2}
}

// Index is the resident SymSpell delete index plus the term frequencies, built once
// per collection and held at the broker, shared across all queries. The delete map
// points at term ids rather than strings so a term that appears in many delete-variants
// is stored once, which is what keeps the index to a few hundred megabytes at a few
// million terms rather than several times that.
type Index struct {
	maxEdit int
	maxLen  int
	deletes map[string][]uint32
	terms   []string
	freqs   []uint64
	byTerm  map[string]uint32
}

// Build constructs the SymSpell index over the corpus dictionary at the given maximum
// edit distance with no length or frequency cutoffs, the unbounded variant tests use on
// small dictionaries. Real collections build through BuildWithOptions so the long tail
// of web noise does not blow up the delete index.
func Build(terms []TermFreq, maxEdit int) *Index {
	return BuildWithOptions(terms, Options{MaxEdit: maxEdit})
}

// BuildWithOptions constructs the SymSpell index honoring the length and frequency
// cutoffs. It deduplicates terms by summing nothing and keeping the larger frequency
// (the caller's Builder already merged the fleet df), enumerates the deletes of every
// surviving term once, and maps each delete-variant back to the terms that produce it.
// This is the offline half of the algorithm: the expensive enumeration happens here,
// once, so the online lookup is hash probes only.
func BuildWithOptions(terms []TermFreq, o Options) *Index {
	if o.MaxEdit < 0 {
		o.MaxEdit = 0
	}
	ix := &Index{
		maxEdit: o.MaxEdit,
		maxLen:  o.MaxLen,
		deletes: make(map[string][]uint32),
		byTerm:  make(map[string]uint32, len(terms)),
	}
	for _, tf := range terms {
		if tf.Term == "" {
			continue
		}
		if tf.Freq < o.MinFreq {
			continue
		}
		if o.MaxLen > 0 && len([]rune(tf.Term)) > o.MaxLen {
			continue
		}
		if id, ok := ix.byTerm[tf.Term]; ok {
			// A term repeated across shards keeps the larger fleet frequency, the
			// merge the collection-wide dictionary performs.
			if tf.Freq > ix.freqs[id] {
				ix.freqs[id] = tf.Freq
			}
			continue
		}
		id := uint32(len(ix.terms))
		ix.byTerm[tf.Term] = id
		ix.terms = append(ix.terms, tf.Term)
		ix.freqs = append(ix.freqs, tf.Freq)
	}
	// Enumerate the deletes of every distinct term and index them by variant. The
	// term itself is the zero-delete variant, so an exact hit is found by the same
	// probe path as a near hit.
	var buf map[string]struct{}
	for id, term := range ix.terms {
		buf = deletesInto(buf, term, ix.maxEdit)
		for v := range buf {
			ix.deletes[v] = append(ix.deletes[v], uint32(id))
		}
	}
	return ix
}

// Len reports the number of distinct dictionary terms.
func (ix *Index) Len() int { return len(ix.terms) }

// Variants reports the number of distinct delete-variants in the index, the memory
// driver and the number tooling reports to size the corrector.
func (ix *Index) Variants() int { return len(ix.deletes) }

// Freq returns a term's corpus document frequency and whether the term is in the
// dictionary as typed. This is the as-typed presence check the correction policy runs
// first: a term present with a healthy frequency is used as-is, no correction.
func (ix *Index) Freq(term string) (uint64, bool) {
	id, ok := ix.byTerm[term]
	if !ok {
		return 0, false
	}
	return ix.freqs[id], true
}

// Lookup returns the dictionary terms within the index's maximum edit distance of the
// query term, verified by Damerau-Levenshtein and ranked by distance ascending then
// frequency descending, the rank doc 10 pins. The candidate set comes from the union
// of the delete-variant probes, which can over-generate, so each candidate's actual
// distance is computed and any beyond the maximum is discarded. The term tie-break on
// equal distance and frequency keeps the order deterministic across runs.
func (ix *Index) Lookup(term string) []Suggestion {
	if term == "" {
		return nil
	}
	// A query term longer than the longest indexed term by more than the edit distance
	// cannot be within range of any dictionary term, so skip the delete enumeration
	// entirely rather than spend it on a token no correction can reach. This also caps
	// the hot-path cost when a user pastes a very long token.
	if ix.maxLen > 0 && len([]rune(term)) > ix.maxLen+ix.maxEdit {
		return nil
	}
	seen := make(map[uint32]struct{})
	var out []Suggestion
	consider := func(id uint32) {
		if _, dup := seen[id]; dup {
			return
		}
		seen[id] = struct{}{}
		cand := ix.terms[id]
		d := damerauLevenshtein(term, cand, ix.maxEdit)
		if d < 0 || d > ix.maxEdit {
			return
		}
		out = append(out, Suggestion{Term: cand, Dist: d, Freq: ix.freqs[id]})
	}
	var buf map[string]struct{}
	buf = deletesInto(buf, term, ix.maxEdit)
	for v := range buf {
		for _, id := range ix.deletes[v] {
			consider(id)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Dist != out[j].Dist {
			return out[i].Dist < out[j].Dist
		}
		if out[i].Freq != out[j].Freq {
			return out[i].Freq > out[j].Freq
		}
		return out[i].Term < out[j].Term
	})
	return out
}

// Best returns the top-ranked correction for a query term, or false if none is within
// range. It is the common case the policy uses, the single suggestion to offer or apply.
func (ix *Index) Best(term string) (Suggestion, bool) {
	sugg := ix.Lookup(term)
	if len(sugg) == 0 {
		return Suggestion{}, false
	}
	return sugg[0], true
}

// deletesInto fills dst with every string obtained by deleting up to d characters from
// term, including term itself (zero deletes). It reuses dst across calls to avoid
// reallocating a fresh map per term during the build, which dominates allocation when
// indexing millions of terms. The enumeration works on runes so multi-byte characters
// delete as one character, which is what edit distance counts.
func deletesInto(dst map[string]struct{}, term string, d int) map[string]struct{} {
	if dst == nil {
		dst = make(map[string]struct{})
	} else {
		for k := range dst {
			delete(dst, k)
		}
	}
	dst[term] = struct{}{}
	if d == 0 {
		return dst
	}
	// Breadth-first over delete levels: each level deletes one more character from the
	// strings of the previous level. A frontier holds the new strings of the current
	// level so the next level only expands those, not the whole accumulated set.
	frontier := []string{term}
	for level := 0; level < d; level++ {
		var next []string
		for _, s := range frontier {
			rs := []rune(s)
			if len(rs) <= 1 {
				continue
			}
			for p := range rs {
				v := string(append(append([]rune{}, rs[:p]...), rs[p+1:]...))
				if _, ok := dst[v]; ok {
					continue
				}
				dst[v] = struct{}{}
				next = append(next, v)
			}
		}
		frontier = next
	}
	return dst
}

// damerauLevenshtein returns the optimal string alignment distance between a and b, the
// restricted Damerau-Levenshtein that counts an adjacent transposition as one edit, so
// "teh" for "the" scores one rather than the two a plain Levenshtein would charge. It
// returns -1 as soon as the running minimum of a row exceeds max, an early exit that
// keeps verification cheap because the corrector only cares whether a candidate is
// within the maximum, not its exact distance when it is far. The comparison is on runes
// so a multi-byte character is one symbol.
func damerauLevenshtein(a, b string, max int) int {
	ra, rb := []rune(a), []rune(b)
	la, lb := len(ra), len(rb)
	if la == 0 {
		if lb <= max {
			return lb
		}
		return -1
	}
	if lb == 0 {
		if la <= max {
			return la
		}
		return -1
	}
	// The length difference is a lower bound on the distance, so two terms whose
	// lengths differ by more than max cannot be within max.
	if diff := la - lb; diff > max || -diff > max {
		return -1
	}
	prevPrev := make([]int, lb+1)
	prev := make([]int, lb+1)
	cur := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		cur[0] = i
		rowMin := cur[0]
		for j := 1; j <= lb; j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			v := min3(cur[j-1]+1, prev[j]+1, prev[j-1]+cost)
			if i > 1 && j > 1 && ra[i-1] == rb[j-2] && ra[i-2] == rb[j-1] {
				if t := prevPrev[j-2] + 1; t < v {
					v = t
				}
			}
			cur[j] = v
			if v < rowMin {
				rowMin = v
			}
		}
		if rowMin > max {
			return -1
		}
		prevPrev, prev, cur = prev, cur, prevPrev
	}
	if prev[lb] > max {
		return -1
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}
