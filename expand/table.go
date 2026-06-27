// Package expand is the cheap, exact, table-driven query expansion doc 10's "query
// expansion" section keeps query-side. The expensive semantic expansion is done offline
// and doc-side by the learned-sparse plane, which expands a document about cars to also
// carry "automobile" and "vehicle" as real impact postings, so the query side never
// repeats that work. What stays on the query is the high-precision kind a curated table
// gives in microseconds: acronyms ("nyc" matches "new york city") and curated synonyms
// (spelling variants like "color" and "colour", common abbreviations, brand variants).
//
// The expansion adds the alternative forms to a query term as optional OR alternatives
// with the original term kept, so "nyc" becomes "nyc OR (new york city)" and the scoring
// sorts out which documents match better. The table is curated, not learned, and applied
// query-side only, the asymmetry doc 04 pinned: the analysis chain is shared between the
// build and the query, but correction and expansion run on the query alone.
package expand

import (
	"sort"
	"strings"
)

// Analyze is the analysis-chain function the table runs over each curated form so a
// stored key and alternative match a dictionary term byte for byte. It is the same
// chain the build and the query share; passing it in keeps this package free of a
// dependency on the lexical package.
type Analyze func(text string) []string

// Table is a curated expansion table: it maps an analyzed single-token key to the
// alternative forms that key expands to, each alternative an analyzed form joined by a
// single space when it is multi-word ("new york city"). Lookups are the query hot path,
// a single map probe; building runs the analyzer over the curated forms once offline.
//
// Only single-token forms become keys, because the query is expanded one analyzed term
// at a time and a multi-token form ("new york city") cannot be matched against one
// query term. A pair of single-token forms is therefore bidirectional ("color" finds
// "colour" and the reverse), while an acronym paired with a multi-word expansion keys
// only on the acronym side ("nyc" finds "new york city"); the reverse, collapsing a
// typed phrase to its acronym, is a phrase rewrite the per-term alternative slice does
// not carry and the learned doc-side plane covers.
type Table struct {
	alts map[string][]string
}

// Group is one curated equivalence set: a set of forms that mean the same thing, any of
// which should match the others. {"nyc", "new york city"} and {"color", "colour"} are
// groups. The order inside a group does not matter; every single-token member keys onto
// the other members.
type Group []string

// Build compiles the curated groups into a Table, running the analyzer over every form
// so the stored keys and alternatives match dictionary terms exactly. Each single-token
// member of a group keys onto the analyzed forms of the other members, deduplicated and
// sorted for a deterministic, order-stable expansion. A form that analyzes to nothing
// (all punctuation, say) is skipped, and a group with fewer than two distinct analyzed
// forms contributes no entry.
func Build(groups []Group, a Analyze) *Table {
	t := &Table{alts: make(map[string][]string)}
	for _, g := range groups {
		// Analyze every raw form once into its space-joined canonical token sequence,
		// dropping forms that analyze to nothing and de-duplicating identical forms.
		forms := make([]string, 0, len(g))
		seen := map[string]bool{}
		for _, raw := range g {
			f := strings.Join(a(raw), " ")
			if f == "" || seen[f] {
				continue
			}
			seen[f] = true
			forms = append(forms, f)
		}
		if len(forms) < 2 {
			continue
		}
		for _, key := range forms {
			// Only a single-token form can be looked up against one analyzed query term.
			if strings.IndexByte(key, ' ') >= 0 {
				continue
			}
			for _, other := range forms {
				if other == key {
					continue
				}
				t.alts[key] = append(t.alts[key], other)
			}
		}
	}
	// Sort and dedup each entry so a key that appears in two groups merges cleanly and
	// the expansion order is stable across builds.
	for key, vs := range t.alts {
		t.alts[key] = dedupSorted(vs)
	}
	return t
}

// Expand returns the curated alternatives for one analyzed query term, or nil when the
// term is not in the table, which is the common case: most terms expand to nothing and
// the lookup is a single map probe that returns nil. The returned slice is owned by the
// table and must not be mutated by the caller.
func (t *Table) Expand(term string) []string {
	if t == nil {
		return nil
	}
	return t.alts[term]
}

// Len reports the number of keyed terms in the table, the count of terms that expand to
// at least one alternative.
func (t *Table) Len() int {
	if t == nil {
		return 0
	}
	return len(t.alts)
}

func dedupSorted(vs []string) []string {
	if len(vs) <= 1 {
		return vs
	}
	sort.Strings(vs)
	out := vs[:1]
	for _, v := range vs[1:] {
		if v != out[len(out)-1] {
			out = append(out, v)
		}
	}
	return out
}
