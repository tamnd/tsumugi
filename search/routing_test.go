package search

import (
	"fmt"
	"sort"
	"testing"
)

// refRoute is the reference routing the front-coded index must match exactly: the plain
// map-based union the index replaced, kept here as the oracle the equivalence test compares
// against so a regression in the front-coded lookup is caught against the obvious
// implementation.
func refRoute(postings map[string][]int, always []int, numShard int, terms []string) []int {
	if len(terms) == 0 {
		out := make([]int, numShard)
		for i := range out {
			out[i] = i
		}
		return out
	}
	seen := make([]bool, numShard)
	var out []int
	add := func(si int) {
		if !seen[si] {
			seen[si] = true
			out = append(out, si)
		}
	}
	for _, si := range always {
		add(si)
	}
	for _, t := range terms {
		for _, si := range postings[t] {
			add(si)
		}
	}
	return out
}

// makeVocab builds a deterministic term-to-shards map over a synthetic vocabulary spread
// across numShard shards, so the index is exercised over many front-coded blocks with
// shared prefixes (the "term000123" shape front-codes heavily) and varied per-term fan-out.
func makeVocab(numTerms, numShard int) (map[string][]int, []string) {
	postings := make(map[string][]int, numTerms)
	terms := make([]string, 0, numTerms)
	for i := 0; i < numTerms; i++ {
		t := fmt.Sprintf("term%06d", i)
		terms = append(terms, t)
		// Each term lands on a deterministic spread of shards: a base shard plus a stride,
		// so fan-out varies from one shard to several and the shard ids are not contiguous.
		fan := 1 + i%4
		var sh []int
		for j := 0; j < fan; j++ {
			sh = append(sh, (i*7+j*13)%numShard)
		}
		postings[t] = sh
	}
	sort.Strings(terms)
	return postings, terms
}

// TestRoutingMatchesReferenceMap is the equivalence gate: the front-coded, bloom-fronted
// index routes every term set exactly as the plain map-based union would, for present terms,
// absent terms, mixed sets, single terms, and the empty set. This is what lets the data
// structure change without changing any routing decision, so the broker's results are
// untouched.
func TestRoutingMatchesReferenceMap(t *testing.T) {
	const numTerms, numShard = 5000, 64
	postings, terms := makeVocab(numTerms, numShard)
	always := []int{3, 17}
	ri := NewRoutingIndex(postings, always, numShard)

	eq := func(name string, in []string) {
		want := refRoute(postings, always, numShard, in)
		got := ri.RouteTerms(in)
		sort.Ints(want)
		sort.Ints(got)
		if len(want) != len(got) {
			t.Fatalf("%s: routed %v, want %v", name, got, want)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("%s: routed %v, want %v", name, got, want)
			}
		}
	}

	// Every present term, one at a time, must route to exactly its shard set plus always.
	for _, term := range terms {
		eq("single:"+term, []string{term})
	}
	// Absent terms route to nothing but the always set, the bloom-reject path.
	eq("absent", []string{"zzz_not_present", "term999999", "", "aaaaa"})
	// Mixed present and absent.
	eq("mixed", []string{"term000000", "nope", "term004999", "term002500"})
	// A wide set spanning many blocks.
	wide := make([]string, 0, 200)
	for i := 0; i < numTerms; i += 25 {
		wide = append(wide, fmt.Sprintf("term%06d", i))
	}
	eq("wide", wide)
	// The empty set routes to every shard.
	eq("empty", nil)
}

// TestRoutingDictLookup checks the front-coded dictionary maps each term to its sorted
// ordinal and rejects absent terms, including the block-boundary anchors and terms that
// sort between two present terms.
func TestRoutingDictLookup(t *testing.T) {
	terms := []string{"apple", "apply", "banana", "band", "bandana", "candle", "cane", "zebra"}
	d := newRoutingDict(terms)
	if d.len() != len(terms) {
		t.Fatalf("len = %d, want %d", d.len(), len(terms))
	}
	for i, term := range terms {
		if got := d.lookup(term); got != i {
			t.Fatalf("lookup(%q) = %d, want %d", term, got, i)
		}
	}
	for _, absent := range []string{"", "ant", "appl", "applz", "bandanas", "candl", "zz", "zebraa"} {
		if got := d.lookup(absent); got != -1 {
			t.Fatalf("lookup(%q) = %d, want -1 (absent)", absent, got)
		}
	}
}

// TestRoutingDictBlocks exercises the front coding across many blocks: a vocabulary larger
// than one block must look up every term and reject the gaps, proving the binary search over
// anchors and the forward block scan compose.
func TestRoutingDictBlocks(t *testing.T) {
	const n = routingBlockSize*7 + 5 // several full blocks plus a partial tail
	terms := make([]string, n)
	for i := range terms {
		terms[i] = fmt.Sprintf("w%05d", i*2) // even numbers only, so odd-numbered are absent
	}
	d := newRoutingDict(terms)
	for i, term := range terms {
		if got := d.lookup(term); got != i {
			t.Fatalf("lookup(%q) = %d, want %d", term, got, i)
		}
	}
	for i := 0; i < n*2; i += 2 {
		gap := fmt.Sprintf("w%05d", i+1) // the odd numbers, between present terms
		if got := d.lookup(gap); got != -1 {
			t.Fatalf("lookup(%q) = %d, want -1", gap, got)
		}
	}
}

// TestRoutingBloomNoFalseNegatives checks the bloom front never rejects a present term: a
// false negative would drop a shard from a route, the one error a routing bloom must never
// make. False positives are allowed, they only cost a dictionary lookup that returns absent.
func TestRoutingBloomNoFalseNegatives(t *testing.T) {
	const n = 20000
	b := newRoutingBloom(n, 0.01)
	terms := make([]string, n)
	for i := range terms {
		terms[i] = fmt.Sprintf("token-%d", i)
		b.add(terms[i])
	}
	for _, term := range terms {
		if !b.mayContain(term) {
			t.Fatalf("bloom rejected a present term %q (false negative)", term)
		}
	}
	// A nil bloom filters nothing, so the dictionary lookup always runs.
	var nb *routingBloom
	if !nb.mayContain("anything") {
		t.Fatal("nil bloom should not reject")
	}
}

// TestRoutingEmptyVocab checks a routing index over an empty vocabulary routes lexical
// queries to nothing but the always set and the empty query to every shard, the degenerate
// case a fresh collection or an all-impact-quantized collection hits.
func TestRoutingEmptyVocab(t *testing.T) {
	ri := NewRoutingIndex(map[string][]int{}, []int{2}, 5)
	if got := ri.RouteTerms([]string{"anything"}); len(got) != 1 || got[0] != 2 {
		t.Fatalf("empty-vocab lexical route = %v, want [2] (always only)", got)
	}
	if got := ri.RouteTerms(nil); len(got) != 5 {
		t.Fatalf("empty-vocab empty query routed %d shards, want 5", len(got))
	}
}

// TestRoutingShardSetsSortedDeduped checks the index sorts and dedups a term's shard set
// regardless of the order the source map listed them in, so the routed set is deterministic.
func TestRoutingShardSetsSortedDeduped(t *testing.T) {
	postings := map[string][]int{"x": {5, 1, 5, 3, 1}}
	ri := NewRoutingIndex(postings, nil, 8)
	got := ri.RouteTerms([]string{"x"})
	want := []int{1, 3, 5}
	if len(got) != len(want) {
		t.Fatalf("routed %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("routed %v, want %v (sorted, deduped)", got, want)
		}
	}
}

// TestRoutingSizeBeatsMap measures the front-coded index against a Go map of the same
// vocabulary and checks the front-coded form is materially smaller, the scale property the
// refactor exists for (doc 11: the routing index must stay small relative to the shards at
// 100,000 shards). It is a coarse estimate, but the front-coded blocks plus the bloom must
// beat the map's per-entry string and slice headers by a wide margin.
func TestRoutingSizeBeatsMap(t *testing.T) {
	const numTerms, numShard = 50000, 1000
	postings, _ := makeVocab(numTerms, numShard)
	ri := NewRoutingIndex(postings, nil, numShard)

	// A Go map's lower bound: per entry, the string header (16) plus the key bytes, the
	// slice header (24), and the shard ids (8 each on a 64-bit map[string][]int), plus a
	// conservative bucket overhead. This undercounts the real map, so beating it is a
	// conservative claim.
	mapBytes := 0
	for t, sh := range postings {
		mapBytes += 16 + len(t) + 24 + len(sh)*8 + 16
	}
	idxBytes := ri.sizeBytes()
	t.Logf("routing index: %d bytes front-coded vs ~%d bytes map (%.1fx smaller)",
		idxBytes, mapBytes, float64(mapBytes)/float64(idxBytes))
	if idxBytes >= mapBytes {
		t.Fatalf("front-coded index %d bytes not smaller than the map's ~%d bytes", idxBytes, mapBytes)
	}
}
