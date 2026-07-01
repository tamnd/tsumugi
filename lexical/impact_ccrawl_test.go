package lexical_test

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/lexical"
)

// impactForURL is the per-document impact the ccrawl gate orders lists by. A real build
// folds the composite static rank here; the gate only needs a deterministic byte with a
// realistic spread over the real documents, so it hashes the docID the same way the
// in-package test does.
func impactForURL(docID uint32) uint8 {
	h := docID*2654435761 + 12345
	return uint8(h >> 24)
}

// buildImpactCCrawl indexes the real crawl documents impact-ordered and returns the opened
// region alongside the documents, so the gate can score against the same corpus it built.
func buildImpactCCrawl(t *testing.T, docs []spimiDoc) *lexical.Region {
	t.Helper()
	b := lexical.NewBuilder(lexical.DefaultParams())
	for i, d := range docs {
		b.AddDoc(uint32(i), d.fields())
	}
	r, err := lexical.Open(b.BuildImpact(impactForURL))
	if err != nil {
		t.Fatalf("open impact region: %v", err)
	}
	return r
}

// buildInvertedOracle analyzes every document once into an independent term-to-docIDs map,
// the black-box oracle's own inverted index built without touching the region. Building it
// once and reusing it across queries keeps the oracle from re-analyzing the whole corpus
// per query, which is what makes the gate run in seconds rather than time out.
func buildInvertedOracle(docs []spimiDoc) map[string][]uint32 {
	inv := map[string][]uint32{}
	for id, d := range docs {
		docID := uint32(id)
		seen := map[string]bool{}
		for _, text := range []string{d.title, d.body, d.url} {
			for _, tok := range lexical.Analyze(text) {
				if seen[tok] {
					continue
				}
				seen[tok] = true
				inv[tok] = append(inv[tok], docID)
			}
		}
	}
	return inv
}

// naiveImpactCCrawl is the black-box oracle: coverage of the region-held query terms times
// the document's impact, top-k under score-descending, docID-ascending order, computed from
// the independent inverted index rather than the region.
func naiveImpactCCrawl(r *lexical.Region, inv map[string][]uint32, terms []string, k int) []lexical.Candidate {
	present := r.DocFreqsTerms(terms)
	seen := map[string]bool{}
	acc := map[uint32]int32{}
	for t := range present {
		if seen[t] {
			continue
		}
		seen[t] = true
		for _, docID := range inv[t] {
			acc[docID]++
		}
	}
	var cands []lexical.Candidate
	for docID, coverage := range acc {
		cands = append(cands, lexical.Candidate{DocID: docID, Score: coverage * int32(impactForURL(docID))})
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Score != cands[j].Score {
			return cands[i].Score > cands[j].Score
		}
		return cands[i].DocID < cands[j].DocID
	})
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands
}

// TestImpactCCrawl proves the impact format and the pruned traversal on real crawl data, the
// skewed language and posting-length distribution the engine is meant to serve. It builds the
// real documents impact-ordered, draws queries from the corpus's most frequent terms so most
// queries hit several multi-block lists, and checks the served path returns exactly the naive
// coverage-times-impact oracle. The served path is the pruned early-termination traversal:
// SearchImpactTerms runs prunedImpact, so a match here is the pruned traversal proven correct
// on real data, at both a broad k and the small k where the early stop discards the tail. It
// runs without the race detector because the ccrawl build times out under it; the in-package
// tests cover the traversal under -race on synthetic corpora and gate it against the
// exhaustive scan directly.
func TestImpactCCrawl(t *testing.T) {
	docs := ccrawlSpimiDocs(t, 20000)
	if len(docs) == 0 {
		t.Skip("no ccrawl documents")
	}
	r := buildImpactCCrawl(t, docs)
	inv := buildInvertedOracle(docs)

	// The scorer is exercised through the exported search path; a result on a real query
	// also confirms the real term distribution decoded.
	if _, err := r.SearchImpact("the", 10); err != nil {
		t.Fatalf("smoke search: %v", err)
	}

	// Draw a vocabulary from the corpus's most frequent terms so queries hit several lists.
	type tf struct {
		term string
		df   uint32
	}
	var vocab []tf
	r.ForEachTerm(func(term string, df uint32) {
		if df >= 5 {
			vocab = append(vocab, tf{term, df})
		}
	})
	sort.Slice(vocab, func(i, j int) bool { return vocab[i].df > vocab[j].df })
	if len(vocab) > 400 {
		vocab = vocab[:400]
	}
	if len(vocab) < 4 {
		t.Skipf("too few frequent terms in ccrawl sample: %d", len(vocab))
	}

	rng := rand.New(rand.NewSource(97))
	queries := 150
	// Both a broad k and a small k: at k=10 the top-k settles early on the frequent terms, so
	// the pruned traversal skips the tail, exactly where an off-by-one in the stop bound would
	// change the result. The oracle is independent of k, so both must still match it exactly.
	for _, k := range []int{100, 10} {
		for q := 0; q < queries; q++ {
			n := 1 + rng.Intn(4)
			terms := make([]string, n)
			for i := range terms {
				terms[i] = vocab[rng.Intn(len(vocab))].term
			}
			got, err := r.SearchImpactTerms(terms, k)
			if err != nil {
				t.Fatalf("search %v k=%d: %v", terms, k, err)
			}
			want := naiveImpactCCrawl(r, inv, terms, k)
			if len(got) != len(want) {
				t.Fatalf("query %v k=%d: got %d results, want %d", terms, k, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("query %v k=%d result %d: got %+v want %+v", terms, k, i, got[i], want[i])
				}
			}
		}
	}
	t.Logf("pruned impact traversal matched the coverage-times-impact oracle over %d queries at k=100 and k=10 on %d ccrawl docs",
		queries, len(docs))
}
