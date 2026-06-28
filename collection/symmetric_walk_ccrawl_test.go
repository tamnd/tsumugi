package collection_test

import (
	"math/rand"
	"os"
	"testing"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/vector"
)

// TestSymmetricWalkCCrawl carries the vector package's symmetric-walk measurement onto the
// real crawl. It embeds real page bodies with the static encoder, builds the same corpus
// into two VEC1 regions, one with the default int8-dot walk and one with spec doc 05's
// mode-1 symmetric one-bit Hamming walk, and measures how much of each region's own
// brute-force top-10 its graph walk recovers. Both regions carry the int8 rerank, so the
// only thing that changed is the navigation metric, and the comparison isolates the walk.
// On the real distribution as on the synthetic clustered corpus, the one-bit Hamming walk
// is the coarser compass: it recovers a real fraction but well under the int8-dot walk,
// which is why the int8-dot walk is the shipped default and the symmetric walk is the
// knob. This is the real-data half of the spec-amendment evidence (vector/hamming_test.go
// has the synthetic half).
func TestSymmetricWalkCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	enc := dense.NewStatic(dense.NewHashTable(denseDim, 8, 1))

	// Collect real page vectors, dropping pages whose body pools to the zero vector (all
	// terms unknown), the same selection the dense-encode test uses.
	const want = 2000
	var vecs [][]float32
	for seen := 0; seen < 80000 && len(vecs) < want; seen++ {
		doc, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read doc: %v", err)
		}
		if !ok {
			break
		}
		bodyTerms := lexical.Analyze(doc.Body)
		if len(bodyTerms) < 8 {
			continue
		}
		v := enc.Encode(bodyTerms)
		zero := true
		for _, x := range v {
			if x != 0 {
				zero = false
				break
			}
		}
		if zero {
			continue
		}
		vecs = append(vecs, v)
	}
	if len(vecs) < 300 {
		t.Skipf("only %d usable page vectors, too few to gate", len(vecs))
	}

	build := func(symmetric bool) *vector.Region {
		b := vector.NewBuilder(denseDim).WithSymmetricWalk(symmetric)
		for _, v := range vecs {
			b.Add(v)
		}
		raw, err := b.Build()
		if err != nil {
			t.Fatalf("build (symmetric=%v): %v", symmetric, err)
		}
		r, err := vector.Open(raw)
		if err != nil {
			t.Fatalf("open (symmetric=%v): %v", symmetric, err)
		}
		return r
	}
	base := build(false)
	sym := build(true)

	// Queries are real page vectors perturbed a little, standing in for a near-but-not-exact
	// dense query against the indexed bodies. graph-vs-brute on each region isolates the
	// walk's loss from the quantization the rerank shares.
	rng := rand.New(rand.NewSource(202))
	graphVsBrute := func(r *vector.Region) float64 {
		const queries = 200
		var sum float64
		for q := 0; q < queries; q++ {
			seed := vecs[rng.Intn(len(vecs))]
			query := make([]float32, len(seed))
			for i, x := range seed {
				query[i] = x + 0.05*float32(rng.NormFloat64())
			}
			brute := r.BruteForce(query, 10)
			bwant := make([]uint32, len(brute))
			for i, b := range brute {
				bwant[i] = b.DocID
			}
			got := r.Search(query, 10, 128, 100)
			set := map[uint32]bool{}
			for _, w := range bwant {
				set[w] = true
			}
			hit := 0
			for _, g := range got {
				if set[g.DocID] {
					hit++
				}
			}
			if len(bwant) > 0 {
				sum += float64(hit) / float64(len(bwant))
			}
		}
		return sum / queries
	}

	baseR := graphVsBrute(base)
	symR := graphVsBrute(sym)
	t.Logf("ccrawl pages=%d graph-vs-brute recall@10: int8-dot=%.3f hamming=%.3f", len(vecs), baseR, symR)

	if base.HasRerank() != true || sym.HasRerank() != true {
		t.Fatal("both regions should carry the int8 rerank copy")
	}
	// Same finding as the synthetic gate, now on real text: the int8-dot walk recovers more
	// of its own brute-force top-10 than the Hamming walk, the data reason it is the default.
	if baseR <= symR {
		t.Errorf("int8-dot walk %.3f should exceed hamming walk %.3f on real data", baseR, symR)
	}
	// The Hamming walk is still a functional coarse compass on real data, not noise.
	if symR < 0.40 {
		t.Errorf("symmetric walk graph-vs-brute %.3f, want >= 0.40 (functional floor)", symR)
	}
}
