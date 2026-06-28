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

// TestDeltaFreshnessCCrawl carries the freshness path onto the real crawl. It embeds real
// page bodies, packs the first part into an immutable VEC1 region, and feeds the rest into
// an in-RAM delta, the way a serving shard takes new documents after it was packed. It then
// measures, over real page-derived queries, how much of the true top-10 over the whole
// corpus the union search recovers, and confirms the documents that landed in the delta are
// recovered at the same rate as the ones baked into the immutable region. Finally it
// compacts the delta and checks the rebuilt region holds the whole live corpus, the fold
// that returns the freshness buffer to a single immutable shard. This is the real-data half
// of the delta evidence; vector/delta_test.go has the synthetic half.
func TestDeltaFreshnessCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer func() { _ = src.Close() }()

	enc := dense.NewStatic(dense.NewHashTable(denseDim, 8, 1))

	// Collect real page vectors, dropping pages that pool to the zero vector, the same
	// selection the other real-data dense tests use.
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
	if len(vecs) < 600 {
		t.Skipf("only %d usable page vectors, too few to gate", len(vecs))
	}

	n := len(vecs)
	split := n * 3 / 4

	b := vector.NewBuilder(denseDim)
	for _, v := range vecs[:split] {
		b.Add(v)
	}
	raw, err := b.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	r, err := vector.Open(raw)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	d := r.NewDelta()
	for i := split; i < n; i++ {
		if _, err := d.Add(vecs[i]); err != nil {
			t.Fatalf("delta add: %v", err)
		}
	}

	// trueTopK is the exact nearest k over the whole corpus by full-precision dot, the
	// ground truth the union is scored against.
	trueTopK := func(q []float32, k int) []uint32 {
		type sc struct {
			id uint32
			d  float64
		}
		all := make([]sc, n)
		for i, v := range vecs {
			var dot float64
			for j := range v {
				dot += float64(v[j]) * float64(q[j])
			}
			all[i] = sc{uint32(i), dot}
		}
		for a := 0; a < k; a++ {
			best := a
			for c := a + 1; c < n; c++ {
				if all[c].d > all[best].d || (all[c].d == all[best].d && all[c].id < all[best].id) {
					best = c
				}
			}
			all[a], all[best] = all[best], all[a]
		}
		out := make([]uint32, k)
		for i := 0; i < k; i++ {
			out[i] = all[i].id
		}
		return out
	}

	rng := rand.New(rand.NewSource(202))
	var sum, deltaShare, deltaHit, deltaWant float64
	const queries = 150
	for q := 0; q < queries; q++ {
		seed := vecs[rng.Intn(n)]
		query := make([]float32, len(seed))
		for i, x := range seed {
			query[i] = x + 0.05*float32(rng.NormFloat64())
		}
		want := trueTopK(query, 10)
		gotSet := map[uint32]bool{}
		for _, g := range d.Search(query, 10, 128, 64, 100) {
			gotSet[g.DocID] = true
		}
		hit := 0
		for _, w := range want {
			if int(w) >= split {
				deltaWant++
				if gotSet[w] {
					deltaHit++
				}
			}
			if gotSet[w] {
				hit++
			}
		}
		sum += float64(hit) / 10
		for _, w := range want {
			if int(w) >= split {
				deltaShare++
			}
		}
	}
	mean := sum / queries
	deltaFrac := deltaShare / float64(queries*10)
	deltaRecall := 0.0
	if deltaWant > 0 {
		deltaRecall = deltaHit / deltaWant
	}
	t.Logf("ccrawl delta union: pages=%d split=%d recall@10=%.3f, true-top-10 in delta=%.1f%%, delta-only recall=%.3f",
		n, split, mean, deltaFrac*100, deltaRecall)

	if deltaFrac < 0.05 {
		t.Fatalf("only %.1f%% of true neighbors in delta, the union is not exercised", deltaFrac*100)
	}
	if mean < 0.80 {
		t.Errorf("ccrawl union recall@10 = %.3f, want >= 0.80", mean)
	}
	// The whole point of the buffer: a document in the delta is recovered about as well as
	// one in the immutable region, not penalized for arriving late.
	if deltaRecall < 0.70 {
		t.Errorf("ccrawl delta-only recall = %.3f, want >= 0.70 (fresh docs as findable as baked-in)", deltaRecall)
	}

	// Compaction folds the delta back into a single immutable region over the live union.
	raw2, err := d.Compact(func(id uint32) []float32 { return vecs[id] })
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	cr, err := vector.Open(raw2)
	if err != nil {
		t.Fatalf("open compacted: %v", err)
	}
	if cr.Count() != n {
		t.Fatalf("compacted count = %d, want %d", cr.Count(), n)
	}
}
