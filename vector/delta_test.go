package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// splitCorpus builds a region over the first split vectors of a clustered corpus and a
// delta over the rest, the freshness arrangement the union search serves: an immutable
// shard that was packed once, plus the recent documents that arrived after. It returns
// the opened region, its delta, and the full corpus so a test can score the union
// against the ground truth over every document. The delta's docIDs continue the
// immutable docIDs (split, split+1, ...), so a corpus index is a global docID.
func splitCorpus(tb testing.TB, dim, n, split, clusters int, seed int64, opt func(*Builder)) (*Region, *Delta, [][]float32) {
	tb.Helper()
	corpus := clusteredCorpus(n, dim, clusters, seed)
	b := NewBuilder(dim)
	if opt != nil {
		opt(b)
	}
	for _, v := range corpus[:split] {
		b.Add(v)
	}
	r, err := Open(mustBuild(tb, b))
	if err != nil {
		tb.Fatalf("open: %v", err)
	}
	d := r.NewDelta()
	for i := split; i < n; i++ {
		id, err := d.Add(corpus[i])
		if err != nil {
			tb.Fatalf("delta add: %v", err)
		}
		if id != uint32(i) {
			tb.Fatalf("delta docID = %d, want %d", id, i)
		}
	}
	return r, d, corpus
}

// TestDeltaUnionRecall is the freshness gate: documents that arrived after the shard was
// packed (the delta) must be as findable as the documents baked into the immutable
// region. The union search of the immutable graph plus the in-RAM delta graph must
// recover the true nearest neighbors over the whole corpus with high probability, and
// because the queries are drawn near random corpus points, a good fraction of each
// query's true top-10 lives in the delta, so this fails if the delta half is not
// genuinely searched.
func TestDeltaUnionRecall(t *testing.T) {
	const dim, n, split = 64, 2000, 1500
	_, d, corpus := splitCorpus(t, dim, n, split, 30, 11, nil)

	rng := rand.New(rand.NewSource(99))
	var sum, deltaShare float64
	const queries = 80
	for q := 0; q < queries; q++ {
		query := normalize(randVec(rng, dim))
		want := trueTopK(corpus, query, 10)
		for _, id := range want {
			if int(id) >= split {
				deltaShare++
			}
		}
		got := d.Search(query, 10, 128, 64, 100)
		sum += recallAt(got, want)
	}
	mean := sum / queries
	deltaFrac := deltaShare / float64(queries*10)
	t.Logf("union mean recall@10 = %.3f, true-top-10 in delta = %.1f%%", mean, deltaFrac*100)
	if deltaFrac < 0.10 {
		t.Fatalf("only %.1f%% of true neighbors in delta, test would not exercise the union", deltaFrac*100)
	}
	if mean < 0.85 {
		t.Fatalf("union mean recall@10 = %.3f, want >= 0.85", mean)
	}
}

// TestDeltaFreshnessImmediate checks the point of the buffer: a document added to the
// delta is its own nearest neighbor on the very next search, with no rebuild. It adds a
// fresh vector and then queries with a tiny perturbation of it; the fresh docID must
// come back rank one.
func TestDeltaFreshnessImmediate(t *testing.T) {
	const dim, n, split = 64, 1500, 1200
	r, d, _ := splitCorpus(t, dim, n, split, 25, 5, nil)
	_ = r

	rng := rand.New(rand.NewSource(7))
	fresh := normalize(randVec(rng, dim))
	id, err := d.Add(fresh)
	if err != nil {
		t.Fatalf("add fresh: %v", err)
	}
	query := make([]float32, dim)
	for i, x := range fresh {
		query[i] = x + 0.01*float32(rng.NormFloat64())
	}
	got := d.Search(normalize(query), 5, 128, 64, 100)
	if len(got) == 0 || got[0].DocID != id {
		t.Fatalf("fresh doc %d not rank one: got %+v", id, got)
	}
}

// TestDeltaTombstone checks that a delete takes effect on the next search, for a document
// in the immutable region and for one in the delta. A tombstoned docID must never appear
// in results even though its vector is still in the index.
func TestDeltaTombstone(t *testing.T) {
	const dim, n, split = 64, 1600, 1200
	_, d, corpus := splitCorpus(t, dim, n, split, 20, 3, nil)

	// Pick one immutable victim and one delta victim, then query right at each so it would
	// otherwise come back rank one.
	immVictim := uint32(42)
	deltaVictim := uint32(split + 17)
	d.Delete(immVictim)
	d.Delete(deltaVictim)

	for _, victim := range []uint32{immVictim, deltaVictim} {
		got := d.Search(corpus[victim], 10, 128, 64, 100)
		for _, g := range got {
			if g.DocID == victim {
				t.Fatalf("tombstoned doc %d still returned: %+v", victim, got)
			}
		}
		if len(got) == 0 {
			t.Fatalf("query at victim %d returned nothing", victim)
		}
	}
}

// TestDeltaCompactEqualsFullBuild is the compaction correctness and determinism gate.
// The immutable docs were added in docID order and the delta docs in insertion order, so
// folding the live union back through the Builder visits the corpus in exactly its
// original order; with no tombstones the rebuilt region must therefore be byte-identical
// to a region built from the whole corpus in one pass. Byte equality proves both that
// compaction is deterministic and that it reconstructs precisely the shard a from-scratch
// pack would have produced.
func TestDeltaCompactEqualsFullBuild(t *testing.T) {
	const dim, n, split = 64, 1000, 700
	r, d, corpus := splitCorpus(t, dim, n, split, 20, 13, nil)

	source := func(id uint32) []float32 { return corpus[id] }
	got, err := d.Compact(source)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	// Determinism: a second compaction of the same live set is byte-identical.
	got2, err := d.Compact(source)
	if err != nil {
		t.Fatalf("compact again: %v", err)
	}
	if !bytes.Equal(got, got2) {
		t.Fatal("compaction is not deterministic: two runs differ")
	}

	full := NewBuilder(dim)
	for _, v := range corpus {
		full.Add(v)
	}
	want := mustBuild(t, full)
	if !bytes.Equal(got, want) {
		t.Fatalf("compacted region (%d bytes) != full build (%d bytes)", len(got), len(want))
	}
	_ = r
}

// TestDeltaCompactDropsTombstones checks that compaction reclaims deleted documents: the
// rebuilt region holds exactly the live count and a search over it never surfaces a
// tombstoned docID. The renumbering is also exercised, since dropping a document shifts
// every later docID down by one.
func TestDeltaCompactDropsTombstones(t *testing.T) {
	const dim, n, split = 64, 1000, 700
	_, d, corpus := splitCorpus(t, dim, n, split, 20, 21, nil)

	// Delete a handful across both halves.
	deleted := map[uint32]bool{7: true, 300: true, 699: true, uint32(split): true, uint32(n - 1): true}
	for id := range deleted {
		d.Delete(id)
	}
	live := n - len(deleted)

	raw, err := d.Compact(func(id uint32) []float32 { return corpus[id] })
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	cr, err := Open(raw)
	if err != nil {
		t.Fatalf("open compacted: %v", err)
	}
	if cr.Count() != live {
		t.Fatalf("compacted count = %d, want %d", cr.Count(), live)
	}

	// The surviving vectors, in the order compaction kept them, are the new docID order.
	var survivors [][]float32
	for i := 0; i < n; i++ {
		if !deleted[uint32(i)] {
			survivors = append(survivors, corpus[i])
		}
	}
	rng := rand.New(rand.NewSource(3))
	for q := 0; q < 40; q++ {
		// Query at a surviving vector; its new docID is its index in survivors, and that
		// must come back rank one, proving the renumbering and the rebuild are consistent.
		idx := rng.Intn(len(survivors))
		got := cr.Search(survivors[idx], 5, 128, 100)
		if len(got) == 0 || got[0].DocID != uint32(idx) {
			t.Fatalf("survivor %d not rank one after compaction: got %+v", idx, got)
		}
	}
}

// TestDeltaModesUnion runs the union recall gate across the region's distance modes, so
// the delta's mirroring of each mode (the symmetric Hamming walk, the multi-bit no-rerank
// estimator, and the one-bit no-rerank estimator) is exercised, not just the default
// two-part int8 path. Each mode keeps the union searchable; the floors differ because the
// modes differ in sharpness, the same ordering the immutable-only gates measure.
func TestDeltaModesUnion(t *testing.T) {
	const dim, n, split = 64, 1200, 900
	cases := []struct {
		name string
		opt  func(*Builder)
		// floor is recall@100, since the no-rerank modes are candidate generators measured
		// at depth, matching TestNoRerankCandidateRecall.
		k     int
		floor float64
	}{
		// The symmetric one-bit Hamming walk is the coarse compass (vector/hamming_test.go
		// floors it at 0.45 on the synthetic corpus); it stays a functional floor here, well
		// under the int8-dot default, the same finding the immutable-only gate records.
		{"symmetric", func(b *Builder) { b.WithSymmetricWalk(true) }, 10, 0.45},
		{"multibit5", func(b *Builder) { b.WithCodeBits(5) }, 100, 0.80},
		{"onebit-norerank", func(b *Builder) { b.WithRerank(false) }, 100, 0.80},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, d, corpus := splitCorpus(t, dim, n, split, 25, 31, tc.opt)
			rng := rand.New(rand.NewSource(55))
			var sum float64
			const queries = 60
			for q := 0; q < queries; q++ {
				query := normalize(randVec(rng, dim))
				want := trueTopK(corpus, query, 10)
				got := d.Search(query, tc.k, 128, 64, 100)
				sum += recallAt(got, want)
			}
			mean := sum / queries
			t.Logf("%s union recall@10-in-top-%d = %.3f", tc.name, tc.k, mean)
			if mean < tc.floor {
				t.Fatalf("%s union recall = %.3f, want >= %.3f", tc.name, mean, tc.floor)
			}
		})
	}
}

// TestDeltaDimMismatch checks the guard: Add rejects a vector of the wrong width rather
// than corrupting the buffer.
func TestDeltaDimMismatch(t *testing.T) {
	b := NewBuilder(64)
	for i := 0; i < 50; i++ {
		b.Add(normalize(randVec(rand.New(rand.NewSource(int64(i))), 64)))
	}
	r, err := Open(mustBuild(t, b))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	d := r.NewDelta()
	if _, err := d.Add(make([]float32, 32)); err != ErrDeltaDim {
		t.Fatalf("Add wrong-dim error = %v, want ErrDeltaDim", err)
	}
}
