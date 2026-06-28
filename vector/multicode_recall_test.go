package vector

import (
	"math/rand"
	"sort"
	"testing"
)

// estimatorTopK brute-forces the multi-bit estimator over a corpus and returns the top
// k docIDs, the raw ranking power of the code at the given bit width with no graph and
// no rerank, so a recall measurement isolates the codec from everything else.
func estimatorTopK(codes []multiCode, qRot []float32, k int) []Result {
	all := make([]Result, len(codes))
	for i := range codes {
		all[i] = Result{DocID: uint32(i), Score: codes[i].estimate(qRot)}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].Score != all[j].Score {
			return all[i].Score > all[j].Score
		}
		return all[i].DocID < all[j].DocID
	})
	out := make([]Result, 0, k)
	for i := 0; i < k && i < len(all); i++ {
		out = append(out, all[i])
	}
	return out
}

// TestMultiBitEstimatorRecall measures the no-rerank codec's raw ranking recall at one,
// four, and five bits per dimension, with no graph and no rerank, on uniformly random
// queries (the hardest case, where true neighbors are nearly tied). The one-bit code is
// candidate grade: it finds the neighborhood, not the order. Five bits is the retrieval
// grade no-rerank knob and must clear recall@10 0.90, matching the int8 two-part path
// without storing the int8 copy. Four bits is the smaller, faster trade: far above one
// bit but below five, since uniform scalar quantization at four bits cannot reach the
// eight-bit rerank's order on tied neighbors. The test pins that ordering so the bit knob
// does what the spec promises.
func TestMultiBitEstimatorRecall(t *testing.T) {
	const dim, n = 64, 2000
	corpus := clusteredCorpus(n, dim, 30, 13)
	rot := newRotator(dim, defaultSeed)

	rotated := make([][]float32, n)
	for i, v := range corpus {
		rotated[i] = rot.rotate(v)
	}

	mean := func(bits int) float64 {
		codes := make([]multiCode, n)
		for i := range rotated {
			codes[i] = encodeMulti(rotated[i], bits)
		}
		rng := rand.New(rand.NewSource(7))
		var sum float64
		const queries = 100
		for q := 0; q < queries; q++ {
			query := normalize(randVec(rng, dim))
			want := trueTopK(corpus, query, 10)
			got := estimatorTopK(codes, rot.rotate(query), 10)
			sum += recallAt(got, want)
		}
		return sum / queries
	}

	r1 := mean(1)
	r4 := mean(4)
	r5 := mean(5)
	t.Logf("estimator recall@10: 1-bit %.3f, 4-bit %.3f, 5-bit %.3f", r1, r4, r5)

	if r5 < 0.90 {
		t.Fatalf("5-bit estimator recall@10 = %.3f, want >= 0.90 (no-rerank retrieval grade)", r5)
	}
	if r4 < 0.85 {
		t.Fatalf("4-bit estimator recall@10 = %.3f, want >= 0.85 (no-rerank smaller trade)", r4)
	}
	if r4 <= r1 {
		t.Fatalf("4-bit recall %.3f did not beat 1-bit %.3f", r4, r1)
	}
	if r5 < r4 {
		t.Fatalf("5-bit recall %.3f fell below 4-bit %.3f", r5, r4)
	}
}

// TestMultiBitRegion is the end-to-end gate for the no-rerank multi-bit region: build a
// five-bit region (which drops the int8 rerank copy), open it, and confirm the full HNSW
// walk plus the multi-bit estimator recovers the true neighbors. The graph adds a little
// loss on top of the codec's raw recall, so the region bar sits below the brute-force
// estimator's, but it must still clear retrieval grade. It also checks the region carries
// no rerank copy and that an unsupported bit width is refused at build.
func TestMultiBitRegion(t *testing.T) {
	const dim, n = 64, 1200
	corpus := clusteredCorpus(n, dim, 30, 21)
	b := NewBuilder(dim).WithCodeBits(5)
	for _, v := range corpus {
		b.Add(v)
	}
	region := mustBuild(t, b)
	r, err := Open(region)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if r.HasRerank() {
		t.Fatal("five-bit no-rerank region reports a rerank copy")
	}

	rng := rand.New(rand.NewSource(5))
	var sum float64
	const queries = 70
	for q := 0; q < queries; q++ {
		query := normalize(randVec(rng, dim))
		want := trueTopK(corpus, query, 10)
		got := r.Search(query, 10, 128, 100)
		sum += recallAt(got, want)
	}
	mean := sum / queries
	t.Logf("five-bit no-rerank region recall@10 = %.3f", mean)
	if mean < 0.88 {
		t.Fatalf("five-bit region recall@10 = %.3f, want >= 0.88", mean)
	}

	if _, err := NewBuilder(dim).WithCodeBits(3).Build(); err == nil {
		t.Fatal("WithCodeBits(3) built without error, want unsupported-bits failure")
	}
}

// TestMultiBitPackRoundTrip checks the bit packer and the byte-reading estimator agree
// with the in-memory code, so the region path scores identically to the build path.
func TestMultiBitPackRoundTrip(t *testing.T) {
	const dim = 96
	rng := rand.New(rand.NewSource(3))
	rot := newRotator(dim, 999)
	for _, bits := range []int{1, 4, 5} {
		for trial := 0; trial < 200; trial++ {
			oRot := rot.rotate(randVec(rng, dim))
			code := encodeMulti(oRot, bits)
			packed := packLevels(code.levels, bits)
			for i := range code.levels {
				if got := unpackLevel(packed, i, bits); got != code.levels[i] {
					t.Fatalf("bits %d trial %d dim %d: unpack %d, want %d", bits, trial, i, got, code.levels[i])
				}
			}
			q := rot.rotate(randVec(rng, dim))
			want := code.estimate(q)
			got := estimateMultiBytes(packed, bits, code.scalar, code.norm, q)
			if d := want - got; d > 1e-9 || d < -1e-9 {
				t.Fatalf("bits %d trial %d: estimateMultiBytes %v, want %v", bits, trial, got, want)
			}
		}
	}
}
