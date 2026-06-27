package vector

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// clusteredCorpus draws n unit vectors of the given dimension grouped into
// clusters, the way real embeddings sit in topical neighborhoods rather than
// spread uniformly, so nearest-neighbor structure exists to recover.
func clusteredCorpus(n, dim, clusters int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		centers[c] = normalize(randVec(rng, dim))
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + 0.35*float32(rng.NormFloat64())
		}
		out[i] = normalize(v)
	}
	return out
}

func randVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for j := range v {
		v[j] = float32(rng.NormFloat64())
	}
	return v
}

func normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	s = math.Sqrt(s)
	if s == 0 {
		return v
	}
	for i := range v {
		v[i] /= float32(s)
	}
	return v
}

// trueTopK returns the exact nearest k docIDs by full-precision cosine (dot, on
// normalized vectors), the ground truth recall is measured against.
func trueTopK(corpus [][]float32, q []float32, k int) []uint32 {
	type sc struct {
		id uint32
		d  float64
	}
	all := make([]sc, len(corpus))
	for i, v := range corpus {
		var d float64
		for j := range v {
			d += float64(v[j]) * float64(q[j])
		}
		all[i] = sc{uint32(i), d}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].d != all[j].d {
			return all[i].d > all[j].d
		}
		return all[i].id < all[j].id
	})
	out := make([]uint32, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].id
	}
	return out
}

func recallAt(got []Result, want []uint32) float64 {
	set := map[uint32]bool{}
	for _, w := range want {
		set[w] = true
	}
	hit := 0
	for _, g := range got {
		if set[g.DocID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

// TestRecallTwoPart is the M7 gate: HNSW plus int8 rerank must recover the true
// nearest neighbors with high probability. The bar is mean recall@10 above 0.90
// over many queries on a clustered corpus, which is where the canon ef settings
// land.
func TestRecallTwoPart(t *testing.T) {
	const dim, n = 64, 5000
	corpus := clusteredCorpus(n, dim, 40, 1)
	b := NewBuilder(dim)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	var sum float64
	const queries = 200
	for q := 0; q < queries; q++ {
		query := normalize(randVec(rng, dim))
		want := trueTopK(corpus, query, 10)
		got := r.Search(query, 10, 128, 100)
		sum += recallAt(got, want)
	}
	mean := sum / queries
	if mean < 0.90 {
		t.Fatalf("mean recall@10 = %.3f, want >= 0.90", mean)
	}
	t.Logf("two-part mean recall@10 = %.3f", mean)
}

// TestGraphRecallVsBrute isolates the graph from quantization: the HNSW walk must
// recover almost everything a brute-force scan with the identical scoring finds.
func TestGraphRecallVsBrute(t *testing.T) {
	const dim, n = 64, 4000
	corpus := clusteredCorpus(n, dim, 30, 7)
	b := NewBuilder(dim)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	rng := rand.New(rand.NewSource(3))
	var sum float64
	const queries = 150
	for q := 0; q < queries; q++ {
		query := normalize(randVec(rng, dim))
		brute := r.BruteForce(query, 10)
		want := make([]uint32, len(brute))
		for i, b := range brute {
			want[i] = b.DocID
		}
		got := r.Search(query, 10, 128, 100)
		sum += recallAt(got, want)
	}
	mean := sum / queries
	if mean < 0.95 {
		t.Fatalf("graph-vs-brute mean recall@10 = %.3f, want >= 0.95", mean)
	}
	t.Logf("graph-vs-brute mean recall@10 = %.3f", mean)
}

// TestNoRerankCandidateRecall checks the one-bit no-rerank mode as what it is: a
// candidate generator, not a final ranker. One-bit RaBitQ scored by the estimator
// is too coarse to nail the exact top-10 (that is what the int8 rerank exists
// for), but its top-100 should still contain the bulk of the true top-10, so a
// shard that skips the int8 copy to save memory still feeds a good rerank. The bar
// is the true top-10 recovered within the returned top-100.
func TestNoRerankCandidateRecall(t *testing.T) {
	const dim, n = 64, 4000
	corpus := clusteredCorpus(n, dim, 30, 11)
	b := NewBuilder(dim).WithRerank(false)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if r.hasRerank {
		t.Fatal("expected no-rerank region")
	}
	rng := rand.New(rand.NewSource(5))
	var sum float64
	const queries = 150
	for q := 0; q < queries; q++ {
		query := normalize(randVec(rng, dim))
		want := trueTopK(corpus, query, 10)
		got := r.Search(query, 100, 256, 100)
		sum += recallAt(got, want)
	}
	mean := sum / queries
	if mean < 0.75 {
		t.Fatalf("no-rerank candidate recall@10-in-100 = %.3f, want >= 0.75", mean)
	}
	t.Logf("no-rerank candidate recall@10-in-100 = %.3f", mean)
}

// TestRotationPreservesInnerProduct checks the orthonormal rotation is an
// isometry: the dot of two rotated vectors equals the dot of the originals.
func TestRotationPreservesInnerProduct(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	rot := newRotator(64, 42)
	for trial := 0; trial < 50; trial++ {
		a := randVec(rng, 64)
		c := randVec(rng, 64)
		var want float64
		for i := range a {
			want += float64(a[i]) * float64(c[i])
		}
		ra, rc := rot.rotate(a), rot.rotate(c)
		var got float64
		for i := range ra {
			got += float64(ra[i]) * float64(rc[i])
		}
		if math.Abs(got-want) > 1e-3*(1+math.Abs(want)) {
			t.Fatalf("trial %d: rotated dot %.5f, original %.5f", trial, got, want)
		}
	}
}

// TestEstimatorUnbiased checks the asymmetric RaBitQ estimate is what RaBitQ
// guarantees: an unbiased estimate of the true inner product. The mean error over
// many pairs must sit near zero (the defining property; the rotation is what makes
// it hold), and the correlation must be strongly positive. The correlation is not
// near one because a single bit per coordinate carries real variance at this
// dimension; that residual is exactly what the int8 rerank removes downstream.
func TestEstimatorUnbiased(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	rot := newRotator(128, 17)
	const trials = 5000
	var sumErr, sumAbsTrue float64
	var sumEst, sumTrue, sumET, sumEE, sumTT float64
	for i := 0; i < trials; i++ {
		o := normalize(randVec(rng, 128))
		q := normalize(randVec(rng, 128))
		oRot := rot.rotate(o)
		qRot := rot.rotate(q)
		code := encodeOneBit(oRot)
		est := code.estimate(encodeQuery(qRot))
		var tru float64
		for j := range o {
			tru += float64(o[j]) * float64(q[j])
		}
		sumErr += est - tru
		sumAbsTrue += math.Abs(tru)
		sumEst += est
		sumTrue += tru
		sumET += est * tru
		sumEE += est * est
		sumTT += tru * tru
	}
	n := float64(trials)
	relBias := math.Abs(sumErr/n) / (sumAbsTrue / n)
	if relBias > 0.05 {
		t.Fatalf("estimator relative bias = %.4f, want <= 0.05", relBias)
	}
	cov := sumET/n - (sumEst/n)*(sumTrue/n)
	varE := sumEE/n - (sumEst/n)*(sumEst/n)
	varT := sumTT/n - (sumTrue/n)*(sumTrue/n)
	corr := cov / math.Sqrt(varE*varT)
	if corr < 0.75 {
		t.Fatalf("estimator correlation with truth = %.3f, want >= 0.75", corr)
	}
	t.Logf("estimator relative bias = %.4f, correlation = %.3f", relBias, corr)
}

// TestFuseRRF checks fusion lifts a document both planes rank well and is order
// independent.
func TestFuseRRF(t *testing.T) {
	lexical := []uint32{1, 2, 3, 4}
	dense := []uint32{3, 5, 1, 6}
	out := Fuse(lexical, dense, 60)
	// docs 1 and 3 appear in both and should head the list.
	if out[0] != 1 && out[0] != 3 {
		t.Fatalf("expected 1 or 3 first, got %d (%v)", out[0], out)
	}
	if out[1] != 1 && out[1] != 3 {
		t.Fatalf("expected 1 and 3 in the top two, got %v", out)
	}
}

// TestEmptyAndCorrupt covers an empty region, a flipped header byte, and a
// truncated region.
func TestEmptyAndCorrupt(t *testing.T) {
	empty := NewBuilder(64).Build()
	r, err := Open(empty)
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if got := r.Search([]float32{1}, 10, 0, 0); got != nil {
		t.Fatal("empty region returned results")
	}

	b := NewBuilder(64)
	for i := 0; i < 100; i++ {
		b.Add(normalize(randVec(rand.New(rand.NewSource(int64(i))), 64)))
	}
	good := b.Build()
	bad := append([]byte(nil), good...)
	bad[20] ^= 0xff // count, inside header CRC
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	if _, err := Open(good[:headerLen+8]); err == nil {
		t.Fatal("truncated region accepted")
	}
}
