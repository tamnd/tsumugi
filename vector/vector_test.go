package vector

import (
	"math"
	"math/rand"
	"runtime"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/codec"
)

// mustBuild builds a region and fails the test on a build error, the convenient
// form for the many tests that expect a normal corpus to build cleanly.
func mustBuild(tb testing.TB, b *Builder) []byte {
	tb.Helper()
	region, err := b.Build()
	if err != nil {
		tb.Fatalf("build: %v", err)
	}
	return region
}

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
	const dim, n = 64, 3500
	corpus := clusteredCorpus(n, dim, 40, 1)
	b := NewBuilder(dim)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(mustBuild(t, b))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	var sum float64
	const queries = 120
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
	const dim, n = 64, 3000
	corpus := clusteredCorpus(n, dim, 30, 7)
	b := NewBuilder(dim)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(mustBuild(t, b))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	rng := rand.New(rand.NewSource(3))
	var sum float64
	const queries = 100
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
	const dim, n = 64, 3000
	corpus := clusteredCorpus(n, dim, 30, 11)
	b := NewBuilder(dim).WithRerank(false)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(mustBuild(t, b))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if r.hasRerank {
		t.Fatal("expected no-rerank region")
	}
	rng := rand.New(rand.NewSource(5))
	var sum float64
	const queries = 100
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

// TestAsymmetricCosineSharperThanSymmetric pins the gain from scoring the rerank
// asymmetrically. Cosine now dots the full-precision query against the dequantized
// int8 document, so only the document carries quantization error; the old form
// quantized the query too. The test compares both against the true full-precision
// cosine over many query/document pairs and requires the asymmetric form's mean
// absolute error to be the smaller, the recovered sharpness the spec asks for.
func TestAsymmetricCosineSharperThanSymmetric(t *testing.T) {
	const dim, n = 96, 800
	corpus := clusteredCorpus(n, dim, 20, 4)
	b := NewBuilder(dim)
	for _, v := range corpus {
		b.Add(v)
	}
	r, err := Open(mustBuild(t, b))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	rdim := int(r.h.rdim)
	rng := rand.New(rand.NewSource(21))
	var asymErr, symErr float64
	const trials = 500
	for trial := 0; trial < trials; trial++ {
		docID := rng.Intn(n)
		// A query near a corpus vector with added noise, the realistic rerank input.
		query := make([]float32, dim)
		for j := range query {
			query[j] = corpus[docID][j] + 0.15*float32(rng.NormFloat64())
		}
		query = normalize(query)

		// True cosine in full precision against the original document vector.
		var dot, qn, dn float64
		for j := range query {
			dot += float64(query[j]) * float64(corpus[docID][j])
			qn += float64(query[j]) * float64(query[j])
			dn += float64(corpus[docID][j]) * float64(corpus[docID][j])
		}
		trueCos := dot / (math.Sqrt(qn) * math.Sqrt(dn))

		// Asymmetric cosine: the value the reader now returns.
		asym, ok := r.Cosine(query, uint32(docID))
		if !ok {
			t.Fatalf("trial %d: cosine unavailable", trial)
		}

		// Symmetric cosine: the old form, query quantized to int8 too.
		qRot := r.rot.rotate(query)
		qi8 := r.iq.encodeQuery(qRot)
		row := r.rerank[docID*rdim : (docID+1)*rdim]
		sqn, sdn := normI8(qi8), normI8(row)
		var sym float64
		if sqn != 0 && sdn != 0 {
			sym = float64(dotI8(qi8, row)) / (sqn * sdn)
		}

		asymErr += math.Abs(asym - trueCos)
		symErr += math.Abs(sym - trueCos)
	}
	asymErr /= trials
	symErr /= trials
	if asymErr >= symErr {
		t.Fatalf("asymmetric mean abs error %.5f not better than symmetric %.5f", asymErr, symErr)
	}
	t.Logf("mean abs cosine error: asymmetric %.5f, symmetric %.5f (%.1f%% lower)",
		asymErr, symErr, 100*(symErr-asymErr)/symErr)
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
	empty := mustBuild(t, NewBuilder(64))
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
	good := mustBuild(t, b)
	bad := append([]byte(nil), good...)
	bad[20] ^= 0xff // count, inside header CRC
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	if _, err := Open(good[:headerLen+8]); err == nil {
		t.Fatal("truncated region accepted")
	}
}

// TestEstimateBytesMatchesCode pins the no-rerank scoring path's byte reader against
// the build-side estimator. estimateBytes reads a one-bit code straight from the
// region bytes the mmap holds; oneBitCode.estimate reads the same code lifted onto
// the heap. The two must return the identical value, so the zero-copy reader scores
// exactly as the copying reader did and the mmap change is invisible to recall.
func TestEstimateBytesMatchesCode(t *testing.T) {
	const dim = 128
	rng := rand.New(rand.NewSource(7))
	rot := newRotator(dim, 12345)
	for trial := 0; trial < 300; trial++ {
		oRot := rot.rotate(randVec(rng, dim))
		code := encodeOneBit(oRot)
		q := encodeQuery(rot.rotate(randVec(rng, dim)))

		rowBits := make([]byte, 0, len(code.bits)*8)
		for _, word := range code.bits {
			rowBits = codec.AppendUint64(rowBits, word)
		}

		want := code.estimate(q)
		got := estimateBytes(rowBits, code.scalar, code.norm, q)
		if math.Abs(want-got) > 1e-12 {
			t.Fatalf("trial %d: estimateBytes = %v, want %v", trial, got, want)
		}
	}
}

// TestOpenIsZeroCopy is the M17 memory gate. The old reader copied every code,
// scalar, norm, int8 row, and link byte onto the Go heap at Open, so a region of
// size S grew the heap by roughly S. The mmap reader keeps views over the region
// bytes instead, so Open must grow the heap by far less than the region size: only
// the small Region struct and the upper-layer link directory, never a second copy
// of the codes and the int8 rerank that are the bulk of the region. At 100k shards
// this is the difference between the codes being resident and being OS-paged.
func TestOpenIsZeroCopy(t *testing.T) {
	const dim, n = 128, 4000
	corpus := clusteredCorpus(n, dim, 40, 3)
	// The zero-copy property is about the region's bytes, not the graph quality, so a
	// cheap low-degree, low-ef build keeps the test fast under the race detector while
	// the codes and int8 rerank are full size and a copy would still be plain to see.
	b := NewBuilder(dim).WithHNSW(8, 16, 16)
	for _, v := range corpus {
		b.Add(v)
	}
	region := mustBuild(t, b)
	regionSize := len(region)

	// Baseline heap with the region bytes already allocated (in production these are
	// the mmap, not the heap), so the measurement isolates what Open itself adds.
	var m0, m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m0)

	r, err := Open(region)
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	runtime.GC()
	runtime.ReadMemStats(&m1)
	grew := int64(m1.HeapAlloc) - int64(m0.HeapAlloc)

	// The codes and int8 rerank are the bulk of the region; a copying reader would
	// add roughly the whole region again. The view reader must stay well under that.
	// A generous bound of an eighth of the region size proves no full second copy.
	bound := int64(regionSize) / 8
	if grew > bound {
		t.Fatalf("Open grew heap by %d bytes over a %d-byte region (bound %d); the codes look copied, not viewed",
			grew, regionSize, bound)
	}
	t.Logf("region %d bytes, Open grew heap by %d bytes (%.1f%% of region)",
		regionSize, grew, 100*float64(grew)/float64(regionSize))

	// The views must stay valid for the region's lifetime, so a search still works.
	got := r.Search(corpus[0], 10, 128, 100)
	if len(got) == 0 {
		t.Fatal("search over the view reader returned nothing")
	}
	runtime.KeepAlive(region)
}

// nearDupRows builds an int8 corpus of tight near-duplicate clusters, the shape that
// makes the diversity heuristic seal a cluster into an island: every point in a
// cluster is a near copy of its center, so the heuristic keeps almost no neighbors
// for it and the others drop it, stranding the whole cluster from the entry walk.
func nearDupRows(n, dim, clusters int, seed int64) [][]int8 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float64, clusters)
	for c := range centers {
		v := make([]float64, dim)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		centers[c] = v
	}
	rows := make([][]int8, n)
	for i := range rows {
		c := centers[rng.Intn(clusters)]
		r := make([]int8, dim)
		for j := range r {
			q := int(math.Round((c[j] + rng.NormFloat64()*0.05) * 20))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			r[j] = int8(q)
		}
		rows[i] = r
	}
	return rows
}

// TestRepairReconnectsOrphans is the M17 connectivity gate at the graph level. On a
// near-duplicate-heavy corpus the raw build strands whole clusters from the entry, so
// the documents in them are invisible to dense search, a silent recall hole. The test
// first asserts the raw build does strand nodes, so it exercises the repair rather
// than a graph that was already connected, then runs the repair the builder runs and
// requires every node reachable, with no node pushed past the layer-0 degree budget
// the fixed-width serialized format silently truncates at.
func TestRepairReconnectsOrphans(t *testing.T) {
	const n, dim, m, m0 = 2000, 128, 8, 16
	rows := nearDupRows(n, dim, 30, 99)
	// A low ef keeps the serial build cheap under the race detector; a narrow beam
	// also strands more nodes, not fewer, so it exercises the repair harder.
	g := newHNSW(rows, m, m0, 16, 7)
	if pre := g.reachableCount(); pre >= n {
		t.Fatalf("expected the raw build to strand nodes so the repair is exercised, got all %d reachable", pre)
	} else {
		t.Logf("raw build stranded %d of %d nodes from the entry", n-pre, n)
	}
	prev := -1
	for {
		g.repair()
		got := g.reachableCount()
		if got == n {
			break
		}
		if got <= prev {
			t.Fatalf("repair stalled at %d of %d reachable", got, n)
		}
		prev = got
	}
	for node, ns := range g.links {
		if len(ns) > g.m0 {
			t.Fatalf("node %d has %d layer-0 neighbors, over the m0 budget %d", node, len(ns), g.m0)
		}
	}
}

// TestBuildRegionFullyReachable is the same gate end to end through the serialized
// region: a near-duplicate corpus built and reopened must expose a layer-0 graph in
// which every document is reachable from the stored entry point, so the connectivity
// the repair guarantees survives serialization and a search can reach any document.
func TestBuildRegionFullyReachable(t *testing.T) {
	const n, dim = 1500, 96
	rng := rand.New(rand.NewSource(5))
	centers := make([][]float32, 25)
	for c := range centers {
		centers[c] = normalize(randVec(rng, dim))
	}
	// A cheap low-degree, low-ef build keeps the test fast under the race detector; the
	// connectivity invariant the test checks holds regardless of graph quality.
	b := NewBuilder(dim).WithHNSW(8, 16, 16)
	for i := 0; i < n; i++ {
		c := centers[rng.Intn(len(centers))]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + 0.02*float32(rng.NormFloat64())
		}
		b.Add(normalize(v))
	}
	r, err := Open(mustBuild(t, b))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if r.Count() != n {
		t.Fatalf("count %d, want %d", r.Count(), n)
	}
	seen := make([]bool, n)
	entry := int32(r.h.entryPoint)
	seen[entry] = true
	reached := 1
	stack := []int32{entry}
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, nb := range r.links.neighbors0(cur) {
			if int(nb) < n && !seen[nb] {
				seen[nb] = true
				reached++
				stack = append(stack, nb)
			}
		}
	}
	if reached != n {
		t.Fatalf("only %d of %d documents reachable from the entry in the built region", reached, n)
	}
}
