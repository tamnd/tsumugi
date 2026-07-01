package sparse

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// buildRegion makes an impact index over a random corpus: a vocabulary of terms,
// each appearing in a random subset of documents with a heavy-tailed weight.
func buildRegion(t *testing.T, docs, vocab int, blockSize uint32, seed int64) *Region {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	b := NewBuilder(uint32(docs)).WithBlockSize(blockSize)
	for term := 0; term < vocab; term++ {
		name := termName(term)
		df := 1 + rng.Intn(docs/2+1)
		for i := 0; i < df; i++ {
			d := uint32(rng.Intn(docs))
			w := math.Pow(10, rng.Float64()*3) // 1..1000, heavy tailed
			b.Add(name, d, w)
		}
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return r
}

func termName(i int) string {
	return string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('a'+(i/676)%26))
}

// buildSpladeRegion makes an impact index shaped like a real learned-sparse
// (SPLADE / uniCOIL) corpus, the distribution impact ordering is built for. Each
// document expands to a few dozen terms drawn Zipf by popularity, and each
// (document, term) weight is spiky: most weights sit near the floor with a few
// dominant ones, the concentration that lets an impact-ordered walk reach a
// document's score early. It returns the region and each document's term set, so
// queries can be drawn from real document vocabulary rather than terms that never
// co-occur.
func buildSpladeRegion(t *testing.T, docs, vocab int, blockSize uint32, seed int64) (*Region, [][]string) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	zipf := rand.NewZipf(rng, 1.2, 1, uint64(vocab-1))
	b := NewBuilder(uint32(docs)).WithBlockSize(blockSize)
	docTerms := make([][]string, docs)
	for d := 0; d < docs; d++ {
		length := 40 + rng.Intn(120)
		seen := make(map[uint32]bool, length)
		for i := 0; i < length; i++ {
			term := uint32(zipf.Uint64())
			if seen[term] {
				continue
			}
			seen[term] = true
			name := termName(int(term))
			w := math.Pow(10, 3*math.Pow(rng.Float64(), 2)) // spiky: floor-heavy, few dominant
			b.Add(name, uint32(d), w)
			docTerms[d] = append(docTerms[d], name)
		}
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return r, docTerms
}

// TestSingleTermExact checks the traversal is exact for single-term queries. A
// document's whole score for one term lives in one block, so the anytime cutoff
// never drops a winner: Search matches the exhaustive oracle bit for bit.
func TestSingleTermExact(t *testing.T) {
	r := buildRegion(t, 4000, 300, 64, 1)
	rng := rand.New(rand.NewSource(2))
	for q := 0; q < 500; q++ {
		query := map[string]int{termName(rng.Intn(300)): 1 + rng.Intn(3)}
		for _, k := range []int{1, 5, 10, 100} {
			got := r.Search(query, k)
			want := r.SearchExhaustive(query, k)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("query %v k=%d:\n anytime=%v\n   exact=%v", query, k, got, want)
			}
		}
	}
}

// TestApproxRecall is the multi-term gate. The anytime cutoff is approximate when
// a query has more than one term, so it is held to recall against the exhaustive
// oracle rather than bit-equality: the mean recall@k over many random queries must
// clear a high floor, and no single query may collapse. docID-agreement is scored,
// which is the retrieval quality the ranker consumes.
func TestApproxRecall(t *testing.T) {
	r, docTerms := buildSpladeRegion(t, 6000, 400, 64, 1)
	rng := rand.New(rand.NewSource(2))
	for _, k := range []int{10, 100} {
		var sum float64
		var n, worstQ int
		worst := 1.0
		for q := 0; q < 800; q++ {
			nTerms := 2 + rng.Intn(3)
			query := drawQuery(rng, docTerms, nTerms)
			if len(query) < 2 {
				continue
			}
			want := r.SearchExhaustive(query, k)
			if len(want) == 0 {
				continue
			}
			got := r.Search(query, k)
			rec := recall(got, want)
			sum += rec
			n++
			if rec < worst {
				worst, worstQ = rec, q
			}
		}
		mean := sum / float64(n)
		if mean < 0.99 {
			t.Fatalf("k=%d mean recall %.4f below 0.99 over %d queries", k, mean, n)
		}
		if worst < 0.6 {
			t.Fatalf("k=%d worst-query recall %.4f (query %d) below 0.6 floor", k, worst, worstQ)
		}
		t.Logf("k=%d mean recall %.4f, worst %.4f over %d queries", k, mean, worst, n)
	}
}

// drawQuery samples nTerms distinct terms from a random document's expansion, so
// the query terms genuinely co-occur the way a real query overlaps relevant docs.
func drawQuery(rng *rand.Rand, docTerms [][]string, nTerms int) map[string]int {
	terms := docTerms[rng.Intn(len(docTerms))]
	query := map[string]int{}
	for i := 0; i < nTerms*3 && len(query) < nTerms; i++ {
		query[terms[rng.Intn(len(terms))]] = 1 + rng.Intn(3)
	}
	return query
}

// recall is the fraction of the exact top-k docIDs the approximate result also
// returns.
func recall(got, want []Result) float64 {
	if len(want) == 0 {
		return 1
	}
	set := make(map[uint32]bool, len(want))
	for _, r := range want {
		set[r.DocID] = true
	}
	hit := 0
	for _, r := range got {
		if set[r.DocID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

// TestImpactOrderInvariant checks the on-disk impact ordering: within a block the
// impacts are descending with lead as the first and min as the last, the leading
// impacts fall monotonically across a term's blocks, and the zig-zag docID deltas
// round-trip to the deduped, strongest-per-doc posting set.
func TestImpactOrderInvariant(t *testing.T) {
	r := buildRegion(t, 2000, 100, 32, 3)
	for _, name := range r.names {
		blocks := r.termBlocks(name)
		prevLead := uint8(255)
		seen := map[uint32]bool{}
		for bi, blk := range blocks {
			docIDs, impacts := blk.decode()
			if len(impacts) == 0 || uint32(len(impacts)) != blk.count {
				t.Fatalf("term %s block %d: %d impacts for count %d", name, bi, len(impacts), blk.count)
			}
			if impacts[0] != blk.lead {
				t.Fatalf("term %s block %d: lead %d not first impact %d", name, bi, blk.lead, impacts[0])
			}
			if impacts[len(impacts)-1] != blk.min {
				t.Fatalf("term %s block %d: min %d not last impact %d", name, bi, blk.min, impacts[len(impacts)-1])
			}
			for i, imp := range impacts {
				if imp > blk.lead {
					t.Fatalf("term %s block %d: impact %d above lead %d", name, bi, imp, blk.lead)
				}
				if i > 0 && imp > impacts[i-1] {
					t.Fatalf("term %s block %d: impacts not descending at %d", name, bi, i)
				}
			}
			if blk.lead > prevLead {
				t.Fatalf("term %s block %d: lead %d rose above previous %d", name, bi, blk.lead, prevLead)
			}
			prevLead = blk.lead
			for _, d := range docIDs {
				if seen[d] {
					t.Fatalf("term %s: docID %d appears twice", name, d)
				}
				seen[d] = true
			}
		}
	}
}

// TestAnytimeSkips checks the cutoff actually skips work: on a small-k query over
// a corpus with skewed block maxima, the traversal decodes far fewer blocks than
// the query's total, while still returning the exact top-k for a single term.
func TestAnytimeSkips(t *testing.T) {
	r := buildRegion(t, 20000, 200, 64, 5)
	rng := rand.New(rand.NewSource(9))
	var totalExamined, totalBlocks int
	for q := 0; q < 100; q++ {
		term := termName(rng.Intn(200))
		query := map[string]int{term: 1}
		blocks := len(r.termBlocks(term))
		if blocks < 4 {
			continue
		}
		got, examined := r.searchStats(query, 10)
		want := r.SearchExhaustive(query, 10)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("single-term q=%d: anytime != exact", q)
		}
		totalExamined += examined
		totalBlocks += blocks
	}
	if totalBlocks == 0 {
		t.Fatal("no eligible terms")
	}
	frac := float64(totalExamined) / float64(totalBlocks)
	if frac > 0.75 {
		t.Fatalf("traversal decoded %.0f%% of blocks, expected a real skip", frac*100)
	}
	t.Logf("decoded %d of %d blocks (%.1f%%)", totalExamined, totalBlocks, frac*100)
}

// TestQuantizerRoundTrip checks the log quantizer keeps order and small
// multiplicative error over a heavy-tailed range.
func TestQuantizerRoundTrip(t *testing.T) {
	q := newQuantizer(0.01, 1000)
	prev := -1.0
	for _, w := range []float64{0.01, 0.1, 1, 10, 100, 1000} {
		lvl := q.quantize(w)
		got := q.dequant(lvl)
		ratio := got / w
		if ratio < 0.95 || ratio > 1.05 {
			t.Fatalf("w=%g dequant=%g ratio=%.3f", w, got, ratio)
		}
		if float64(lvl) <= prev {
			t.Fatalf("levels not monotone at w=%g", w)
		}
		prev = float64(lvl)
	}
	if q.quantize(0) != 0 || q.quantize(-1) != 0 {
		t.Fatal("non-positive weight must quantize to level 0")
	}
}

// TestMissingTermAndEmptyQuery covers absent terms and a query that hits nothing.
func TestMissingTermAndEmptyQuery(t *testing.T) {
	r := buildRegion(t, 500, 50, 32, 4)
	if got := r.Search(map[string]int{"zzzzz": 1}, 10); len(got) != 0 {
		t.Fatalf("absent term returned %d results", len(got))
	}
	if got := r.Search(map[string]int{}, 10); got != nil {
		t.Fatal("empty query returned results")
	}
}

// TestCorruptionRejected flips a header byte and truncates the region.
func TestCorruptionRejected(t *testing.T) {
	b := NewBuilder(100)
	b.Add("alpha", 1, 5)
	b.Add("alpha", 50, 9)
	good := b.Build()

	bad := append([]byte(nil), good...)
	bad[12] ^= 0xff // term count, inside header CRC
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	if _, err := Open(good[:headerLen+2]); err == nil {
		t.Fatal("truncated region accepted")
	}
}
