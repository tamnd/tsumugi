package vector

import (
	"math/bits"
	"math/rand"
	"testing"

	"github.com/tamnd/tsumugi/codec"
)

// TestHammingKernelRoundTrip checks the three pieces of the symmetric mode-1 kernel
// against a direct sign-by-sign count: encodeQueryBits packs the same signs encodeOneBit
// does, hammingWords counts disagreeing signs as a word-wise XOR popcount, and
// hammingBytes reads a packed code straight from a region row and returns the same count.
func TestHammingKernelRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const dim = 192 // a multiple of 64 so the packing fills whole words
	for trial := 0; trial < 200; trial++ {
		a := randVec(rng, dim)
		b := randVec(rng, dim)
		// Direct count of dimensions where the two vectors disagree in sign, the
		// definition the popcount kernel must match.
		var want int
		for i := 0; i < dim; i++ {
			if (a[i] >= 0) != (b[i] >= 0) {
				want++
			}
		}

		qa := encodeQueryBits(a)
		qb := encodeQueryBits(b)
		if got := hammingWords(qa.bits, qb.bits); got != want {
			t.Fatalf("hammingWords = %d, want %d", got, want)
		}

		// hammingBytes reads one side from a little-endian byte block, the layout the
		// codes part of a region uses, and must agree with hammingWords over the words.
		rowBits := make([]byte, 0, len(qa.bits)*8)
		for _, word := range qa.bits {
			rowBits = codec.AppendUint64(rowBits, word)
		}
		if got := hammingBytes(rowBits, qb.bits); got != want {
			t.Fatalf("hammingBytes = %d, want %d", got, want)
		}
	}
}

// TestHammingSelfIsZero is the degenerate check: a code's Hamming distance to itself is
// zero, so the walk never strays from an exact match.
func TestHammingSelfIsZero(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	q := encodeQueryBits(randVec(rng, 128))
	if d := hammingWords(q.bits, q.bits); d != 0 {
		t.Fatalf("self distance = %d, want 0", d)
	}
	var pop int
	for _, w := range q.bits {
		pop += bits.OnesCount64(w)
	}
	_ = pop
}

// TestSymmetricWalkRecall measures the spec mode-1 design against the default int8-dot
// walk on one clustered corpus, both with the int8 rerank, so the only difference is the
// navigation metric. It records the recall of each so the default is chosen by data, not
// assertion: the Hamming walk is much cheaper but the one-bit code is a coarser compass
// than the int8 dot, so it is expected to recall lower. The gate is a floor the symmetric
// walk must clear to be a usable trade (it still finds most neighbors), not parity with
// the int8 walk; the measured gap is logged for the spec amendment.
func TestSymmetricWalkRecall(t *testing.T) {
	const dim, n = 64, 1400
	corpus := clusteredCorpus(n, dim, 20, 13)

	build := func(symmetric bool) *Region {
		b := NewBuilder(dim).WithSymmetricWalk(symmetric)
		for _, v := range corpus {
			b.Add(v)
		}
		r, err := Open(mustBuild(t, b))
		if err != nil {
			t.Fatalf("open (symmetric=%v): %v", symmetric, err)
		}
		return r
	}
	base := build(false)
	sym := build(true)

	if base.symmetric {
		t.Fatal("default region should not carry the symmetric flag")
	}
	if !sym.symmetric {
		t.Fatal("symmetric region should carry the symmetric flag")
	}
	// The symmetric region keeps the int8 rerank for scoring; only its walk changed.
	if !sym.hasRerank {
		t.Fatal("symmetric region should still carry the int8 rerank copy")
	}

	rng := rand.New(rand.NewSource(101))
	const queries = 60
	var baseSum, symSum float64
	for q := 0; q < queries; q++ {
		query := normalize(randVec(rng, dim))
		want := trueTopK(corpus, query, 10)
		baseSum += recallAt(base.Search(query, 10, 128, 100), want)
		symSum += recallAt(sym.Search(query, 10, 128, 100), want)
	}
	baseR := baseSum / queries
	symR := symSum / queries
	// The walk's own loss (graph-vs-brute, the Hamming walk against a full scan that shares
	// its rerank) is measured on real data in collection/symmetric_walk_ccrawl_test.go; here
	// the end-to-end recall against the true neighbors is enough to pin the default choice.
	t.Logf("walk recall@10 vs true: int8-dot=%.3f hamming=%.3f", baseR, symR)

	// This is the spec-amendment measurement, the reconciliation of doc 05's mode-1 walk
	// against what the corpus shows. The one-bit code is a coarser compass than the int8
	// dot: it loses about as much to the walk (graph-vs-brute well under one) as to nothing
	// else, so the int8-dot walk recalls strictly higher and is the default. The symmetric
	// walk is shipped and measured, not the default. The two checks below pin both halves
	// of that finding so a regression in either direction is caught.
	if baseR <= symR {
		t.Errorf("int8-dot walk recall %.3f should exceed hamming walk %.3f (the reason int8 is default)", baseR, symR)
	}
	// The Hamming walk still finds a real fraction of the neighbors, so it is a coarse but
	// functional compass for the latency-over-recall trade, not a broken metric. The floor
	// is well below the int8 walk on purpose; it only guards against the popcount path
	// regressing into noise.
	if symR < 0.45 {
		t.Errorf("symmetric walk recall@10 = %.3f, want >= 0.45 (functional floor)", symR)
	}
}
