package collection

import (
	"math"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
)

// neutralDocs returns n documents with the same short prose body, so the quality
// term is constant and the authority and spam tests isolate the signal under test.
func neutralDocs(n int) []convert.Document {
	docs := make([]convert.Document, n)
	for i := range docs {
		docs[i] = convert.Document{Body: "A plain paragraph of ordinary prose content with no links in it at all."}
	}
	return docs
}

// baseSignals returns a graphSignals with a non-degenerate spread in every signal the
// composite reads, so normalization is meaningful rather than all-zero.
func baseSignals(n int) graphSignals {
	s := graphSignals{
		pageRank:       make([]float64, n),
		hostRank:       make([]float64, n),
		domainRank:     make([]float64, n),
		trust:          make([]float64, n),
		spamMass:       make([]float64, n),
		linkingDomains: make([]int, n),
		nearDup:        make([]float64, n),
	}
	for i := 0; i < n; i++ {
		f := float64(i + 1)
		s.pageRank[i] = f * 1e-6
		s.hostRank[i] = f * 1e-5
		s.domainRank[i] = f * 1e-5
		s.trust[i] = f * 1e-6
		s.linkingDomains[i] = i
		s.spamMass[i] = 0.1
		s.nearDup[i] = 0.1
	}
	return s
}

func cloneSignals(s graphSignals) graphSignals {
	c := s
	c.pageRank = append([]float64(nil), s.pageRank...)
	c.hostRank = append([]float64(nil), s.hostRank...)
	c.domainRank = append([]float64(nil), s.domainRank...)
	c.trust = append([]float64(nil), s.trust...)
	c.spamMass = append([]float64(nil), s.spamMass...)
	c.linkingDomains = append([]int(nil), s.linkingDomains...)
	c.nearDup = append([]float64(nil), s.nearDup...)
	return c
}

// TestCompositeStaticRankAuthorityMonotone checks that raising any authority signal
// for one document can only raise its static rank, the monotonicity the early
// termination safety requires.
func TestCompositeStaticRankAuthorityMonotone(t *testing.T) {
	const n, d = 5, 2
	docs := neutralDocs(n)
	base := baseSignals(n)
	r0 := compositeStaticRank(docs, base)

	bump := func(name string, mut func(graphSignals)) {
		s := cloneSignals(base)
		mut(s)
		r1 := compositeStaticRank(docs, s)
		if r1[d] < r0[d]-1e-12 {
			t.Fatalf("raising %s lowered static rank: %g -> %g", name, r0[d], r1[d])
		}
		if r1[d] <= r0[d] {
			t.Fatalf("raising %s did not raise static rank: %g -> %g", name, r0[d], r1[d])
		}
	}
	bump("pagerank", func(s graphSignals) { s.pageRank[d] *= 100 })
	bump("hostrank", func(s graphSignals) { s.hostRank[d] *= 100 })
	bump("domainrank", func(s graphSignals) { s.domainRank[d] *= 100 })
	bump("trust", func(s graphSignals) { s.trust[d] *= 100 })
	bump("linking domains", func(s graphSignals) { s.linkingDomains[d] += 50 })
}

// TestCompositeStaticRankSpamMonotone checks that raising a spam signal can only lower
// the static rank.
func TestCompositeStaticRankSpamMonotone(t *testing.T) {
	const n, d = 5, 2
	docs := neutralDocs(n)
	base := baseSignals(n)
	r0 := compositeStaticRank(docs, base)

	drop := func(name string, mut func(graphSignals)) {
		s := cloneSignals(base)
		mut(s)
		r1 := compositeStaticRank(docs, s)
		if r1[d] >= r0[d] {
			t.Fatalf("raising %s did not lower static rank: %g -> %g", name, r0[d], r1[d])
		}
	}
	drop("spam mass", func(s graphSignals) { s.spamMass[d] = 0.9 })
	drop("near-dup penalty", func(s graphSignals) { s.nearDup[d] = 0.9 })
}

// TestCompositeStaticRankQualityMonotone checks the content-quality term flows through:
// a page that is mostly link chrome (low quality) ranks below an otherwise identical
// prose page.
func TestCompositeStaticRankQualityMonotone(t *testing.T) {
	const n = 4
	base := baseSignals(n)
	prose := neutralDocs(n)
	nav := neutralDocs(n)
	// Doc 1 becomes a nav-heavy page; everything else identical.
	nav[1] = convert.Document{Body: strings.Repeat("[Link](https://x.test/a) ", 20)}
	rp := compositeStaticRank(prose, base)
	rn := compositeStaticRank(nav, base)
	if rn[1] >= rp[1] {
		t.Fatalf("nav-heavy page %g not below prose page %g for equal authority", rn[1], rp[1])
	}
}

// TestCompositeStaticRankFiniteAndVaries checks the blend produces finite values that
// are not all identical on a spread of inputs.
func TestCompositeStaticRankFiniteAndVaries(t *testing.T) {
	const n = 8
	docs := neutralDocs(n)
	r := compositeStaticRank(docs, baseSignals(n))
	if len(r) != n {
		t.Fatalf("length %d want %d", len(r), n)
	}
	first := r[0]
	var varied bool
	for _, v := range r {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite static rank %g", v)
		}
		if v != first {
			varied = true
		}
	}
	if !varied {
		t.Fatal("static rank is constant across a spread of inputs")
	}
}

// TestNormalize01 checks the helper maps to [0,1] and collapses a degenerate all-equal
// slice to zero rather than dividing by zero.
func TestNormalize01(t *testing.T) {
	got := normalize01([]float64{2, 4, 6})
	want := []float64{0, 0.5, 1}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("normalize01[%d]=%g want %g", i, got[i], want[i])
		}
	}
	flat := normalize01([]float64{3, 3, 3})
	for i, v := range flat {
		if v != 0 {
			t.Fatalf("degenerate normalize01[%d]=%g want 0", i, v)
		}
	}
	if len(normalize01(nil)) != 0 {
		t.Fatal("normalize01(nil) should be empty")
	}
}
