package collection

import (
	"math/bits"

	"github.com/tamnd/tsumugi/codec"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/lexical"
)

// This file detects near-duplicate documents with SimHash (Charikar's
// locality-sensitive hash) and turns each cluster into a per-document demotion
// column, the spec's content-quality near-dup signal (doc 07 "SimHash
// near-duplicate clustering and penalty"). Scraper mirrors, self-mirrored hosts,
// and thin-content template farms all surface as clusters of near-identical
// bodies; the copy is demoted, the representative is not, and the demotion is a
// ranking feature, not a hard drop, because the copy is sometimes the version a
// user wants.

// nearDupK is the Hamming radius two fingerprints may differ by and still count as
// near-duplicates: the spec's default of 3 bits over the 64-bit SimHash, which it
// puts at roughly 95% content similarity.
const nearDupK = 3

// minSimhashTokens is the floor of content tokens a document needs before it gets a
// fingerprint. Below it the SimHash is unstable (a handful of tokens makes most bit
// columns tie at zero), and an empty or near-empty body would otherwise hash to the
// same all-zero fingerprint as every other empty body and cluster them as mutual
// duplicates. Short pages simply do not participate in near-dup detection.
const minSimhashTokens = 8

// simhash computes Charikar's SimHash over a document body and reports whether the
// body had enough content to fingerprint. Each content token votes its 64-bit hash's
// bits up or down with unit weight, and the sign of each column's tally sets that
// fingerprint bit, so two bodies sharing most tokens differ in few bits. Tokenizing
// through lexical.Analyze means the fingerprint sees the same normalized terms the
// index does, so two pages with the same words but different markup hash alike.
func simhash(body string) (uint64, bool) {
	toks := lexical.Analyze(body)
	if len(toks) < minSimhashTokens {
		return 0, false
	}
	var v [64]int
	for _, t := range toks {
		h := codec.XXHash64([]byte(t))
		for b := 0; b < 64; b++ {
			if h&(uint64(1)<<uint(b)) != 0 {
				v[b]++
			} else {
				v[b]--
			}
		}
	}
	var fp uint64
	for b := 0; b < 64; b++ {
		if v[b] > 0 {
			fp |= uint64(1) << uint(b)
		}
	}
	return fp, true
}

// ufFind returns the root of x with path compression.
func ufFind(parent []int, x int) int {
	for parent[x] != x {
		parent[x] = parent[parent[x]]
		x = parent[x]
	}
	return x
}

// ufUnion merges the sets of a and b.
func ufUnion(parent []int, a, b int) {
	ra, rb := ufFind(parent, a), ufFind(parent, b)
	if ra != rb {
		parent[ra] = rb
	}
}

// nearDupPenalties fingerprints every document, clusters the near-duplicates by
// Hamming distance, picks a representative per cluster, and returns the per-document
// near-dup penalty the build bakes into FeatNearDup. The representative scores zero;
// a copy scores higher the larger its cluster and the further its PageRank falls
// below the representative's, so a low-rank page in a big template farm is demoted
// hard while an authoritative mirror is demoted mildly.
//
// Clustering uses the pigeonhole form of banded blocking: two fingerprints within k
// bits must agree exactly on at least one of k+1 equal 16-bit blocks (any larger
// disagreement would need more than k differing bits), so grouping the fingerprints
// by each block's value surfaces every near-duplicate pair as a within-block pair
// without the all-pairs comparison. Each candidate pair is then confirmed with an
// exact Hamming check before it is unioned, so the blocking only prunes, it never
// admits a false duplicate.
func nearDupPenalties(docs []convert.Document, pageRank []float64) []float64 {
	n := len(docs)
	pen := make([]float64, n)
	if n == 0 {
		return pen
	}

	fps := make([]uint64, n)
	has := make([]bool, n)
	for i := range docs {
		if fp, ok := simhash(docs[i].Body); ok {
			fps[i] = fp
			has[i] = true
		}
	}

	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}

	// k+1 blocks of 16 bits (64 = 4 * 16). For each block, group documents that
	// share that block's value, then Hamming-check the within-group pairs.
	for blk := 0; blk < nearDupK+1; blk++ {
		shift := uint(16 * blk)
		buckets := map[uint16][]int{}
		for i := 0; i < n; i++ {
			if !has[i] {
				continue
			}
			buckets[uint16(fps[i]>>shift)] = append(buckets[uint16(fps[i]>>shift)], i)
		}
		for _, members := range buckets {
			for a := 0; a < len(members); a++ {
				for b := a + 1; b < len(members); b++ {
					i, j := members[a], members[b]
					if bits.OnesCount64(fps[i]^fps[j]) <= nearDupK {
						ufUnion(parent, i, j)
					}
				}
			}
		}
	}

	// Gather clusters by root, then demote every non-representative member.
	clusters := map[int][]int{}
	for i := 0; i < n; i++ {
		if !has[i] {
			continue
		}
		r := ufFind(parent, i)
		clusters[r] = append(clusters[r], i)
	}
	for _, members := range clusters {
		if len(members) < 2 {
			continue
		}
		rep := members[0]
		for _, m := range members[1:] {
			if betterRepresentative(m, rep, docs, pageRank) {
				rep = m
			}
		}
		sizeFactor := 1 - 1/float64(len(members))
		repRank := rankOf(pageRank, rep)
		for _, m := range members {
			if m == rep {
				continue
			}
			gap := 0.0
			if repRank > 0 {
				g := 1 - rankOf(pageRank, m)/repRank
				if g < 0 {
					g = 0
				} else if g > 1 {
					g = 1
				}
				gap = g
			}
			// Half the penalty is the cluster-size weight a copy carries even at no
			// rank gap (the spec's mild penalty for an authoritative mirror); the
			// other half scales with how far the copy sits below the representative.
			pen[m] = sizeFactor * (0.5 + 0.5*gap)
		}
	}
	return pen
}

// rankOf reads a document's PageRank, treating a missing or short rank vector (a
// crawl with no resolved links) as zero so the penalty falls back to the size-only
// term rather than indexing out of range.
func rankOf(pageRank []float64, i int) float64 {
	if i < len(pageRank) {
		return pageRank[i]
	}
	return 0
}

// betterRepresentative reports whether candidate c should replace the current
// representative r of a near-duplicate cluster. The canonical page is the one with
// the highest PageRank; ties break to the earliest crawl, then the shortest URL,
// then the lowest document id, the usual canonicalization preferences so the choice
// is deterministic.
func betterRepresentative(c, r int, docs []convert.Document, pageRank []float64) bool {
	rc, rr := rankOf(pageRank, c), rankOf(pageRank, r)
	if rc != rr {
		return rc > rr
	}
	dc, dr := docs[c].CrawlDate, docs[r].CrawlDate
	if dc != dr {
		// Empty crawl dates sort last so a dated page wins over an undated one.
		if dc == "" {
			return false
		}
		if dr == "" {
			return true
		}
		return dc < dr
	}
	if len(docs[c].URL) != len(docs[r].URL) {
		return len(docs[c].URL) < len(docs[r].URL)
	}
	return c < r
}
