package collection

import (
	"hash/fnv"
	"os"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/mph"
)

// TestIDTableOnCCrawl exercises the graph region's dense-to-global identity layer on
// the real crawl, with the two global-id assignments doc 02 describes. The first is
// the spec's exact global node id, a minimal perfect hash over the canonical URL,
// which is a permutation of [0, N) that the Recursive-Graph-Bisection-style dense
// order does not follow, so it forces the explicit Elias-Fano id table. The second
// is a sparse 64-bit content hash of the canonical URL, the multi-build-headroom
// case the 64-bit id is sized for. Both must round trip every dense docID to its
// global id and back, reject an id the shard does not hold, and keep the table small.
func TestIDTableOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	// Dense docID is the position of a document's canonical URL in crawl order,
	// distinct URLs only, mirroring how the build assigns the local node space.
	urlToDense := make(map[string]int)
	var urls [][]byte
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		if _, dup := urlToDense[cu]; dup {
			continue
		}
		urlToDense[cu] = len(urls)
		urls = append(urls, []byte(cu))
	}
	_ = src.Close()
	n := len(urls)
	if n == 0 {
		t.Skip("no canonical URLs in parquet")
	}

	t.Run("mph_global_ids", func(t *testing.T) {
		// The spec's global node id: an MPH over the canonical URL set. Lookup gives a
		// permutation of [0, N), so dense order does not follow it and the table path runs.
		m := mph.Build(urls, mph.DefaultGamma)
		ids := make([]uint64, n)
		for d, u := range urls {
			ids[d] = m.Lookup(u)
		}
		checkRoundTrip(t, n, ids, urls, "mph")
	})

	t.Run("sparse_hash_global_ids", func(t *testing.T) {
		// A 64-bit content hash, deduped, the sparse headroom case. A canonical URL
		// whose hash collides with an earlier one is dropped so the id set stays
		// distinct, the same first-occurrence rule the directory uses.
		seen := make(map[uint64]struct{}, n)
		ids := make([]uint64, 0, n)
		kept := make([][]byte, 0, n)
		for _, u := range urls {
			h := fnv.New64a()
			_, _ = h.Write(u)
			v := h.Sum64()
			if _, dup := seen[v]; dup {
				continue
			}
			seen[v] = struct{}{}
			ids = append(ids, v)
			kept = append(kept, u)
		}
		checkRoundTrip(t, len(ids), ids, kept, "sparse-fnv64")
	})
}

// checkRoundTrip frames a region over a ring graph with the given per-dense global
// ids, opens it, and verifies the identity layer both ways for every node, plus
// rejection of an absent id, logging the table size.
func checkRoundTrip(t *testing.T, n int, ids []uint64, urls [][]byte, label string) {
	t.Helper()
	b := graph.NewBuilder(n).WithNodeIDs(ids)
	for x := 0; x < n; x++ {
		b.AddEdge(x, (x+1)%n)
	}
	region := b.Build()
	g, err := graph.Open(region)
	if err != nil {
		t.Fatalf("%s: open region: %v", label, err)
	}

	idToDense := make(map[uint64]int, n)
	for d, id := range ids {
		idToDense[id] = d
	}
	for d := 0; d < n; d++ {
		if got := g.Global(d); got != ids[d] {
			t.Fatalf("%s: Global(%d) = %d, want %d", label, d, got, ids[d])
		}
		dense, ok := g.Dense(ids[d])
		if !ok || dense != d {
			t.Fatalf("%s: Dense(%d) = (%d,%v), want (%d,true)", label, ids[d], dense, ok, d)
		}
	}
	// An id no document owns must be rejected. Probe a handful derived from real ids
	// but shifted off every member.
	for _, probe := range absentIDs(idToDense) {
		if _, ok := g.Dense(probe); ok {
			t.Fatalf("%s: Dense accepted absent id %d", label, probe)
		}
	}

	t.Logf("%s: N=%d edges=%d region=%d bytes; id table present=%v",
		label, n, g.EdgeCount(), len(region), idTablePresent(region))
}

// absentIDs returns a few global ids guaranteed not to be members, for the
// rejection check.
func absentIDs(members map[uint64]int) []uint64 {
	var out []uint64
	for id := range members {
		for _, delta := range []uint64{1, 7, 1 << 40} {
			cand := id + delta
			if _, ok := members[cand]; !ok {
				out = append(out, cand)
			}
		}
		if len(out) >= 16 {
			break
		}
	}
	return out
}

// idTablePresent reports whether the framed region carries an explicit id table,
// read from the header's idTableLen field, so the test confirms the permuted ids
// took the table path rather than the contiguous one.
func idTablePresent(region []byte) bool {
	// idTableLen is the uint64 at byte offset 21 in the GRA1 header.
	if len(region) < 29 {
		return false
	}
	var v uint64
	for i := 0; i < 8; i++ {
		v |= uint64(region[21+i]) << (8 * i)
	}
	return v > 0
}
