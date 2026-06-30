package search

import (
	"context"
	"math"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/rank"
)

// fleetTerms samples up to max distinct terms from across the shards' vocabularies, the
// query terms a routing-equivalence test probes. It strides each shard's dictionary so the
// sample spans the vocabulary rather than clustering at its start.
func fleetTerms(shards []*Shard, max int) []string {
	seen := make(map[string]bool)
	var out []string
	for _, s := range shards {
		if s.lex == nil {
			continue
		}
		n := s.lex.TermCount()
		for id := uint32(0); id < n; id += 7 {
			t, ok := s.lex.Term(id)
			if !ok || seen[t] {
				continue
			}
			seen[t] = true
			out = append(out, t)
			if len(out) >= max {
				return out
			}
		}
	}
	return out
}

// routeSet returns the routed shard indices in ascending order, so two routes can be
// compared regardless of the order each produced them.
func routeSet(st *brokerState, terms []string) []int {
	r := st.routeTerms(terms)
	sort.Ints(r)
	return r
}

// intsEqual reports whether two ascending index slices are identical.
func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// buildIncremental opens a broker over the first base shards and publishes the rest one at a
// time, the incremental path. It returns the broker so a test can compare its state to a
// full rebuild over the same shards in the same order.
func buildIncremental(t *testing.T, shards []*Shard, model *rank.Model, base int) *Broker {
	t.Helper()
	b := NewBroker(shards[:base], newTestCascade(model))
	for i := base; i < len(shards); i++ {
		b.Publish(shards[i])
	}
	return b
}

// TestIncrementalRouteMatchesFullRebuild checks that publishing shards one at a time routes
// every query, single-term, multi-term, absent-term, and term-free, to exactly the shards a
// full rebuild over the same set routes it to. The incremental broker appends the published
// shards in the same global order, so the shard indices line up and the routed sets must be
// identical, not merely equivalent up to renumbering.
func TestIncrementalRouteMatchesFullRebuild(t *testing.T) {
	shards, model := buildShardSet(t, 1200, 12)
	full := NewBroker(shards, newTestCascade(model))
	defer func() { _ = full.Close() }()
	inc := buildIncremental(t, shards, model, 3)
	// Do not Close inc: it shares shard pointers with full, and full closes them.

	if inc.NumShards() != full.NumShards() {
		t.Fatalf("incremental NumShards = %d, full = %d", inc.NumShards(), full.NumShards())
	}
	fs, is := full.loadState(), inc.loadState()

	terms := fleetTerms(shards, 200)
	if len(terms) < 20 {
		t.Fatalf("only %d sampled terms, corpus too small to exercise routing", len(terms))
	}
	// Single terms.
	for _, tm := range terms {
		if !intsEqual(routeSet(fs, []string{tm}), routeSet(is, []string{tm})) {
			t.Fatalf("route mismatch for %q", tm)
		}
	}
	// Multi-term queries, including a guaranteed-absent term.
	for i := 0; i+2 < len(terms); i += 5 {
		q := []string{terms[i], terms[i+1], terms[i+2], "zzzqxnotaterm"}
		if !intsEqual(routeSet(fs, q), routeSet(is, q)) {
			t.Fatalf("multi-term route mismatch at %d: %v", i, q)
		}
	}
	// Term-free query routes everywhere in both.
	if !intsEqual(routeSet(fs, nil), routeSet(is, nil)) {
		t.Fatalf("term-free route mismatch")
	}
	// An entirely absent term routes nowhere in both.
	if r := routeSet(is, []string{"zzzqxnotaterm"}); len(r) != 0 {
		t.Fatalf("absent term routed to %v, want nowhere", r)
	}
}

// TestIncrementalSearchMatchesFullRebuild checks the incremental broker returns the same
// ranked top-k as a full rebuild for real queries, the end-to-end proof that the overlay
// routing and the folded statistics produce identical results, not just identical routes.
func TestIncrementalSearchMatchesFullRebuild(t *testing.T) {
	shards, model := buildShardSet(t, 1200, 12)
	full := NewBroker(shards, newTestCascade(model))
	defer func() { _ = full.Close() }()
	inc := buildIncremental(t, shards, model, 4)

	terms := fleetTerms(shards, 60)
	ctx := context.Background()
	for i := 0; i+1 < len(terms); i += 3 {
		q := Query{Terms: []string{terms[i], terms[i+1]}, K: 20}
		fh := full.Search(ctx, q)
		ih := inc.Search(ctx, q)
		if len(fh) != len(ih) {
			t.Fatalf("hit count differs for %v: full %d, inc %d", q.Terms, len(fh), len(ih))
		}
		for j := range fh {
			if fh[j].DocID != ih[j].DocID || math.Abs(float64(fh[j].Score-ih[j].Score)) > 1e-6 {
				t.Fatalf("hit %d differs for %v: full {%d %.6f}, inc {%d %.6f}",
					j, q.Terms, fh[j].DocID, fh[j].Score, ih[j].DocID, ih[j].Score)
			}
		}
	}
}

// TestIncrementalStatsMatchFullRebuild checks the folded fleet statistics match a full
// rescan: a publish adds one shard's raw sums to the running totals and re-divides, so the
// document count is exact and the averages match to floating-point tolerance.
func TestIncrementalStatsMatchFullRebuild(t *testing.T) {
	shards, model := buildShardSet(t, 1000, 10)
	full := NewBroker(shards, newTestCascade(model))
	defer func() { _ = full.Close() }()
	inc := buildIncremental(t, shards, model, 2)

	fg, ig := full.Stats(), inc.Stats()
	if fg.DocCount != ig.DocCount {
		t.Fatalf("DocCount: full %d, inc %d", fg.DocCount, ig.DocCount)
	}
	if math.Abs(fg.AvgDocLen-ig.AvgDocLen) > 1e-6 {
		t.Fatalf("AvgDocLen: full %.9f, inc %.9f", fg.AvgDocLen, ig.AvgDocLen)
	}
	for f := range fg.AvgFieldLen {
		if math.Abs(fg.AvgFieldLen[f]-ig.AvgFieldLen[f]) > 1e-6 {
			t.Fatalf("AvgFieldLen[%d]: full %.9f, inc %.9f", f, fg.AvgFieldLen[f], ig.AvgFieldLen[f])
		}
	}
}

// TestOverlayCompaction checks the overlay folds back into the base once it passes its
// limit: after enough publishes to cross overlayLimit, the state's baseN reaches the whole
// set and the overlay is empty, and below the limit the overlay grows by one per publish.
// Either way routing stays correct, the property the fold must preserve.
func TestOverlayCompaction(t *testing.T) {
	shards, model := buildShardSet(t, 1700, 17)
	base := 1
	b := NewBroker(shards[:base], newTestCascade(model))
	full := NewBroker(shards, newTestCascade(model))
	defer func() { _ = full.Close() }()

	lim := overlayLimit(base)
	// The first lim publishes stay in the overlay; the publish that would make it lim+1
	// folds everything into a fresh base.
	for i := base; i < len(shards); i++ {
		b.Publish(shards[i])
		st := b.loadState()
		published := len(st.shards) - st.baseN
		if len(st.overlay) > lim {
			t.Fatalf("overlay grew to %d past limit %d without folding", len(st.overlay), lim)
		}
		if published > lim {
			t.Fatalf("published-since-base %d exceeded limit %d, fold did not fire", published, lim)
		}
	}
	// After all publishes the routing still matches a full rebuild.
	terms := fleetTerms(shards, 120)
	bs, fs := b.loadState(), full.loadState()
	for _, tm := range terms {
		if !intsEqual(routeSet(bs, []string{tm}), routeSet(fs, []string{tm})) {
			t.Fatalf("post-compaction route mismatch for %q", tm)
		}
	}
	if bs.baseN != len(shards) && len(bs.overlay) == 0 {
		t.Fatalf("expected a fold to leave baseN at the full count or a non-empty overlay, got baseN=%d overlay=%d of %d shards",
			bs.baseN, len(bs.overlay), len(shards))
	}
}

// TestPublishThenRetireRebuildsBase checks a retire after incremental publishes returns to a
// clean full-rebuilt state (empty overlay, baseN at the survivor count) and routes the
// survivors exactly, since retire takes the full-rebuild path that folds the overlay in.
func TestPublishThenRetireRebuildsBase(t *testing.T) {
	shards, model := buildShardSet(t, 800, 8)
	b := NewBroker(shards[:2], newTestCascade(model))
	defer func() { _ = b.Close() }()
	for i := 2; i < len(shards); i++ {
		b.Publish(shards[i])
	}
	victim := shards[5]
	lo, hi := idRange(victim)
	if n := b.Retire(func(s *Shard) bool { return s == victim }); n != 1 {
		t.Fatalf("retire removed %d, want 1", n)
	}
	st := b.loadState()
	if len(st.overlay) != 0 || st.baseN != len(st.shards) {
		t.Fatalf("retire did not rebuild the base: overlay=%d baseN=%d shards=%d", len(st.overlay), st.baseN, len(st.shards))
	}
	hits := b.Search(context.Background(), Query{Text: "the", K: 200})
	if hitsTouchRange(hits, lo, hi) {
		t.Fatalf("retired shard's documents still served")
	}
}

// TestRoutingBytesCountsOverlay checks the reported routing footprint includes the overlay:
// publishing a shard onto a base with room in the overlay grows RoutingBytes (the new shard's
// dictionary is now charged), and the base routing index alone never accounts for the whole
// footprint once the overlay is non-empty. This is the accounting a scale measurement relies
// on to see the overlay's cost between compactions rather than undercounting routing memory.
func TestRoutingBytesCountsOverlay(t *testing.T) {
	shards, model := buildShardSet(t, 1000, 10)
	b := NewBroker(shards[:1], newTestCascade(model))
	defer func() { _ = b.Close() }()

	before := b.RoutingBytes()
	b.Publish(shards[1])
	after := b.RoutingBytes()
	st := b.loadState()
	if len(st.overlay) == 0 {
		t.Fatalf("expected the publish to land in the overlay, got baseN=%d shards=%d", st.baseN, len(st.shards))
	}
	if after <= before {
		t.Fatalf("RoutingBytes did not grow after publish: before %d, after %d", before, after)
	}
	if st.routing.sizeBytes() >= after {
		t.Fatalf("base routing size %d accounts for the whole footprint %d, overlay not charged", st.routing.sizeBytes(), after)
	}
}
