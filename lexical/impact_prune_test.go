package lexical

import (
	"context"
	"testing"
)

// queryInfos resolves a query string to the termInfo slice the traversals take, the same
// resolution SearchImpact does, so the prune tests can call the exhaustive oracle and the
// pruned traversal on identical inputs.
func queryInfos(r *Region, query string) []termInfo {
	return r.termInfos(Analyze(query), nil)
}

func sameCandidates(a, b []Candidate) bool {
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

// TestPrunedImpactMatchesExhaustive is the correctness gate: the pruned traversal must
// return exactly what the exhaustive scorer returns, over many random multi-term queries at
// several k. The small k values are where the early stop bites, so this checks the pruning
// is exact precisely where it discards work.
func TestPrunedImpactMatchesExhaustive(t *testing.T) {
	docs := genCorpus(23, 4000, 200)
	r := buildImpactRegion(t, docs)
	queries := genQueries(29, 200, 200, 4)
	for _, k := range []int{1, 3, 10, 50, 500, 4000} {
		for _, q := range queries {
			infos := queryInfos(r, q)
			want, err := r.exhaustiveImpact(infos, k)
			if err != nil {
				t.Fatalf("exhaustive %q k=%d: %v", q, k, err)
			}
			got, err := r.prunedImpact(infos, k)
			if err != nil {
				t.Fatalf("pruned %q k=%d: %v", q, k, err)
			}
			if !sameCandidates(got, want) {
				t.Fatalf("query %q k=%d: pruned %+v != exhaustive %+v", q, k, got, want)
			}
		}
	}
}

// TestPrunedImpactMatchesNaive checks the pruned traversal against the independent naive
// oracle too, so the gate does not rest on the exhaustive scorer alone.
func TestPrunedImpactMatchesNaive(t *testing.T) {
	docs := genCorpus(23, 4000, 200)
	r := buildImpactRegion(t, docs)
	queries := genQueries(31, 200, 200, 4)
	const k = 50
	for _, q := range queries {
		got, err := r.SearchImpact(q, k)
		if err != nil {
			t.Fatalf("search %q: %v", q, err)
		}
		want := naiveImpactTopK(r, docs, q, k)
		if !sameCandidates(got, want) {
			t.Fatalf("query %q: pruned %+v != naive %+v", q, got, want)
		}
	}
}

// TestPrunedImpactSkips proves the early stop actually skips: on a query whose common terms
// fill the top-k at high impact, the pruned traversal examines fewer postings than the lists
// hold, while the exhaustive scan must touch every one. Without the stop the two counts
// would match, so the strict inequality is the skip.
func TestPrunedImpactSkips(t *testing.T) {
	docs := genCorpus(23, 6000, 300)
	r := buildImpactRegion(t, docs)
	queries := genQueries(37, 400, 200, 4)

	var total, skipped int
	for _, q := range queries {
		infos := queryInfos(r, q)
		if len(infos) == 0 {
			continue
		}
		var listPostings int
		for _, info := range infos {
			listPostings += int(info.entry.docFreq)
		}
		_, examined, err := r.prunedImpactStats(infos, 10)
		if err != nil {
			t.Fatalf("pruned %q: %v", q, err)
		}
		if examined > listPostings {
			t.Fatalf("query %q examined %d > list postings %d", q, examined, listPostings)
		}
		total++
		if examined < listPostings {
			skipped++
		}
	}
	if total == 0 {
		t.Fatal("no queries resolved")
	}
	// At k=10 over a corpus with skewed term frequencies most queries settle their top-k
	// well before the tail, so the majority must skip. A traversal that never stops early
	// would leave this at zero.
	if skipped*2 < total {
		t.Fatalf("only %d of %d queries skipped, expected the majority", skipped, total)
	}
	t.Logf("%d of %d queries skipped part of their lists at k=10", skipped, total)
}

// TestPrunedImpactSingleTerm covers the one-cursor path, where the merge is a plain descent
// down one list and every group is a single posting.
func TestPrunedImpactSingleTerm(t *testing.T) {
	docs := genCorpus(13, 2000, 120)
	r := buildImpactRegion(t, docs)
	term, ok := r.Term(0)
	if !ok {
		t.Fatal("no terms")
	}
	for _, k := range []int{1, 5, 50} {
		infos := queryInfos(r, term)
		got, err := r.prunedImpact(infos, k)
		if err != nil {
			t.Fatalf("pruned single %q k=%d: %v", term, k, err)
		}
		want, err := r.exhaustiveImpact(infos, k)
		if err != nil {
			t.Fatalf("exhaustive single %q k=%d: %v", term, k, err)
		}
		if !sameCandidates(got, want) {
			t.Fatalf("single term %q k=%d: pruned %+v != exhaustive %+v", term, k, got, want)
		}
	}
}

// TestPrunedImpactEmpty checks the degenerate corpus serves nothing through the pruned path.
func TestPrunedImpactEmpty(t *testing.T) {
	b := NewBuilder(DefaultParams())
	r, err := Open(b.BuildImpact(impactFor))
	if err != nil {
		t.Fatalf("open empty impact region: %v", err)
	}
	got, err := r.SearchImpact("anything", 10)
	if err != nil {
		t.Fatalf("search empty: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("empty region returned %d results", len(got))
	}
}

// TestPrunedImpactRejectsBM25Region checks the cross-mode guard holds on the pruned path: a
// docID-ordered region refuses an impact search rather than misreading its postings.
func TestPrunedImpactRejectsBM25Region(t *testing.T) {
	docs := genCorpus(5, 100, 40)
	r := buildRegion(t, docs)
	infos := queryInfos(r, "term0001")
	if _, err := r.prunedImpact(infos, 10); err != errNotImpactRegion {
		t.Fatalf("prunedImpact on bm25 region: got %v, want errNotImpactRegion", err)
	}
}

// TestSearchImpactTermsCtxCancel checks the deadline contract the shard fan-out relies on: a
// search whose context is already cancelled abandons the walk and returns context.Canceled,
// so a preempted shard drops its result rather than serve a half-walked list.
func TestSearchImpactTermsCtxCancel(t *testing.T) {
	docs := genCorpus(13, 3000, 200)
	r := buildImpactRegion(t, docs)
	terms := Analyze("term0001 term0002 term0003")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.SearchImpactTermsCtx(ctx, terms, 10); err != context.Canceled {
		t.Fatalf("cancelled search: got %v, want context.Canceled", err)
	}
	// An un-cancelled context returns the same result the un-budgeted path does.
	got, err := r.SearchImpactTermsCtx(context.Background(), terms, 10)
	if err != nil {
		t.Fatalf("background search: %v", err)
	}
	want, err := r.SearchImpactTerms(terms, 10)
	if err != nil {
		t.Fatalf("plain search: %v", err)
	}
	if !sameCandidates(got, want) {
		t.Fatalf("ctx search %+v != plain search %+v", got, want)
	}
}
