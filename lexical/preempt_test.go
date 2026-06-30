package lexical

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// tripContext is a context whose Err trips to cancelled after a fixed number of Err
// calls, so a test can pin exactly where in a traversal the deadline appears to pass. Its
// Done channel never fires (it embeds Background), and the WAND walk polls Err and never
// selects on Done, so the walk abandons precisely at the stride poll the call count lands
// on. The walk polls once per stride+1 pivot steps, so tripAt=n abandons after n polls,
// that is after n strides of pivots.
type tripContext struct {
	context.Context
	calls  *int
	tripAt int
}

func (c tripContext) Err() error {
	*c.calls++
	if *c.calls > c.tripAt {
		return context.Canceled
	}
	return c.Context.Err()
}

// commonCorpus builds n single-term documents all holding the same body term, so a query
// for that term matches every document and the WAND traversal scores one document per
// pivot step with no pruning (the top-k never fills when k covers the corpus). That makes
// the pivot-step count equal the document count, so a stride-placed deadline abandons the
// walk after an exact, predictable number of documents.
func commonCorpus(n int) []map[Field]string {
	docs := make([]map[Field]string, n)
	for i := range docs {
		docs[i] = map[Field]string{FieldBody: "common"}
	}
	return docs
}

// TestSearchTermsCtxLiveMatchesPlain is the no-regression check: under a context with no
// deadline the context-aware path returns exactly what the un-budgeted path returns and
// reports no cancellation, for the plain and the pushed-down-idf variants alike.
func TestSearchTermsCtxLiveMatchesPlain(t *testing.T) {
	docs := genCorpus(1, 3000, 250)
	r := buildRegion(t, docs)
	queries := genQueries(2, 200, 250, 4)
	idf := map[string]float64{"term0001": 3.5, "term0100": 1.2}
	for _, k := range []int{1, 10, 100} {
		for _, q := range queries {
			terms := Analyze(q)
			want, err := r.SearchTerms(terms, k)
			if err != nil {
				t.Fatalf("plain search %q: %v", q, err)
			}
			got, err := r.SearchTermsCtx(context.Background(), terms, k)
			if err != nil {
				t.Fatalf("ctx search %q: %v", q, err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("k=%d q=%q ctx result diverged\n got=%v\nwant=%v", k, q, got, want)
			}
			wantIDF, _ := r.SearchTermsWithIDF(terms, k, idf)
			gotIDF, err := r.SearchTermsWithIDFCtx(context.Background(), terms, k, idf)
			if err != nil {
				t.Fatalf("ctx idf search %q: %v", q, err)
			}
			if !reflect.DeepEqual(gotIDF, wantIDF) {
				t.Fatalf("k=%d q=%q ctx idf result diverged", k, q)
			}
		}
	}
}

// TestSearchTermsCtxCancelledAbandons checks that a context already cancelled when the
// traversal starts makes the first stride poll trip, so the WAND walk abandons at its
// first pivot step and reports context.Canceled with nothing scored.
func TestSearchTermsCtxCancelledAbandons(t *testing.T) {
	r := buildRegion(t, commonCorpus(2000))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err := r.SearchTermsCtx(ctx, []string{"common"}, 2000)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled search error = %v, want context.Canceled", err)
	}
	if len(got) != 0 {
		t.Fatalf("cancelled-before-walk search scored %d documents, want 0", len(got))
	}
}

// TestSearchTermsCtxPreemptsMidWalk is the headline: a deadline that passes after the
// first stride poll abandons the WAND traversal mid-list rather than at its start or only
// at its end. With one stride of pivots scored before the second poll trips, the partial
// holds exactly one stride of documents, proving the poll lives inside the traversal loop
// and is strided, not a single check at the boundary.
func TestSearchTermsCtxPreemptsMidWalk(t *testing.T) {
	const n = wandPreemptStride*2 + 100 // more than two strides, so a mid-walk trip is unambiguous
	r := buildRegion(t, commonCorpus(n))
	calls := 0
	ctx := tripContext{Context: context.Background(), calls: &calls, tripAt: 1}
	got, err := r.SearchTermsCtx(ctx, []string{"common"}, n)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-walk search error = %v, want context.Canceled", err)
	}
	if want := wandPreemptStride + 1; len(got) != want {
		t.Fatalf("mid-walk traversal scored %d documents, want %d (one stride)", len(got), want)
	}
}
