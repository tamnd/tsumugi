package sparse

import (
	"context"
	"math/rand"
	"reflect"
	"testing"
)

// tripContext is a context whose Err trips to cancelled after a fixed number of Err
// calls, so a test can pin exactly where in the block walk the deadline appears to pass.
// Its Done channel never fires (it embeds Background), and the walk polls Err and never
// selects on Done, so the walk abandons at the stride poll the call count lands on. The
// walk polls once per stride+1 blocks, so tripAt=n abandons after n strides of blocks.
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

// oneTermRegion builds a region where a single term holds one document per block over n
// blocks, so a query for that term walks exactly n blocks of one document each. With k
// covering the corpus the anytime stop never fires, so the walk scores one document per
// block and a stride-placed deadline abandons after an exact, predictable count.
func oneTermRegion(t *testing.T, n int) *Region {
	t.Helper()
	b := NewBuilder(uint32(n)).WithBlockSize(1)
	for d := 0; d < n; d++ {
		b.Add("aaa", uint32(d), float64(d%100+1))
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return r
}

// TestSearchCtxLiveMatchesPlain is the no-regression check: under a context with no
// deadline the context-aware path returns exactly what the un-budgeted path returns and
// reports completed.
func TestSearchCtxLiveMatchesPlain(t *testing.T) {
	r := buildRegion(t, 4000, 300, 64, 1)
	rng := rand.New(rand.NewSource(2))
	for q := 0; q < 200; q++ {
		nTerms := 1 + rng.Intn(4)
		query := map[string]int{}
		for i := 0; i < nTerms; i++ {
			query[termName(rng.Intn(300))] = 1 + rng.Intn(3)
		}
		for _, k := range []int{1, 10, 100} {
			want := r.Search(query, k)
			got, completed := r.SearchCtx(context.Background(), query, k)
			if !completed {
				t.Fatalf("live search reported not completed, query %v k=%d", query, k)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("query %v k=%d ctx result diverged\n got=%v\nwant=%v", query, k, got, want)
			}
		}
	}
}

// TestSearchCtxCancelledAbandons checks the first stride poll: a context already
// cancelled when the walk starts trips at the first block, so the walk abandons before
// scoring anything and reports not completed.
func TestSearchCtxCancelledAbandons(t *testing.T) {
	r := oneTermRegion(t, 2000)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, completed := r.SearchCtx(ctx, map[string]int{"aaa": 1}, 2000)
	if completed {
		t.Fatalf("cancelled-before-walk search reported completed")
	}
	if len(got) != 0 {
		t.Fatalf("cancelled-before-walk search scored %d documents, want 0", len(got))
	}
}

// TestSearchCtxPreemptsMidWalk is the headline: a deadline that passes after the first
// stride poll abandons the block walk mid-corpus, with exactly one stride of blocks
// scored, proving the poll lives inside the walk loop and is strided.
func TestSearchCtxPreemptsMidWalk(t *testing.T) {
	const n = rangePreemptStride*2 + 100
	r := oneTermRegion(t, n)
	calls := 0
	ctx := tripContext{Context: context.Background(), calls: &calls, tripAt: 1}
	got, completed := r.SearchCtx(ctx, map[string]int{"aaa": 1}, n)
	if completed {
		t.Fatalf("mid-walk search reported completed")
	}
	if want := rangePreemptStride + 1; len(got) != want {
		t.Fatalf("mid-walk walk scored %d documents, want %d (one stride)", len(got), want)
	}
}
