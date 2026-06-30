package vector

import (
	"context"
	"math/rand"
	"reflect"
	"testing"
)

// tripContext is a context whose Err trips to cancelled after a fixed number of Err
// calls, so a test can pin where in the beam walk the deadline appears to pass. Its Done
// channel never fires (it embeds Background), and the walk polls Err and never selects on
// Done, so the walk abandons at the stride poll the call count lands on. The beam polls
// once per stride+1 popped candidates, so tripAt=n abandons after n strides of pops.
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

// regionOnly builds an immutable region holding the whole corpus (an empty delta), the
// shard's dense serving shape, and returns it with the corpus for drawing queries.
func regionOnly(t *testing.T, dim, n, clusters int, seed int64) (*Region, [][]float32) {
	t.Helper()
	r, _, corpus := splitCorpus(t, dim, n, n, clusters, seed, nil)
	return r, corpus
}

// TestSearchCtxLiveMatchesPlain is the no-regression check: under a context with no
// deadline the context-aware beam returns exactly what the un-budgeted beam returns and
// reports completed.
func TestSearchCtxLiveMatchesPlain(t *testing.T) {
	const dim, n = 64, 2000
	r, _ := regionOnly(t, dim, n, 30, 7)
	rng := rand.New(rand.NewSource(13))
	for q := 0; q < 60; q++ {
		query := normalize(randVec(rng, dim))
		for _, k := range []int{1, 10, 50} {
			want := r.Search(query, k, DefaultEfSearch, DefaultRerankDepth)
			got, completed := r.SearchCtx(context.Background(), query, k, DefaultEfSearch, DefaultRerankDepth)
			if !completed {
				t.Fatalf("live search reported not completed at q=%d k=%d", q, k)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("q=%d k=%d ctx result diverged\n got=%v\nwant=%v", q, k, got, want)
			}
		}
	}
}

// TestSearchCtxCancelledAbandons checks the first stride poll: a context already
// cancelled when the beam starts trips at the first popped candidate, so the walk
// abandons and reports not completed.
func TestSearchCtxCancelledAbandons(t *testing.T) {
	const dim, n = 64, 2000
	r, corpus := regionOnly(t, dim, n, 30, 8)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, completed := r.SearchCtx(ctx, corpus[0], 10, DefaultEfSearch, DefaultRerankDepth)
	if completed {
		t.Fatalf("cancelled-before-walk search reported completed")
	}
}

// TestSearchCtxPollsAndPreemptsMidWalk proves the poll lives inside the beam loop and is
// strided: with a wide beam over a large graph the walk pops more than one stride of
// candidates, so a context that counts its Err calls is polled more than once over a
// single search (the poll fires at the loop's first step and again a stride later), and a
// deadline that passes after the first poll abandons the walk mid-beam rather than
// running it out.
func TestSearchCtxPollsAndPreemptsMidWalk(t *testing.T) {
	const dim, n = 32, 4000
	r, corpus := regionOnly(t, dim, n, 40, 9)
	// A beam as wide as the graph forces the walk to pop far more than one stride of
	// candidates, so the strided poll must fire more than once.
	const ef = 4000
	calls := 0
	live := tripContext{Context: context.Background(), calls: &calls, tripAt: 1 << 30}
	_, completed := r.SearchCtx(live, corpus[0], 10, ef, ef)
	if !completed {
		t.Fatalf("live wide-beam search reported not completed")
	}
	if calls < 2 {
		t.Fatalf("beam polled the context %d times, want at least 2 (poll is inside the loop and strided)", calls)
	}

	tripped := 0
	ctx := tripContext{Context: context.Background(), calls: &tripped, tripAt: 1}
	_, completed = r.SearchCtx(ctx, corpus[0], 10, ef, ef)
	if completed {
		t.Fatalf("mid-walk beam reported completed")
	}
}
