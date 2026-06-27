package sparse

import (
	"container/heap"
	"sort"
)

// topK keeps the k strongest results seen. It is a min-heap whose root is the
// weakest kept candidate, so a new result replaces the root when it beats it and
// the root's score is the pruning threshold. "Weaker" means a smaller score, and
// among equal scores a larger docID, matching the final order which prefers the
// smaller docID on a tie.
type topK struct {
	k int
	h resultHeap
}

func (t *topK) full() bool { return len(t.h) >= t.k }

// threshold is the weakest kept score, valid only once the heap is full.
func (t *topK) threshold() int64 { return t.h[0].Score }

func (t *topK) offer(r Result) {
	if len(t.h) < t.k {
		heap.Push(&t.h, r)
		return
	}
	w := t.h[0]
	if r.Score > w.Score || (r.Score == w.Score && r.DocID < w.DocID) {
		t.h[0] = r
		heap.Fix(&t.h, 0)
	}
}

func (t *topK) sorted() []Result {
	out := make([]Result, len(t.h))
	copy(out, t.h)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DocID < out[j].DocID
	})
	return out
}

type resultHeap []Result

func (h resultHeap) Len() int { return len(h) }

func (h resultHeap) Less(i, j int) bool {
	if h[i].Score != h[j].Score {
		return h[i].Score < h[j].Score
	}
	return h[i].DocID > h[j].DocID
}

func (h resultHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *resultHeap) Push(x any) { *h = append(*h, x.(Result)) }

func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	r := old[n-1]
	*h = old[:n-1]
	return r
}
