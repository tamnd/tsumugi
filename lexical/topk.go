package lexical

import "sort"

// Candidate is one scored document the traversal produces, in the integer score
// domain.
type Candidate struct {
	DocID uint32
	Score int32
}

// topK is a bounded min-heap of the best k candidates seen so far, keyed by
// score. The threshold is the score of the weakest candidate currently held; it
// only ever rises, which is what makes pruning more aggressive the deeper a
// traversal goes. Ties on score break by docID so the result order is
// deterministic and the oracle comparison is exact.
type topK struct {
	heap      []Candidate
	k         int
	threshold int32
	full      bool
}

func newTopK(k int) *topK {
	return &topK{heap: make([]Candidate, 0, k), k: k}
}

// less orders the min-heap: the root is the weakest candidate, the one a new
// offer must beat. Weakest means lowest score, and on a score tie the higher
// docID is weaker so the lower docID is preferred into the result.
func (t *topK) less(i, j int) bool {
	if t.heap[i].Score != t.heap[j].Score {
		return t.heap[i].Score < t.heap[j].Score
	}
	return t.heap[i].DocID > t.heap[j].DocID
}

func (t *topK) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !t.less(i, parent) {
			break
		}
		t.heap[i], t.heap[parent] = t.heap[parent], t.heap[i]
		i = parent
	}
}

func (t *topK) down(i int) {
	n := len(t.heap)
	for {
		l, r := 2*i+1, 2*i+2
		smallest := i
		if l < n && t.less(l, smallest) {
			smallest = l
		}
		if r < n && t.less(r, smallest) {
			smallest = r
		}
		if smallest == i {
			break
		}
		t.heap[i], t.heap[smallest] = t.heap[smallest], t.heap[i]
		i = smallest
	}
}

// offer admits a candidate if there is room or it beats the threshold.
func (t *topK) offer(c Candidate) {
	if len(t.heap) < t.k {
		t.heap = append(t.heap, c)
		t.up(len(t.heap) - 1)
		if len(t.heap) == t.k {
			t.full = true
			t.threshold = t.heap[0].Score
		}
		return
	}
	// Beat the weakest, with the same tie rule the heap orders by.
	root := t.heap[0]
	if c.Score > root.Score || (c.Score == root.Score && c.DocID < root.DocID) {
		t.heap[0] = c
		t.down(0)
		t.threshold = t.heap[0].Score
	}
}

// results returns the candidates sorted strongest first: descending score, then
// ascending docID. This is the order the cascade and the oracle compare on.
func (t *topK) results() []Candidate {
	out := make([]Candidate, len(t.heap))
	copy(out, t.heap)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DocID < out[j].DocID
	})
	return out
}
