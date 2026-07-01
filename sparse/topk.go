package sparse

import "sort"

// accTopK keeps the k highest-scoring documents seen so far, keyed by docID so a
// document offered again as its accumulated score grows updates in place rather
// than entering twice. The anytime traversal needs this because a doc's impact
// contributions arrive across several blocks, so the same doc is offered many
// times with a rising partial. It is a min-heap whose root is the weakest kept
// candidate, its score the pruning threshold, exactly like topK but with a docID
// index laid over the heap. "Weaker" means a smaller score, and among equal scores
// a larger docID, matching the final order which prefers the smaller docID.
type accTopK struct {
	k   int
	h   []Result
	pos map[uint32]int // docID -> index in h
}

func newAccTopK(k int) *accTopK {
	return &accTopK{k: k, pos: make(map[uint32]int)}
}

func (t *accTopK) full() bool { return len(t.h) >= t.k }

// threshold is the weakest kept score, valid only once the heap is full.
func (t *accTopK) threshold() int64 { return t.h[0].Score }

func (t *accTopK) weaker(a, b Result) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.DocID > b.DocID
}

func (t *accTopK) swap(i, j int) {
	t.h[i], t.h[j] = t.h[j], t.h[i]
	t.pos[t.h[i].DocID] = i
	t.pos[t.h[j].DocID] = j
}

func (t *accTopK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !t.weaker(t.h[i], t.h[p]) {
			break
		}
		t.swap(i, p)
		i = p
	}
}

func (t *accTopK) down(i int) {
	n := len(t.h)
	for {
		l, rr, s := 2*i+1, 2*i+2, i
		if l < n && t.weaker(t.h[l], t.h[s]) {
			s = l
		}
		if rr < n && t.weaker(t.h[rr], t.h[s]) {
			s = rr
		}
		if s == i {
			break
		}
		t.swap(i, s)
		i = s
	}
}

// offer records score for docID. A doc already in the heap only ever grows its
// score, so it moves away from the weak root and sifts down. A new doc joins while
// the heap has room, or replaces the root when it beats the weakest kept score.
func (t *accTopK) offer(docID uint32, score int64) {
	if i, ok := t.pos[docID]; ok {
		t.h[i].Score = score
		t.down(i)
		return
	}
	if len(t.h) < t.k {
		t.h = append(t.h, Result{DocID: docID, Score: score})
		t.pos[docID] = len(t.h) - 1
		t.up(len(t.h) - 1)
		return
	}
	w := t.h[0]
	if score > w.Score || (score == w.Score && docID < w.DocID) {
		delete(t.pos, w.DocID)
		t.h[0] = Result{DocID: docID, Score: score}
		t.pos[docID] = 0
		t.down(0)
	}
}

func (t *accTopK) sorted() []Result {
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
