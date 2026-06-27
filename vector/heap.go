package vector

// minHeap and maxHeap are tiny binary heaps of (node, distance) pairs for the
// beam search: the candidate frontier is a min-heap so the closest unexpanded
// node pops first, and the result set is a max-heap so the farthest kept node is
// at the root and is dropped when a closer one arrives. They are hand-rolled
// rather than container/heap to avoid the interface boxing on this hot path.
type minHeap []cand

func (h minHeap) Len() int { return len(h) }

func (h *minHeap) pushItem(c cand) {
	*h = append(*h, c)
	h.up(len(*h) - 1)
}

func (h *minHeap) popMin() cand {
	old := *h
	n := len(old)
	top := old[0]
	old[0] = old[n-1]
	*h = old[:n-1]
	if len(*h) > 0 {
		h.down(0)
	}
	return top
}

func (h minHeap) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if h[i].d >= h[p].d {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
}

func (h minHeap) down(i int) {
	n := len(h)
	for {
		l, r, s := 2*i+1, 2*i+2, i
		if l < n && h[l].d < h[s].d {
			s = l
		}
		if r < n && h[r].d < h[s].d {
			s = r
		}
		if s == i {
			break
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
}

type maxHeap []cand

func (h maxHeap) Len() int { return len(h) }

func (h *maxHeap) pushItem(c cand) {
	*h = append(*h, c)
	h.up(len(*h) - 1)
}

func (h *maxHeap) popMax() cand {
	old := *h
	n := len(old)
	top := old[0]
	old[0] = old[n-1]
	*h = old[:n-1]
	if len(*h) > 0 {
		h.down(0)
	}
	return top
}

func (h maxHeap) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if h[i].d <= h[p].d {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
}

func (h maxHeap) down(i int) {
	n := len(h)
	for {
		l, r, s := 2*i+1, 2*i+2, i
		if l < n && h[l].d > h[s].d {
			s = l
		}
		if r < n && h[r].d > h[s].d {
			s = r
		}
		if s == i {
			break
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
}
