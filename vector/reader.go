package vector

import (
	"math"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// Region is a parsed, read-only dense retrieval region.
type Region struct {
	h         header
	rot       *rotator
	iq        int8Quant
	words     int
	scalars   []float32
	norms     []float32
	bits      []uint64 // flat, words per node
	rerank    []int8   // flat, rdim per node, nil if no rerank
	links     *linksReader
	hasRerank bool
}

// Result is one retrieved document with its dense score, higher meaning nearer.
type Result struct {
	DocID uint32
	Score float64
}

// Open parses a VEC1 region, regenerates the rotation from the stored seed, and
// makes the codes and int8 copy resident for the walk and rerank.
func Open(b []byte) (*Region, error) {
	h, err := decodeHeader(b)
	if err != nil {
		return nil, err
	}
	off := headerLen
	codesEnd := off + int(h.codesLen)
	rerankEnd := codesEnd + int(h.rerankLen)
	linksEnd := rerankEnd + int(h.linksLen)
	if linksEnd > len(b) {
		return nil, ErrCorrupt
	}
	n := int(h.count)
	words := int(h.rdim) / 64
	if int(h.codeStride) != 8+words*8 || (n > 0 && len(b[off:codesEnd]) != n*int(h.codeStride)) {
		return nil, ErrCorrupt
	}

	r := &Region{
		h:         h,
		rot:       newRotator(int(h.dimKept), h.rotationSeed),
		iq:        newInt8Quant(h.i8scale),
		words:     words,
		scalars:   make([]float32, n),
		norms:     make([]float32, n),
		bits:      make([]uint64, n*words),
		hasRerank: h.flags&flagHasRerank != 0,
	}
	if r.rot.rdim != int(h.rdim) {
		return nil, ErrCorrupt
	}
	cp := b[off:codesEnd]
	for i := 0; i < n; i++ {
		row := cp[i*int(h.codeStride):]
		r.scalars[i] = math.Float32frombits(codec.Uint32(row))
		r.norms[i] = math.Float32frombits(codec.Uint32(row[4:]))
		for w := 0; w < words; w++ {
			r.bits[i*words+w] = codec.Uint64(row[8+w*8:])
		}
	}
	if r.hasRerank {
		rr := b[codesEnd:rerankEnd]
		r.rerank = make([]int8, len(rr))
		for i, c := range rr {
			r.rerank[i] = int8(c)
		}
	}
	lr, err := openLinks(b[rerankEnd:linksEnd], n, int(h.m0))
	if err != nil {
		return nil, err
	}
	r.links = lr
	return r, nil
}

// Count returns the number of indexed vectors.
func (r *Region) Count() int { return int(r.h.count) }

func (r *Region) codeBits(node int32) []uint64 {
	return r.bits[int(node)*r.words : (int(node)+1)*r.words]
}

func (r *Region) oneBitCode(node int32) oneBitCode {
	return oneBitCode{scalar: r.scalars[node], norm: r.norms[node], bits: r.codeBits(node)}
}

// Search runs the HNSW walk to gather candidates, reranks the candidate set,
// and returns the top-k by dense score. efSearch widens the layer-0 beam for
// recall; rerankDepth bounds how many candidates get the sharp int8 refine. Pass
// zero for either to take the defaults.
func (r *Region) Search(query []float32, k, efSearch, rerankDepth int) []Result {
	if r.h.count == 0 || k <= 0 {
		return nil
	}
	if efSearch <= 0 {
		efSearch = DefaultEfSearch
	}
	if rerankDepth <= 0 {
		rerankDepth = DefaultRerankDepth
	}
	qRot := r.rot.rotate(query)
	qc := encodeQuery(qRot)

	cands := r.walk(r.navDist(qRot, qc), efSearch)
	return r.rerankAndTop(qRot, qc, cands, k, rerankDepth)
}

// navDist returns the navigation metric for the walk, where smaller is nearer.
// In the two-part mode it negates the int8 dot, the sharp copy the region already
// stores (it tracks the true dot far better than the one-bit code), so a narrow
// beam over the graph lands on the true neighbors. In the no-rerank mode there is
// no int8 copy, so it falls back to the negated asymmetric RaBitQ estimate over
// the one-bit code.
func (r *Region) navDist(qRot []float32, qc queryCode) func(int32) float64 {
	if r.hasRerank {
		qi8 := r.iq.encodeQuery(qRot)
		rdim := int(r.h.rdim)
		return func(node int32) float64 {
			row := r.rerank[int(node)*rdim : (int(node)+1)*rdim]
			return -float64(dotI8(qi8, row))
		}
	}
	return func(node int32) float64 { return -r.oneBitCode(node).estimate(qc) }
}

// walk is the HNSW descent: greedy through the upper layers, then a beam on
// layer 0 under the supplied navigation metric (smaller is nearer). It returns
// candidate node IDs ordered nearest first.
func (r *Region) walk(distQ func(int32) float64, efSearch int) []cand {
	ep := int32(r.h.entryPoint)

	for layer := int(r.h.maxLayer); layer >= 1; layer-- {
		cur := ep
		curD := distQ(cur)
		for {
			improved := false
			for _, nb := range r.links.neighborsUpper(cur, layer) {
				if d := distQ(nb); d < curD {
					cur, curD = nb, d
					improved = true
				}
			}
			if !improved {
				break
			}
		}
		ep = cur
	}

	visited := map[int32]bool{ep: true}
	d0 := distQ(ep)
	candHeap := &minHeap{{ep, d0}}
	resHeap := &maxHeap{{ep, d0}}
	for candHeap.Len() > 0 {
		c := candHeap.popMin()
		if resHeap.Len() >= efSearch && c.d > (*resHeap)[0].d {
			break
		}
		for _, nb := range r.links.neighbors0(c.id) {
			if visited[nb] {
				continue
			}
			visited[nb] = true
			d := distQ(nb)
			if resHeap.Len() < efSearch || d < (*resHeap)[0].d {
				candHeap.pushItem(cand{nb, d})
				resHeap.pushItem(cand{nb, d})
				if resHeap.Len() > efSearch {
					resHeap.popMax()
				}
			}
		}
	}
	out := make([]cand, resHeap.Len())
	copy(out, *resHeap)
	sort.Slice(out, func(i, j int) bool {
		if out[i].d != out[j].d {
			return out[i].d < out[j].d
		}
		return out[i].id < out[j].id
	})
	return out
}

// rerankAndTop scores the candidate set sharply and returns the top-k. In the
// two-part mode it dequantizes the int8 copy and dots against the int8 query; in
// the no-rerank mode it uses the asymmetric RaBitQ estimator off the one-bit code.
func (r *Region) rerankAndTop(qRot []float32, qc queryCode, cands []cand, k, rerankDepth int) []Result {
	if len(cands) > rerankDepth {
		cands = cands[:rerankDepth]
	}
	scored := make([]Result, len(cands))
	if r.hasRerank {
		qi8 := r.iq.encodeQuery(qRot)
		rdim := int(r.h.rdim)
		for i, c := range cands {
			row := r.rerank[int(c.id)*rdim : (int(c.id)+1)*rdim]
			scored[i] = Result{DocID: uint32(c.id), Score: float64(dotI8(qi8, row))}
		}
	} else {
		for i, c := range cands {
			scored[i] = Result{DocID: uint32(c.id), Score: r.oneBitCode(c.id).estimate(qc)}
		}
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].DocID < scored[j].DocID
	})
	if len(scored) > k {
		scored = scored[:k]
	}
	return scored
}

// BruteForce scans every vector with the same scoring the rerank uses, the oracle
// the HNSW recall is measured against. It isolates graph recall from quantization
// error because both paths score identically; only the candidate set differs.
func (r *Region) BruteForce(query []float32, k int) []Result {
	if r.h.count == 0 || k <= 0 {
		return nil
	}
	qRot := r.rot.rotate(query)
	qc := encodeQuery(qRot)
	n := int(r.h.count)
	scored := make([]Result, n)
	if r.hasRerank {
		qi8 := r.iq.encodeQuery(qRot)
		rdim := int(r.h.rdim)
		for i := 0; i < n; i++ {
			row := r.rerank[i*rdim : (i+1)*rdim]
			scored[i] = Result{DocID: uint32(i), Score: float64(dotI8(qi8, row))}
		}
	} else {
		for i := 0; i < n; i++ {
			scored[i] = Result{DocID: uint32(i), Score: r.oneBitCode(int32(i)).estimate(qc)}
		}
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].DocID < scored[j].DocID
	})
	if len(scored) > k {
		scored = scored[:k]
	}
	return scored
}
