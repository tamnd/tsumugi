package vector

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/tamnd/tsumugi/codec"
)

// DefaultEfDelta is the beam width the delta graph is walked with. The delta holds
// only the recent vectors, so a narrower beam than the immutable region's efSearch
// already visits a large fraction of it; the union's recall comes mostly from the
// immutable half, and the delta beam only has to surface the handful of fresh
// neighbors that beat the immutable candidates.
const DefaultEfDelta = 64

// deltaSeedXor perturbs the immutable region's rotation seed into the delta graph's
// layer-assignment seed, so the delta graph's random layers do not lockstep with the
// immutable build's even though the rotation is shared. The rotation itself must stay
// the immutable region's, because the delta encodes its vectors with the same rotator
// the region was built with, or the two halves would live in different rotated spaces
// and their codes would not compare.
const deltaSeedXor int64 = 0x2545f4914f6cdd1d

// ErrDeltaDim is returned by Add when the vector width does not match the region's
// kept dimension, the same width Search and Cosine demand.
var ErrDeltaDim = errors.New("vector: delta vector dimension mismatch")

// Delta is the in-RAM freshness buffer in front of an immutable Region, the
// FreshDiskANN pattern of spec doc 05 (line 953): the immutable mmap index plus a
// small in-RAM delta of recent vectors, with the search unioning the two. New
// documents land here without rebuilding the shard; a Search walks both the
// immutable graph and the delta's small in-RAM HNSW, drops tombstoned candidates,
// and reranks the union. Deletes are a tombstone bitset over the whole id space, so
// a deleted document drops out of results immediately whether it lives in the
// immutable region or the delta. The delta is folded back into the immutable region
// by Compact, which rebuilds the shard deterministically over the live union.
//
// The delta descriptor lives only in RAM. It is not stored in the immutable file, so
// no header flag marks it; a server rebuilds it from its own side log on restart, and
// a Region opened cold simply has no delta until one is attached. That is why
// flagSymmetric (bit 3) is the last header flag and the delta needs none.
//
// The id space is contiguous: immutable documents keep their dense docIDs [0, N), and
// delta documents take [N, N+count) where N is the immutable count, captured as idBase
// at construction. Compact renumbers everything back into [0, live) when it rebuilds.
type Delta struct {
	r              *Region
	idBase         uint32
	efConstruction int

	mu sync.RWMutex
	// Per-delta-document state, all appended in lockstep so a local index addresses the
	// same document across every slice. scalars and norms are float16-rounded to match the
	// precision the immutable region stores, so a delta candidate and an immutable candidate
	// are scored on the same footing. codeBytes is the region rowBits payload (words sign
	// blocks for the one-bit code, packed levels for the multi-bit). codeWords is the one-bit
	// sign words, kept only in the symmetric mode for the graph's node-to-node Hamming. rows
	// is the int8 rerank copy, kept always because the non-symmetric graph is built over the
	// int8 dot exactly as the immutable builder builds it. vecs is the original input, kept so
	// Compact can rebuild the delta documents through the same encode path as the immutable.
	scalars   []float32
	norms     []float32
	codeBytes [][]byte
	codeWords [][]uint64
	rows      [][]int8
	vecs      [][]float32
	// g is the delta's incremental HNSW over the same metric the immutable graph uses.
	g *hnswGraph
	// tomb is a bitset over [0, idBase+count): a set bit marks a deleted document, checked
	// at candidate collection so a delete takes effect on the next Search.
	tomb []uint64
}

// NewDelta attaches a fresh, empty delta buffer to a region. The delta encodes with
// the region's rotator and int8 scale and navigates with the region's distance mode,
// so the two halves of a union search compare like for like.
func (r *Region) NewDelta() *Delta {
	d := &Delta{
		r:              r,
		idBase:         uint32(r.Count()),
		efConstruction: int(r.h.efConstruction),
	}
	if d.efConstruction <= 0 {
		d.efConstruction = DefaultEfConstruction
	}
	// The delta graph navigates with the immutable region's build metric: the symmetric
	// one-bit Hamming popcount in mode-1, the negated int8 dot otherwise (the same metric
	// the immutable builder builds over, even in the multi-bit no-rerank form, which is why
	// the int8 rows are kept regardless of mode).
	var distFn func(a, b int32) float64
	if r.symmetric {
		distFn = func(a, b int32) float64 {
			return float64(hammingWords(d.codeWords[a], d.codeWords[b]))
		}
	} else {
		distFn = func(a, b int32) float64 {
			return -float64(dotI8(d.rows[a], d.rows[b]))
		}
	}
	d.g = newHNSWInc(distFn, int(r.h.m), int(r.h.m0), d.efConstruction, r.h.rotationSeed^deltaSeedXor)
	d.ensureTomb(d.idBase)
	return d
}

// Count returns the number of documents in the delta buffer, including any since
// deleted (a tombstone does not shrink the buffer; Compact reclaims the space).
func (d *Delta) Count() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.vecs)
}

// IDBase returns the first docID the delta assigns, the immutable region's count at
// the time the delta was attached. Delta documents occupy [IDBase, IDBase+Count).
func (d *Delta) IDBase() uint32 { return d.idBase }

// Add encodes one new document into the delta and returns its docID. The docID is
// idBase plus the delta-local insertion index, so it never collides with an immutable
// docID. The encode mirrors the immutable builder exactly: rotate, one-bit or
// multi-bit code, int8 row, then insert into the delta graph. Every per-document slice
// is appended before the graph insert, because the insert's distance reads them during
// its descent.
func (d *Delta) Add(vec []float32) (uint32, error) {
	if len(vec) != d.r.Dim() {
		return 0, ErrDeltaDim
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	r := d.r
	oRot := r.rot.rotate(vec)

	var scalar, norm float32
	var payload []byte
	var words []uint64
	if r.multibit {
		mc := encodeMulti(oRot, r.codeBits)
		scalar, norm = mc.scalar, mc.norm
		payload = packLevels(mc.levels, r.codeBits)
	} else {
		oc := encodeOneBit(oRot)
		scalar, norm = oc.scalar, oc.norm
		words = oc.bits
		payload = make([]byte, 0, len(words)*8)
		for _, w := range words {
			payload = codec.AppendUint64(payload, w)
		}
	}
	// Round the scalar to the float16 the region stores, and replace the norm with the
	// constant one when the region dropped its norm field (the normalized case), so the
	// estimator scores a delta candidate exactly as it would the same document in the
	// immutable region.
	scalar = codec.Float16frombits(codec.Float16bits(scalar))
	if r.hasStoredNorm {
		norm = codec.Float16frombits(codec.Float16bits(norm))
	} else {
		norm = 1
	}
	row := r.iq.encode(oRot)

	d.scalars = append(d.scalars, scalar)
	d.norms = append(d.norms, norm)
	d.codeBytes = append(d.codeBytes, payload)
	if r.symmetric {
		d.codeWords = append(d.codeWords, words)
	}
	d.rows = append(d.rows, row)
	cp := make([]float32, len(vec))
	copy(cp, vec)
	d.vecs = append(d.vecs, cp)

	local := d.g.addNode(d.efConstruction)
	docID := d.idBase + uint32(local)
	d.ensureTomb(docID + 1)
	return docID, nil
}

// Delete tombstones a docID anywhere in the union. A tombstoned document is skipped at
// candidate collection on the next Search and dropped from the rebuild on the next
// Compact. Deleting an out-of-range or already-deleted id is a no-op.
func (d *Delta) Delete(docID uint32) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.ensureTomb(docID + 1)
	d.setTomb(docID)
}

// Search runs the union dense search of spec doc 05 (line 962): beam the immutable
// graph and the delta graph, drop tombstoned candidates from both, keep the closest
// rerankDepth by the walk metric, rerank that set sharply, and return the top k. Pass
// zero for any of efSearch, efDelta, or rerankDepth to take the defaults.
func (d *Delta) Search(query []float32, k, efSearch, efDelta, rerankDepth int) []Result {
	if k <= 0 {
		return nil
	}
	if efSearch <= 0 {
		efSearch = DefaultEfSearch
	}
	if efDelta <= 0 {
		efDelta = DefaultEfDelta
	}
	if rerankDepth <= 0 {
		rerankDepth = DefaultRerankDepth
	}
	d.mu.RLock()
	defer d.mu.RUnlock()

	r := d.r
	qRot := r.rot.rotate(query)
	qc := encodeQuery(qRot)
	qbits := encodeQueryBits(qRot)
	qi8 := r.iq.encodeQuery(qRot)

	// A merged candidate carries its global docID, its walk distance (smaller nearer, the
	// same metric on both halves so the two are comparable), and enough to find its sharp
	// score: whether it came from the delta and, if so, its delta-local index.
	type gcand struct {
		id    uint32
		dist  float64
		delta bool
		local int32
	}
	var merged []gcand

	if r.h.count > 0 {
		regionCands, _ := r.walk(context.Background(), r.navDist(qRot, qc), efSearch)
		for _, c := range regionCands {
			if d.isTomb(uint32(c.id)) {
				continue
			}
			merged = append(merged, gcand{id: uint32(c.id), dist: c.d})
		}
	}
	if len(d.rows) > 0 {
		for _, c := range d.walk(d.navDist(qRot, qc, qbits, qi8), efDelta) {
			gid := d.idBase + uint32(c.id)
			if d.isTomb(gid) {
				continue
			}
			merged = append(merged, gcand{id: gid, dist: c.d, delta: true, local: c.id})
		}
	}

	sort.Slice(merged, func(i, j int) bool {
		if merged[i].dist != merged[j].dist {
			return merged[i].dist < merged[j].dist
		}
		return merged[i].id < merged[j].id
	})
	if len(merged) > rerankDepth {
		merged = merged[:rerankDepth]
	}

	scored := make([]Result, len(merged))
	for i, m := range merged {
		var s float64
		if m.delta {
			s = d.rerankScore(m.local, qRot, qc)
		} else {
			s = r.rerankScore(int32(m.id), qRot, qc)
		}
		scored[i] = Result{DocID: m.id, Score: s}
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

// navDist returns the delta's walk metric, mirroring Region.navDist over the delta's
// in-RAM arrays so the immutable and delta halves of a union search navigate by the
// same distance and their candidate distances are directly comparable. Smaller is
// nearer; the local index addresses the delta arrays.
func (d *Delta) navDist(qRot []float32, qc queryCode, qbits queryBits, qi8 []int8) func(int32) float64 {
	r := d.r
	switch {
	case r.symmetric:
		return func(local int32) float64 {
			return float64(hammingBytes(d.codeBytes[local], qbits.bits))
		}
	case r.hasRerank:
		return func(local int32) float64 {
			return -float64(dotI8(qi8, d.rows[local]))
		}
	case r.multibit:
		return func(local int32) float64 {
			return -estimateMultiBytes(d.codeBytes[local], r.codeBits, d.scalars[local], d.norms[local], qRot)
		}
	default:
		return func(local int32) float64 {
			return -estimateBytes(d.codeBytes[local], d.scalars[local], d.norms[local], qc)
		}
	}
}

// rerankScore is the sharp final score for one delta document, mirroring
// Region.rerankScore so a delta candidate and an immutable candidate that hold the
// same vector earn the same score. Higher is nearer.
func (d *Delta) rerankScore(local int32, qRot []float32, qc queryCode) float64 {
	r := d.r
	switch {
	case r.hasRerank:
		return float64(r.iq.scale) * dotF32I8(qRot, d.rows[local])
	case r.multibit:
		return estimateMultiBytes(d.codeBytes[local], r.codeBits, d.scalars[local], d.norms[local], qRot)
	default:
		return estimateBytes(d.codeBytes[local], d.scalars[local], d.norms[local], qc)
	}
}

// walk beams the delta graph under the supplied query distance, the same beamSearchQ
// that drives the immutable region, here reading neighbors from the in-RAM graph.
func (d *Delta) walk(distQ func(int32) float64, ef int) []cand {
	cands, _ := beamSearchQ(context.Background(), distQ, d.g.entry, d.g.maxLayer, ef,
		func(node int32, layer int) []int32 { return d.g.neighbors(node, layer) },
		func(node int32) []int32 { return d.g.neighbors(node, 0) })
	return cands
}

// Compact folds the delta into the immutable region by rebuilding the whole shard over
// the live union, the FreshDiskANN compaction of spec doc 05 (line 1004). source must
// return the original vector for an immutable docID in [0, idBase); the delta supplies
// its own originals. The rebuild walks the immutable docs in docID order then the delta
// docs in insertion order, skipping tombstones, and runs them through the same Builder
// the shard was first built with (same seed, graph params, and mode), so the result is
// deterministic and byte-identical across runs given the same live set. The returned
// bytes are a fresh region the caller writes in place of the old shard; the delta is
// then discarded.
func (d *Delta) Compact(source func(docID uint32) []float32) ([]byte, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	r := d.r
	b := NewBuilder(r.Dim()).
		WithSeed(r.h.rotationSeed).
		WithHNSW(int(r.h.m), int(r.h.m0), int(r.h.efConstruction)).
		WithNormalized(!r.hasStoredNorm)
	if r.multibit {
		b.WithCodeBits(r.codeBits)
	}
	if r.symmetric {
		b.WithSymmetricWalk(true)
	}
	// WithRerank last: WithCodeBits forces rerank off for the multi-bit form, and this
	// restores the immutable region's actual setting for the one-bit form.
	b.WithRerank(r.hasRerank)

	for i := uint32(0); i < r.h.count; i++ {
		if d.isTomb(i) {
			continue
		}
		v := source(i)
		if v == nil {
			return nil, fmt.Errorf("vector: compact source returned nil for docID %d", i)
		}
		b.Add(v)
	}
	for local := 0; local < len(d.vecs); local++ {
		if d.isTomb(d.idBase + uint32(local)) {
			continue
		}
		b.Add(d.vecs[local])
	}
	return b.Build()
}

// ensureTomb grows the tombstone bitset to cover ids below n.
func (d *Delta) ensureTomb(n uint32) {
	need := int((n + 63) / 64)
	for len(d.tomb) < need {
		d.tomb = append(d.tomb, 0)
	}
}

func (d *Delta) setTomb(id uint32) {
	w := int(id >> 6)
	if w >= len(d.tomb) {
		return
	}
	d.tomb[w] |= 1 << (id & 63)
}

func (d *Delta) isTomb(id uint32) bool {
	w := int(id >> 6)
	if w >= len(d.tomb) {
		return false
	}
	return d.tomb[w]&(1<<(id&63)) != 0
}
