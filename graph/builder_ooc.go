package graph

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// OOCBuilder encodes the same GRA1 region as Builder, but without ever holding
// the whole adjacency in RAM. Edges arrive as (from, to) dense docID pairs in any
// order; they are buffered and, once the buffer fills, sorted and spilled to run
// files on disk. Build joins the runs with an external merge sort and streams the
// sorted edge groups through a windowed encoder that keeps only the last Window
// adjacency records resident, the reference window the layered codec needs.
//
// The output is byte-identical to Builder.Build over the same edge set: both share
// encodeNodeCore for the record coding and dedup edges the same way, so a reader
// cannot tell which builder produced a region, and the equivalence is gated by
// test. The corpus graph has billions of edges past RAM, so doc 06's build path is
// "streaming passes joined by external sorts"; this is the transpose-and-compress
// stage of that pipeline, the one that would otherwise materialize the full
// forward and transpose planes as [][]int32.
//
// When the whole edge set fits under the spill threshold nothing touches disk: the
// builder sorts the buffer in RAM twice (by source, then by target) and encodes,
// which is the per-shard case doc 06 says runs in memory. Disk is reached only when
// the edge count exceeds the threshold, the corpus-wide case.
type OOCBuilder struct {
	n       int
	params  Params
	spillN  int      // edges held in RAM before a run is spilled
	nodeIDs []uint64 // per-dense-docID global node id, nil for the identity mapping

	buf   []oocEdge   // current in-RAM batch, sorted and spilled when it reaches spillN
	runs  []string    // spilled run files, each sorted by (from, to)
	dir   string      // temp dir for run files, created on the first spill
	err   error       // first spill error, surfaced at Build
	cross []crossEdge // far out-edges, kept in RAM: a small fraction of the edges
}

// oocEdge is one directed edge as a pair of dense node ids. uint32 holds the
// full dense range (up to ~4.29e9 nodes, past the 2e9-node target) in 8 bytes a
// record, which is what the run files store.
type oocEdge struct{ from, to uint32 }

// defaultSpillEdges is the edge buffer cap before a run spills: 16M edges, about
// 128MB of oocEdge records. Below this a build stays entirely in RAM.
const defaultSpillEdges = 16 << 20

// NewOOCBuilder returns an out-of-core builder over a dense node space [0, n).
func NewOOCBuilder(n int) *OOCBuilder {
	return &OOCBuilder{n: n, params: DefaultParams(), spillN: defaultSpillEdges}
}

// WithParams overrides the adjacency-coder settings before any edges are added.
func (b *OOCBuilder) WithParams(p Params) *OOCBuilder {
	b.params = p
	return b
}

// WithNodeIDs supplies the global node id of each dense docID so the region carries
// the dense-to-global identity mapping, matching Builder.WithNodeIDs. When it is not
// set the region uses the identity mapping.
func (b *OOCBuilder) WithNodeIDs(ids []uint64) *OOCBuilder {
	b.nodeIDs = ids
	return b
}

// WithSpillThreshold sets how many edges are buffered before a run spills to disk.
// A small value forces the external-sort path for tests; the default keeps a
// per-shard build in RAM.
func (b *OOCBuilder) WithSpillThreshold(edges int) *OOCBuilder {
	if edges > 0 {
		b.spillN = edges
	}
	return b
}

// AddEdge records a directed link from -> to. Self-loops, out-of-range ids, and
// duplicates are dropped (the latter at Build), matching Builder.AddEdge, so a
// caller may add freely.
func (b *OOCBuilder) AddEdge(from, to int) {
	if b.err != nil {
		return
	}
	if from < 0 || from >= b.n || to < 0 || to >= b.n || from == to {
		return
	}
	b.buf = append(b.buf, oocEdge{uint32(from), uint32(to)})
	if len(b.buf) >= b.spillN {
		if err := b.spill(); err != nil {
			b.err = err
		}
	}
}

// AddCrossEdge records a far out-edge from local dense docID from to the global node
// id toGlobal in another shard, matching Builder.AddCrossEdge. The cross edges are a
// small fraction of the total (most edges stay within the shard's own adjacency in
// any one routing pass), so they are buffered in RAM rather than spilled, and framed
// into the region's cross-shard edge list at Build.
func (b *OOCBuilder) AddCrossEdge(from int, toGlobal uint64) {
	if b.err != nil || from < 0 || from >= b.n {
		return
	}
	b.cross = append(b.cross, crossEdge{from: from, to: toGlobal})
}

// Build encodes the forward and transpose planes and frames the region, then
// removes any spilled run files. It returns an error only on a disk failure in the
// spill or merge; on the in-RAM path it never fails.
func (b *OOCBuilder) Build() ([]byte, error) {
	if b.err != nil {
		b.cleanup()
		return nil, b.err
	}
	defer b.cleanup()

	// In-RAM path: nothing spilled, so sort the buffer both ways and encode. This
	// is the per-shard case, where the adjacency fits in memory by design.
	if len(b.runs) == 0 {
		return b.buildInRAM(), nil
	}

	// Out-of-core path: flush the tail, then encode each plane from a merge of
	// sorted runs. The forward runs are already sorted by (from, to); the transpose
	// needs (to, from), so it is a second external sort over the same edges.
	if err := b.spill(); err != nil {
		return nil, err
	}

	fwdNext, fwdClose, err := mergeGroups(b.runs, true)
	if err != nil {
		return nil, err
	}
	fwdAdj, fwdOff, edges := encodePlaneWindowed(b.n, b.params, fwdNext)
	fwdClose()

	xpRuns, err := b.externalSort(b.runs, false)
	if err != nil {
		return nil, err
	}
	xpNext, xpClose, err := mergeGroups(xpRuns, false)
	if err != nil {
		return nil, err
	}
	xpAdj, xpOff, _ := encodePlaneWindowed(b.n, b.params, xpNext)
	xpClose()

	fwdEF := buildEF(fwdOff).encode()
	xpEF := buildEF(xpOff).encode()
	xsBlob := buildCrossBlob(b.cross, b.params)
	return frameRegion(b.n, edges, b.nodeIDs, fwdAdj, fwdEF, xpAdj, xpEF, xsBlob, b.params), nil
}

// buildInRAM sorts the buffered edges by source then by target and encodes both
// planes through the same windowed encoder the disk path uses, so the two paths
// produce identical bytes.
func (b *OOCBuilder) buildInRAM() []byte {
	edges := b.buf
	sortEdges(edges, true)
	fwdAdj, fwdOff, ecount := encodePlaneWindowed(b.n, b.params, groupSlice(edges, true))
	sortEdges(edges, false)
	xpAdj, xpOff, _ := encodePlaneWindowed(b.n, b.params, groupSlice(edges, false))
	fwdEF := buildEF(fwdOff).encode()
	xpEF := buildEF(xpOff).encode()
	xsBlob := buildCrossBlob(b.cross, b.params)
	return frameRegion(b.n, ecount, b.nodeIDs, fwdAdj, fwdEF, xpAdj, xpEF, xsBlob, b.params)
}

// spill sorts the current buffer by (from, to) and writes it as a run file.
func (b *OOCBuilder) spill() error {
	if len(b.buf) == 0 {
		return nil
	}
	sortEdges(b.buf, true)
	if b.dir == "" {
		d, err := os.MkdirTemp("", "tsumugi-graph-ooc-")
		if err != nil {
			return err
		}
		b.dir = d
	}
	path := filepath.Join(b.dir, fmt.Sprintf("run-%05d.edges", len(b.runs)))
	if err := writeRun(path, b.buf); err != nil {
		return err
	}
	b.runs = append(b.runs, path)
	b.buf = b.buf[:0]
	return nil
}

// externalSort reads every edge from the given runs sequentially, re-chunks them
// into spillN-sized batches sorted by the requested order, and writes each batch
// as a new run, returning the new run paths. It is the second sort the transpose
// plane needs: the forward runs are sorted by source, the transpose by target.
func (b *OOCBuilder) externalSort(inputs []string, byFrom bool) ([]string, error) {
	rd, err := newMultiReader(inputs)
	if err != nil {
		return nil, err
	}
	defer rd.close()

	var out []string
	batch := make([]oocEdge, 0, b.spillN)
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		sortEdges(batch, byFrom)
		path := filepath.Join(b.dir, fmt.Sprintf("xp-%05d.edges", len(out)))
		if err := writeRun(path, batch); err != nil {
			return err
		}
		out = append(out, path)
		batch = batch[:0]
		return nil
	}
	for {
		e, ok, err := rd.next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		batch = append(batch, e)
		if len(batch) >= b.spillN {
			if err := flush(); err != nil {
				return nil, err
			}
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return out, nil
}

// cleanup removes the temp dir and its run files.
func (b *OOCBuilder) cleanup() {
	if b.dir != "" {
		_ = os.RemoveAll(b.dir)
		b.dir = ""
		b.runs = nil
	}
}

// sortEdges sorts in place by (from, to) when byFrom, else by (to, from).
func sortEdges(e []oocEdge, byFrom bool) {
	if byFrom {
		sort.Slice(e, func(i, j int) bool {
			if e[i].from != e[j].from {
				return e[i].from < e[j].from
			}
			return e[i].to < e[j].to
		})
		return
	}
	sort.Slice(e, func(i, j int) bool {
		if e[i].to != e[j].to {
			return e[i].to < e[j].to
		}
		return e[i].from < e[j].from
	})
}

// edgeLess reports whether a sorts before b in the order keyed by byFrom, the
// comparator the run merge uses so the merged stream matches the per-run sort.
func edgeLess(a, b oocEdge, byFrom bool) bool {
	if byFrom {
		if a.from != b.from {
			return a.from < b.from
		}
		return a.to < b.to
	}
	if a.to != b.to {
		return a.to < b.to
	}
	return a.from < b.from
}

// groupSlice yields adjacency lists from a slice already sorted in the grouping
// order: by (from, to) when byFrom (forward plane) or by (to, from) otherwise
// (transpose plane). Each call returns the next node's id and its ascending,
// deduped neighbor list, freshly allocated so the encoder's window may retain it.
func groupSlice(edges []oocEdge, byFrom bool) func() (int, []int32, bool) {
	i := 0
	return func() (int, []int32, bool) {
		if i >= len(edges) {
			return 0, nil, false
		}
		prim := primaryOf(edges[i], byFrom)
		var list []int32
		var last int64 = -1
		for i < len(edges) && primaryOf(edges[i], byFrom) == prim {
			sec := secondaryOf(edges[i], byFrom)
			if int64(sec) != last {
				list = append(list, int32(sec))
				last = int64(sec)
			}
			i++
		}
		return int(prim), list, true
	}
}

func primaryOf(e oocEdge, byFrom bool) uint32 {
	if byFrom {
		return e.from
	}
	return e.to
}

func secondaryOf(e oocEdge, byFrom bool) uint32 {
	if byFrom {
		return e.to
	}
	return e.from
}

// encodePlaneWindowed encodes a plane streaming over node ids 0..n-1, pulling each
// node's sorted-deduped list from next (which yields lists in ascending node
// order, only for nodes that have one). It keeps only the last Window+1 records
// resident in a ring so the reference coder can look back up to Window nodes, the
// only state the layered codec needs, so the resident memory is the window and the
// offsets table, never the adjacency. It returns the bitstream, the N+1 bit
// offsets, and the total edge count.
func encodePlaneWindowed(n int, p Params, next func() (int, []int32, bool)) (data []byte, offsets []uint64, edges uint64) {
	w := &bitWriter{}
	offsets = make([]uint64, n+1)
	capacity := p.Window + 1
	ringList := make([][]int32, capacity)
	ringDepth := make([]int, capacity)

	curNode, curList, has := next()
	for x := 0; x < n; x++ {
		offsets[x] = w.bits
		var s []int32
		if has && curNode == x {
			s = curList
			edges += uint64(len(s))
			curNode, curList, has = next()
		}
		xx := x
		listAt := func(back int) []int32 { return ringList[(xx-back)%capacity] }
		depthAt := func(back int) int { return ringDepth[(xx-back)%capacity] }
		depth := encodeNodeCore(w, x, s, listAt, depthAt, p)
		ringList[x%capacity] = s
		ringDepth[x%capacity] = depth
	}
	offsets[n] = w.bits
	return w.finish(), offsets, edges
}

// frameRegion assembles the GRA1 region header and concatenates the region parts
// in doc 03's order: the id table (when the global ids need one), the forward
// adjacency and its offsets, the transpose adjacency and its offsets, then the
// cross-shard edge list. Builder.Build and OOCBuilder.Build share it so the framing
// is one definition. ids is the per-dense-docID global node id, nil for the identity
// mapping; xsBlob is the cross-shard edge list, nil when the graph has no far edges.
func frameRegion(n int, edges uint64, ids []uint64, fwdAdj, fwdEF, xpAdj, xpEF, xsBlob []byte, p Params) []byte {
	nodeBase, idBlob := computeIDTable(n, ids)
	h := header{
		version:    regionVersion,
		params:     p,
		nodeCount:  uint32(n),
		edgeCount:  edges,
		idTableLen: uint64(len(idBlob)),
		nodeBase:   nodeBase,
		fwdAdjLen:  uint64(len(fwdAdj)),
		fwdEFLen:   uint64(len(fwdEF)),
		xpAdjLen:   uint64(len(xpAdj)),
		xpEFLen:    uint64(len(xpEF)),
		xsLen:      uint64(len(xsBlob)),
	}
	region := h.encode()
	region = append(region, idBlob...)
	region = append(region, fwdAdj...)
	region = append(region, fwdEF...)
	region = append(region, xpAdj...)
	region = append(region, xpEF...)
	region = append(region, xsBlob...)
	return region
}

// --- run-file I/O and k-way merge ---

const edgeRecordBytes = 8

// writeRun writes edges as fixed 8-byte little-endian (from, to) records.
func writeRun(path string, edges []oocEdge) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	var rec [edgeRecordBytes]byte
	for _, e := range edges {
		binary.LittleEndian.PutUint32(rec[0:4], e.from)
		binary.LittleEndian.PutUint32(rec[4:8], e.to)
		if _, err := bw.Write(rec[:]); err != nil {
			_ = f.Close()
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// runReader reads 8-byte edge records from one run file in order.
type runReader struct {
	f   *os.File
	br  *bufio.Reader
	rec [edgeRecordBytes]byte
}

func openRunReader(path string) (*runReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &runReader{f: f, br: bufio.NewReaderSize(f, 1<<20)}, nil
}

// next returns the next edge, false at clean EOF.
func (r *runReader) next() (oocEdge, bool, error) {
	if _, err := io.ReadFull(r.br, r.rec[:]); err != nil {
		// A run file is a whole number of records, so a clean EOF lands on a record
		// boundary; a partial read is a real corruption and is surfaced.
		if err == io.EOF {
			return oocEdge{}, false, nil
		}
		return oocEdge{}, false, err
	}
	return oocEdge{
		from: binary.LittleEndian.Uint32(r.rec[0:4]),
		to:   binary.LittleEndian.Uint32(r.rec[4:8]),
	}, true, nil
}

func (r *runReader) close() { _ = r.f.Close() }

// multiReader reads every edge across a set of run files in sequence, used by the
// transpose re-sort that does not care about order, only that it sees them all.
type multiReader struct {
	paths []string
	idx   int
	cur   *runReader
}

func newMultiReader(paths []string) (*multiReader, error) {
	m := &multiReader{paths: paths}
	return m, nil
}

func (m *multiReader) next() (oocEdge, bool, error) {
	for {
		if m.cur == nil {
			if m.idx >= len(m.paths) {
				return oocEdge{}, false, nil
			}
			rr, err := openRunReader(m.paths[m.idx])
			if err != nil {
				return oocEdge{}, false, err
			}
			m.cur = rr
			m.idx++
		}
		e, ok, err := m.cur.next()
		if err != nil {
			return oocEdge{}, false, err
		}
		if ok {
			return e, true, nil
		}
		m.cur.close()
		m.cur = nil
	}
}

func (m *multiReader) close() {
	if m.cur != nil {
		m.cur.close()
		m.cur = nil
	}
}

// mergeHeapItem is one run's current head in the merge heap.
type mergeHeapItem struct {
	edge   oocEdge
	reader *runReader
}

type mergeHeap struct {
	items  []mergeHeapItem
	byFrom bool
}

func (h *mergeHeap) Len() int { return len(h.items) }
func (h *mergeHeap) Less(i, j int) bool {
	return edgeLess(h.items[i].edge, h.items[j].edge, h.byFrom)
}
func (h *mergeHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }
func (h *mergeHeap) Push(x any)    { h.items = append(h.items, x.(mergeHeapItem)) }
func (h *mergeHeap) Pop() any {
	old := h.items
	n := len(old)
	it := old[n-1]
	h.items = old[:n-1]
	return it
}

// mergeGroups opens the run files, k-way merges them in the order keyed by byFrom,
// and returns a group pull function over the merged stream plus a close function.
// The runs must each be sorted in that same order, which spill and externalSort
// guarantee. The merged stream is fully sorted, so grouping and dedup are a single
// linear pass, identical to groupSlice over an in-RAM sort.
func mergeGroups(paths []string, byFrom bool) (func() (int, []int32, bool), func(), error) {
	h := &mergeHeap{byFrom: byFrom}
	for _, p := range paths {
		rr, err := openRunReader(p)
		if err != nil {
			for _, it := range h.items {
				it.reader.close()
			}
			return nil, nil, err
		}
		e, ok, err := rr.next()
		if err != nil {
			rr.close()
			for _, it := range h.items {
				it.reader.close()
			}
			return nil, nil, err
		}
		if !ok {
			rr.close()
			continue
		}
		h.items = append(h.items, mergeHeapItem{edge: e, reader: rr})
	}
	heap.Init(h)

	// pull yields the next edge in sorted order across all runs.
	pull := func() (oocEdge, bool) {
		if h.Len() == 0 {
			return oocEdge{}, false
		}
		top := h.items[0]
		e := top.edge
		nxt, ok, err := top.reader.next()
		if err != nil || !ok {
			top.reader.close()
			heap.Pop(h)
		} else {
			h.items[0].edge = nxt
			heap.Fix(h, 0)
		}
		return e, true
	}

	cur, has := pull()
	group := func() (int, []int32, bool) {
		if !has {
			return 0, nil, false
		}
		prim := primaryOf(cur, byFrom)
		var list []int32
		var last int64 = -1
		for has && primaryOf(cur, byFrom) == prim {
			sec := secondaryOf(cur, byFrom)
			if int64(sec) != last {
				list = append(list, int32(sec))
				last = int64(sec)
			}
			cur, has = pull()
		}
		return int(prim), list, true
	}
	closeFn := func() {
		for _, it := range h.items {
			it.reader.close()
		}
		h.items = nil
	}
	return group, closeFn, nil
}
