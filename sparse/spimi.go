package sparse

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// SpimiBuilder builds an IMP1 impact region without holding every posting in memory,
// the build the spec's scale target needs where a large shard's learned-sparse postings
// overflow RAM. It mirrors the lexical SPIMI build: Add appends (term, docID, weight)
// records to a bounded buffer; a full buffer is sorted by term then docID and spilled to
// a run file; Build k-way merges the sorted runs into one ascending term stream and feeds
// it to the same encoder the in-memory Builder uses, so the output is byte-identical.
//
// The global weight range the log quantizer needs is tracked as records arrive rather
// than scanned at the end, so the build never has to revisit the spilled postings to find
// it. Peak memory is the record buffer plus the merge's per-run read buffers, neither of
// which grows with the number of postings.
type SpimiBuilder struct {
	docCount  uint32
	blockSize uint32
	dir       string
	maxBytes  int

	wmin, wmax float64
	first      bool

	buf      []spimiRec
	bufBytes int
	runs     []string
	spillSeq int
	err      error
}

// defaultSpimiBudget is the record-buffer budget before a spill when the caller passes a
// non-positive maxBytes.
const defaultSpimiBudget = 64 << 20

// spimiRec is one impact record: a term occurrence in a document with its raw weight.
// Unlike the in-memory builder a (term, docID) pair may appear more than once across the
// records, so the merge keeps the strongest weight per docID, the same as dedupByDoc.
type spimiRec struct {
	term   string
	docID  uint32
	weight float64
}

// NewSpimiBuilder starts an external-merge build over a dense docID space of size
// docCount, spilling run files under dir and holding at most about maxBytes of records in
// memory before each spill. A non-positive maxBytes uses defaultSpimiBudget. dir must
// exist and be writable; the builder removes its run files when Build finishes.
func NewSpimiBuilder(docCount uint32, dir string, maxBytes int) *SpimiBuilder {
	if maxBytes <= 0 {
		maxBytes = defaultSpimiBudget
	}
	return &SpimiBuilder{
		docCount:  docCount,
		blockSize: DefaultBlockSize,
		dir:       dir,
		maxBytes:  maxBytes,
		first:     true,
	}
}

// WithBlockSize overrides the block width before any postings are added.
func (s *SpimiBuilder) WithBlockSize(n uint32) *SpimiBuilder {
	s.blockSize = n
	return s
}

// Add records a learned impact weight for a term in a document, spilling a run when the
// buffer fills. It applies the same guards as the in-memory builder, dropping a
// non-positive weight or an out-of-range docID, and folds the weight into the running
// global range. Errors are held and surfaced by Build.
func (s *SpimiBuilder) Add(term string, docID uint32, weight float64) {
	if s.err != nil {
		return
	}
	if weight <= 0 || docID >= s.docCount {
		return
	}
	if s.first || weight < s.wmin {
		s.wmin = weight
	}
	if s.first || weight > s.wmax {
		s.wmax = weight
	}
	s.first = false
	s.buf = append(s.buf, spimiRec{term: term, docID: docID, weight: weight})
	s.bufBytes += recSize(term)
	if s.bufBytes >= s.maxBytes {
		s.spill()
	}
}

// recSize estimates a buffered record's resident footprint, enough to trigger spills near
// the budget.
func recSize(term string) int {
	return len(term) + 32
}

// spill sorts the buffered records by term then docID and writes them to a new run file,
// then resets the buffer. A write error is held and returned by Build.
func (s *SpimiBuilder) spill() {
	if len(s.buf) == 0 {
		return
	}
	sort.Slice(s.buf, func(i, j int) bool {
		if s.buf[i].term != s.buf[j].term {
			return s.buf[i].term < s.buf[j].term
		}
		return s.buf[i].docID < s.buf[j].docID
	})
	path := filepath.Join(s.dir, fmt.Sprintf("run-%06d.tmp", s.spillSeq))
	s.spillSeq++
	if err := writeRun(path, s.buf); err != nil {
		if s.err == nil {
			s.err = err
		}
		return
	}
	s.runs = append(s.runs, path)
	s.buf = s.buf[:0]
	s.bufBytes = 0
}

// Spills reports how many run files the build has written so far, the build's memory
// pressure made visible.
func (s *SpimiBuilder) Spills() int { return s.spillSeq }

// Build flushes the final run, builds the quantizer from the tracked weight range, k-way
// merges every run into one ascending term stream keeping the strongest weight per docID,
// and frames the region through the shared encoder, so its bytes match the in-memory
// builder's. It removes the run files before returning.
func (s *SpimiBuilder) Build() ([]byte, error) {
	if s.err != nil {
		return nil, s.err
	}
	s.spill()
	if s.err != nil {
		return nil, s.err
	}
	defer s.cleanup()

	q := newQuantizer(s.wmin, s.wmax)

	m, err := newMerger(s.runs)
	if err != nil {
		return nil, err
	}
	defer m.close()

	out := assembleRegion(s.docCount, s.blockSize, q, m.nextTerm)
	if m.err != nil {
		return nil, m.err
	}
	return out, nil
}

// cleanup removes the spilled run files after the merger has closed them.
func (s *SpimiBuilder) cleanup() {
	for _, p := range s.runs {
		_ = os.Remove(p)
	}
	s.runs = nil
}

// writeRun encodes records to a run file in their given (term, docID) order. Each record
// is a uvarint term length, the term bytes, a uvarint docID, and the weight as eight
// raw float64 bits, the same shape advance decodes.
func writeRun(path string, recs []spimiRec) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	bw := bufio.NewWriter(f)
	var buf []byte
	for i := range recs {
		buf = encodeRec(buf[:0], &recs[i])
		if _, werr := bw.Write(buf); werr != nil {
			return werr
		}
	}
	return bw.Flush()
}

// encodeRec appends one record's wire form to out.
func encodeRec(out []byte, r *spimiRec) []byte {
	out = codec.AppendUvarint(out, uint64(len(r.term)))
	out = append(out, r.term...)
	out = codec.AppendUvarint(out, uint64(r.docID))
	out = codec.AppendUint64(out, math.Float64bits(r.weight))
	return out
}

// runCursor streams records from one run file in order, holding the next record in rec.
type runCursor struct {
	f   *os.File
	br  *bufio.Reader
	rec spimiRec
}

// newRunCursor opens a run file and reads its first record. hasFirst is false for an
// empty run, in which case the file is already closed.
func newRunCursor(path string) (*runCursor, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	c := &runCursor{f: f, br: bufio.NewReader(f)}
	ok, err := c.advance()
	if err != nil {
		_ = c.Close()
		return nil, false, err
	}
	if !ok {
		_ = c.Close()
		return nil, false, nil
	}
	return c, true, nil
}

// advance reads the next record into c.rec. ok is false at a clean end of file.
func (c *runCursor) advance() (bool, error) {
	tl, err := binary.ReadUvarint(c.br)
	if err == io.EOF {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	term := make([]byte, tl)
	if _, err := io.ReadFull(c.br, term); err != nil {
		return false, err
	}
	docID, err := binary.ReadUvarint(c.br)
	if err != nil {
		return false, err
	}
	var bits [8]byte
	if _, err := io.ReadFull(c.br, bits[:]); err != nil {
		return false, err
	}
	c.rec = spimiRec{
		term:   string(term),
		docID:  uint32(docID),
		weight: math.Float64frombits(codec.Uint64(bits[:])),
	}
	return true, nil
}

// Close closes the run file. It is safe to call more than once.
func (c *runCursor) Close() error {
	if c.f == nil {
		return nil
	}
	err := c.f.Close()
	c.f = nil
	return err
}

// cursorHeap orders run cursors by their next record's (term, docID), so the root is the
// globally smallest unconsumed record.
type cursorHeap []*runCursor

func (h cursorHeap) Len() int { return len(h) }
func (h cursorHeap) Less(i, j int) bool {
	if h[i].rec.term != h[j].rec.term {
		return h[i].rec.term < h[j].rec.term
	}
	return h[i].rec.docID < h[j].rec.docID
}
func (h cursorHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *cursorHeap) Push(x any)   { *h = append(*h, x.(*runCursor)) }
func (h *cursorHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// merger performs the k-way merge over the sorted run files. For one term it groups the
// records by docID and keeps the strongest weight per docID, exactly what dedupByDoc does
// for the in-memory build, so the postings it hands the encoder match byte for byte.
type merger struct {
	h       *cursorHeap
	cursors []*runCursor
	err     error
}

// newMerger opens every run, primes its first record, and heapifies the live cursors.
func newMerger(paths []string) (*merger, error) {
	m := &merger{h: &cursorHeap{}}
	for _, p := range paths {
		c, ok, err := newRunCursor(p)
		if err != nil {
			m.close()
			return nil, err
		}
		if !ok {
			continue
		}
		m.cursors = append(m.cursors, c)
		*m.h = append(*m.h, c)
	}
	heap.Init(m.h)
	return m, nil
}

// nextTerm returns the smallest remaining term with its postings deduped to one per
// docID, strongest weight kept, in docID order, the termSource assembleRegion pulls from.
func (m *merger) nextTerm() (string, []posting, bool) {
	if m.err != nil || m.h.Len() == 0 {
		return "", nil, false
	}
	term := (*m.h)[0].rec.term
	var ps []posting
	for m.h.Len() > 0 && (*m.h)[0].rec.term == term {
		cur := (*m.h)[0]
		rec := cur.rec
		if n := len(ps); n > 0 && ps[n-1].docID == rec.docID {
			// Same document seen again; keep the strongest weight, matching dedupByDoc.
			if rec.weight > ps[n-1].weight {
				ps[n-1].weight = rec.weight
			}
		} else {
			ps = append(ps, posting{docID: rec.docID, weight: rec.weight})
		}
		ok, err := cur.advance()
		if err != nil {
			m.err = err
			return "", nil, false
		}
		if ok {
			heap.Fix(m.h, 0)
		} else {
			heap.Pop(m.h)
			_ = cur.Close()
		}
	}
	return term, ps, true
}

// close releases any run files still open.
func (m *merger) close() {
	for _, c := range m.cursors {
		_ = c.Close()
	}
}
