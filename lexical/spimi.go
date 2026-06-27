package lexical

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// SpimiBuilder builds a lexical region without holding the whole inverted index in
// memory, the build the spec calls for at fleet scale where a 2B-doc shard's postings
// dwarf RAM. It is single-pass memory-indexing with constant-memory inversion (SPIMI):
// AddDoc inverts a document into (term, docID, field-tf) records into a bounded buffer;
// when the buffer fills it is sorted by term then docID and spilled to a run file on
// disk; Build k-way merges the sorted runs into one ascending term stream and feeds it
// to the same encoder the in-memory Builder uses. Peak memory is the buffer plus the
// merge's per-run read buffers plus the per-doc norms, none of which grows with the
// number of postings, so the build scales to a corpus far larger than memory.
//
// The norms table stays resident because it is one small fixed record per document and
// the block-max computation needs random access to it, but that is O(docs), not
// O(postings), and the postings are what overflow RAM. The output is byte-identical to
// Builder.Build for the same documents and params, which is what lets every correctness
// test written against the in-memory builder stand for this one too.
type SpimiBuilder struct {
	params   Params
	dir      string
	maxBytes int

	norms   map[uint32]*[numFields]uint32
	maxDoc  uint32
	hasDocs bool

	buf      []spimiRec
	bufBytes int
	runs     []string
	spillSeq int
	err      error
	codec    docCodec
}

// defaultSpimiBudget is the in-memory record-buffer budget before a spill, used when
// the caller passes a non-positive maxBytes. 64 MiB keeps the resident posting buffer
// small while making run files large enough that the merge fan-in stays modest.
const defaultSpimiBudget = 64 << 20

// spimiRec is one inverted record: a term occurrence in a document with the term's
// per-field frequencies in that document. AddDoc aggregates a document's field counts
// before emitting one record per distinct term, so each (term, docID) pair is unique
// across the whole build and the merge never has to combine two records.
type spimiRec struct {
	term    string
	docID   uint32
	fieldTF [numFields]uint32
}

// NewSpimiBuilder starts an external-merge build that spills run files under dir and
// holds at most about maxBytes of records in memory before each spill. A non-positive
// maxBytes uses defaultSpimiBudget. dir must exist and be writable; the builder removes
// its run files when Build finishes.
func NewSpimiBuilder(params Params, dir string, maxBytes int) *SpimiBuilder {
	if maxBytes <= 0 {
		maxBytes = defaultSpimiBudget
	}
	return &SpimiBuilder{
		params:   params,
		dir:      dir,
		maxBytes: maxBytes,
		norms:    map[uint32]*[numFields]uint32{},
		codec:    varintCodec{},
	}
}

// WithDocCodec selects the docID gap codec before any documents are added, matching
// Builder.WithDocCodec so the two builders produce byte-identical regions under the
// same codec. An unknown id keeps the default.
func (s *SpimiBuilder) WithDocCodec(id uint16) *SpimiBuilder {
	if dc, err := codecByID(id); err == nil {
		s.codec = dc
	}
	return s
}

// AddDoc inverts one document into the record buffer, spilling a run when the buffer
// fills. It analyzes each field, counts per-field term frequencies for the document,
// records the document's per-field token lengths for length normalization, then emits
// one record per distinct term. docID is the dense id the document occupies. Errors are
// held and surfaced by Build so the caller can drive a tight add loop.
func (s *SpimiBuilder) AddDoc(docID uint32, fields map[Field]string) {
	if s.err != nil {
		return
	}
	if !s.hasDocs || docID > s.maxDoc {
		s.maxDoc = docID
		s.hasDocs = true
	}
	dn := s.norms[docID]
	if dn == nil {
		dn = &[numFields]uint32{}
		s.norms[docID] = dn
	}
	// Aggregate this document's per-field term frequencies first, so each term emits a
	// single record carrying the full field vector, matching the in-memory builder's
	// per-(term, docID) aggregation.
	local := map[string]*[numFields]uint32{}
	for f, text := range fields {
		toks := Analyze(text)
		dn[f] += uint32(len(toks))
		for _, tok := range toks {
			tf := local[tok]
			if tf == nil {
				tf = &[numFields]uint32{}
				local[tok] = tf
			}
			tf[f]++
		}
	}
	for term, tf := range local {
		s.buf = append(s.buf, spimiRec{term: term, docID: docID, fieldTF: *tf})
		s.bufBytes += recSize(term)
	}
	if s.bufBytes >= s.maxBytes {
		s.spill()
	}
}

// recSize estimates a buffered record's resident footprint: the struct (string header,
// docID, field vector) plus the term's backing bytes. It only has to track growth well
// enough to trigger spills near the budget, not be exact.
func recSize(term string) int {
	return len(term) + 40
}

// spill sorts the buffered records by term then docID and writes them to a new run
// file, then resets the buffer. An empty buffer is a no-op. A write error is held and
// returned by Build.
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

// Spills reports how many run files the build has written so far, the number of times
// the record buffer filled and flushed to disk. It is the build's memory pressure made
// visible: zero means the whole corpus fit in one buffer, a large count means the build
// stayed within its budget by spilling, which is the behavior the external merge exists
// to provide.
func (s *SpimiBuilder) Spills() int { return s.spillSeq }

// docCount returns N, the dense docID space size.
func (s *SpimiBuilder) docCount() uint32 {
	if !s.hasDocs {
		return 0
	}
	return s.maxDoc + 1
}

// Build flushes the final run, k-way merges every run into one ascending term stream,
// and encodes the region through the shared assembler, so its bytes match the in-memory
// builder's for the same input. It removes the run files before returning.
func (s *SpimiBuilder) Build() ([]byte, error) {
	if s.err != nil {
		return nil, s.err
	}
	s.spill()
	if s.err != nil {
		return nil, s.err
	}
	defer s.cleanup()

	n := s.docCount()
	st := computeStats(n, s.norms)
	norms := normsTable(n, s.norms)

	m, err := newMerger(s.runs)
	if err != nil {
		return nil, err
	}
	defer m.close()

	out := assembleRegion(n, norms, st, s.params, fieldLenFrom(s.norms), s.codec, m.nextTerm)
	if m.err != nil {
		return nil, m.err
	}
	return out, nil
}

// cleanup removes the spilled run files. It runs after the merger has closed them.
func (s *SpimiBuilder) cleanup() {
	for _, p := range s.runs {
		_ = os.Remove(p)
	}
	s.runs = nil
}

// writeRun encodes records to a run file in their given order. Each record is a uvarint
// term length, the term bytes, a uvarint docID, a one-byte field mask, and a uvarint per
// set field, the same shape readRec decodes. The records are written in (term, docID)
// order so the file is a sorted run the merger can stream.
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
	var mask uint8
	for f := 0; f < numFields; f++ {
		if r.fieldTF[f] > 0 {
			mask |= 1 << uint8(f)
		}
	}
	out = append(out, mask)
	for f := 0; f < numFields; f++ {
		if r.fieldTF[f] > 0 {
			out = codec.AppendUvarint(out, uint64(r.fieldTF[f]))
		}
	}
	return out
}

// runCursor streams records from one run file in order, holding the next record to be
// consumed in rec. The bufio.Reader keeps the per-run read buffer small and bounded.
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
	mask, err := c.br.ReadByte()
	if err != nil {
		return false, err
	}
	var ft [numFields]uint32
	for f := 0; f < numFields; f++ {
		if mask&(1<<uint8(f)) != 0 {
			v, verr := binary.ReadUvarint(c.br)
			if verr != nil {
				return false, verr
			}
			ft[f] = uint32(v)
		}
	}
	c.rec = spimiRec{term: string(term), docID: uint32(docID), fieldTF: ft}
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

// cursorHeap orders run cursors by their next record's (term, docID), so the heap root
// is always the globally smallest unconsumed record across the runs.
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

// merger performs the k-way merge over the sorted run files, grouping consecutive
// records with the same term into one posting list. Because the runs are each sorted by
// (term, docID) and a (term, docID) pair is unique across the build, the merge yields
// terms in ascending order with each term's postings in ascending docID order, exactly
// what the encoder wants.
type merger struct {
	h       *cursorHeap
	cursors []*runCursor
	err     error
}

// newMerger opens every run, primes its first record, and heapifies the live cursors.
// Empty runs are skipped.
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

// nextTerm returns the smallest remaining term and all its postings in docID order, or
// ok false when the merge is drained or an error has been recorded. It is the termSource
// assembleRegion pulls from.
func (m *merger) nextTerm() (string, []posting, bool) {
	if m.err != nil || m.h.Len() == 0 {
		return "", nil, false
	}
	term := (*m.h)[0].rec.term
	var ps []posting
	for m.h.Len() > 0 && (*m.h)[0].rec.term == term {
		cur := (*m.h)[0]
		ps = append(ps, posting{docID: cur.rec.docID, fieldTF: cur.rec.fieldTF})
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

// close releases any run files still open, used on the error path and after the merge.
func (m *merger) close() {
	for _, c := range m.cursors {
		_ = c.Close()
	}
}
