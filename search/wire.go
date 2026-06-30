package search

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// The RPC seam in remote.go carries one message in each direction on the per-query hot path:
// the Query an aggregator pushes down to a child and the Results the child sends back. Every
// other route a RemoteSearcher calls, the metadata snapshot, the document frequencies, the
// vocabulary stream, runs once per peer at construction or off the serving path, so the bytes
// that ride the 10ms budget are these two. remote.go marshals them as JSON, the field-name
// default, which is the right first cut: a deployment needs no extra dependency, and the wire
// is human-readable for debugging.
//
// JSON pays for that readability in size on exactly the two messages that repeat per query. A
// Query carries a dense vector, written as a decimal-text array where each float32 takes a
// dozen-odd characters in place of the four bytes it is, plus the idf and sparse maps written
// as quoted keys and decimal values. A Results carries the top-k, each hit a decimal document
// id and a full-precision float64 score written long. Across a fleet, an aggregator sends the
// query to every leaf and reads a Results back from each, so the per-query wire is paid once
// per leaf per query, which is where a denser codec earns its keep.
//
// wireCodec is the seam the remote.go comment names: the encoding of the two hot-path messages
// behind one interface, so the JSON form and a compact binary form are interchangeable and the
// aggregator, the broker, and the handler never see which is in use. The default stays JSON,
// so an existing deployment and every test is byte-for-byte unchanged; a deployment that opts
// into the binary codec (WithBinaryWire) swaps it on the RemoteSearcher, and the handler reads
// the request's content type to answer in the same codec, so a binary client and a JSON client
// can both reach one handler and each gets its own wire back.
type wireCodec interface {
	// contentType is the HTTP Content-Type a request carries so the handler decodes it with the
	// matching codec and answers in kind.
	contentType() string
	encodeQuery(q Query) ([]byte, error)
	decodeQuery(b []byte) (Query, error)
	encodeResults(r Results) ([]byte, error)
	decodeResults(b []byte) (Results, error)
}

const (
	// wireContentTypeJSON is the default wire's content type, the field-name JSON remote.go has
	// always spoken, kept so a RemoteSearcher with no codec option is unchanged on the wire.
	wireContentTypeJSON = "application/json"
	// wireContentTypeBinary is the dense wire's content type, a tsumugi-specific media type so a
	// handler tells a binary request from a JSON one and never mistakes one codec's bytes for the
	// other's.
	wireContentTypeBinary = "application/x-tsumugi-wire"
)

// jsonCodec is the default wire: encoding/json over the Query and Results types, the same
// encoding remote.go marshalled inline before the seam existed. It exists so JSON is one
// implementation of the interface rather than a special case the handler branches on, and so a
// RemoteSearcher with no binary option behaves exactly as it did before.
type jsonCodec struct{}

func (jsonCodec) contentType() string { return wireContentTypeJSON }

func (jsonCodec) encodeQuery(q Query) ([]byte, error) { return json.Marshal(q) }

func (jsonCodec) decodeQuery(b []byte) (Query, error) {
	var q Query
	err := json.Unmarshal(b, &q)
	return q, err
}

func (jsonCodec) encodeResults(r Results) ([]byte, error) { return json.Marshal(r) }

func (jsonCodec) decodeResults(b []byte) (Results, error) {
	var r Results
	err := json.Unmarshal(b, &r)
	return r, err
}

// codecForContentType returns the codec a handler decodes a request with and answers it in,
// chosen from the request's Content-Type so a binary request gets a binary answer and anything
// else, including a missing or unknown type, falls back to JSON. Defaulting unknown to JSON
// keeps a handler answering an old or hand-rolled JSON client even as it learns the binary wire.
func codecForContentType(ct string) wireCodec {
	if mediaType(ct) == wireContentTypeBinary {
		return binaryCodec{}
	}
	return jsonCodec{}
}

// mediaType strips any parameters (a "; charset=..." suffix) from a Content-Type header and
// trims it, so a header that carries parameters still matches a bare media type.
func mediaType(ct string) string {
	for i := 0; i < len(ct); i++ {
		if ct[i] == ';' {
			ct = ct[:i]
			break
		}
	}
	// Trim surrounding spaces without importing strings for one call.
	for len(ct) > 0 && ct[0] == ' ' {
		ct = ct[1:]
	}
	for len(ct) > 0 && ct[len(ct)-1] == ' ' {
		ct = ct[:len(ct)-1]
	}
	return ct
}

// binaryCodec is the dense wire: a little-endian binary encoding of the two hot-path messages
// that writes a float32 as its four bytes rather than a dozen decimal characters, a document id
// as four bytes rather than up to ten, and a score as its eight IEEE-754 bytes rather than a
// long decimal. It is bit-preserving on the floats (math.Float*bits round-trips exactly), so a
// score sent over the binary wire decodes to the same float64 the in-process broker produced,
// the exactness the merge up the tree depends on. Map fields are written in sorted key order so
// the encoding is deterministic, the byte-identical-build property the rest of the engine holds,
// carried onto the wire.
type binaryCodec struct{}

func (binaryCodec) contentType() string { return wireContentTypeBinary }

// encodeQuery writes a Query in the dense form. Every nilable field carries a one-byte present
// flag before its contents so a nil slice or map round-trips as nil and an empty one as empty,
// the distinction Query.lexTerms turns on (a nil Terms means re-analyze Text, an empty Terms
// means no terms) and the one json.Marshal preserves through null, kept here so the two codecs
// decode the same value.
func (binaryCodec) encodeQuery(q Query) ([]byte, error) {
	buf := make([]byte, 0, 64+len(q.Vector)*4)
	buf = putString(buf, q.Text)
	buf = binary.AppendUvarint(buf, uint64(q.K))
	buf = binary.AppendVarint(buf, int64(q.L0))

	buf = putStringSlice(buf, q.Terms)

	// Sparse: present flag, count, then key/value pairs in sorted key order.
	if q.Sparse == nil {
		buf = append(buf, 0)
	} else {
		buf = append(buf, 1)
		buf = binary.AppendUvarint(buf, uint64(len(q.Sparse)))
		for _, k := range sortedKeysInt(q.Sparse) {
			buf = putString(buf, k)
			buf = binary.AppendVarint(buf, int64(q.Sparse[k]))
		}
	}

	// Vector: present flag, count, then each float32 as four little-endian bytes.
	if q.Vector == nil {
		buf = append(buf, 0)
	} else {
		buf = append(buf, 1)
		buf = binary.AppendUvarint(buf, uint64(len(q.Vector)))
		for _, f := range q.Vector {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(f))
		}
	}

	// TermIDF: present flag, count, then key/value pairs in sorted key order.
	if q.TermIDF == nil {
		buf = append(buf, 0)
	} else {
		buf = append(buf, 1)
		buf = binary.AppendUvarint(buf, uint64(len(q.TermIDF)))
		for _, k := range sortedKeysFloat(q.TermIDF) {
			buf = putString(buf, k)
			buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(q.TermIDF[k]))
		}
	}

	// AvgFieldLen: a present flag, then the three field averages when set.
	if q.AvgFieldLen == nil {
		buf = append(buf, 0)
	} else {
		buf = append(buf, 1)
		for _, f := range q.AvgFieldLen {
			buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(f))
		}
	}
	return buf, nil
}

func (binaryCodec) decodeQuery(b []byte) (Query, error) {
	r := reader{b: b}
	var q Query
	q.Text = r.string()
	q.K = int(r.uvarint())
	q.L0 = int(r.varint())

	q.Terms = r.stringSlice()

	if r.flag() {
		n := int(r.uvarint())
		q.Sparse = make(map[string]int, n)
		for i := 0; i < n; i++ {
			k := r.string()
			q.Sparse[k] = int(r.varint())
		}
	}

	if r.flag() {
		n := int(r.uvarint())
		q.Vector = make([]float32, n)
		for i := 0; i < n; i++ {
			q.Vector[i] = r.f32()
		}
	}

	if r.flag() {
		n := int(r.uvarint())
		q.TermIDF = make(map[string]float64, n)
		for i := 0; i < n; i++ {
			k := r.string()
			q.TermIDF[k] = r.f64()
		}
	}

	if r.flag() {
		var a [3]float64
		a[0], a[1], a[2] = r.f64(), r.f64(), r.f64()
		q.AvgFieldLen = &a
	}

	if r.err != nil {
		return Query{}, fmt.Errorf("decode query: %w", r.err)
	}
	return q, nil
}

// encodeResults writes a Results in the dense form: the two shard counts and the degradation
// rung, then the hits, each a four-byte document id and an eight-byte score. The hits carry a
// present flag so a nil hit slice (the dropped-subtree result SearchComplete returns) round-trips
// as nil rather than as an empty slice, matching what json.Marshal does through null.
func (binaryCodec) encodeResults(res Results) ([]byte, error) {
	buf := make([]byte, 0, 16+len(res.Hits)*12)
	buf = binary.AppendVarint(buf, int64(res.ShardsTotal))
	buf = binary.AppendVarint(buf, int64(res.ShardsOK))
	buf = binary.AppendUvarint(buf, uint64(res.Degraded))
	if res.Hits == nil {
		buf = append(buf, 0)
	} else {
		buf = append(buf, 1)
		buf = binary.AppendUvarint(buf, uint64(len(res.Hits)))
		for _, h := range res.Hits {
			buf = binary.LittleEndian.AppendUint32(buf, h.DocID)
			buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(h.Score))
		}
	}
	return buf, nil
}

func (binaryCodec) decodeResults(b []byte) (Results, error) {
	r := reader{b: b}
	var res Results
	res.ShardsTotal = int(r.varint())
	res.ShardsOK = int(r.varint())
	res.Degraded = DegradeLevel(r.uvarint())
	if r.flag() {
		n := int(r.uvarint())
		res.Hits = make([]Hit, n)
		for i := 0; i < n; i++ {
			res.Hits[i].DocID = r.u32()
			res.Hits[i].Score = r.f64()
		}
	}
	if r.err != nil {
		return Results{}, fmt.Errorf("decode results: %w", r.err)
	}
	return res, nil
}

// putString writes a length-prefixed string: an unsigned varint length then the bytes.
func putString(buf []byte, s string) []byte {
	buf = binary.AppendUvarint(buf, uint64(len(s)))
	return append(buf, s...)
}

// putStringSlice writes a present flag, then a count and the strings when the slice is non-nil,
// so a nil slice round-trips as nil and an empty one as empty.
func putStringSlice(buf []byte, ss []string) []byte {
	if ss == nil {
		return append(buf, 0)
	}
	buf = append(buf, 1)
	buf = binary.AppendUvarint(buf, uint64(len(ss)))
	for _, s := range ss {
		buf = putString(buf, s)
	}
	return buf
}

// sortedKeysInt and sortedKeysFloat return a map's keys in sorted order so the encoding of a
// map is deterministic regardless of Go's randomized map iteration, the same byte-identical
// property the build holds, carried onto the wire.
func sortedKeysInt(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func sortedKeysFloat(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// reader is a cursor over a binary message that reads the encoder's fields in order and stops
// at the first malformed or truncated field, recording the error so the caller checks it once
// at the end rather than on every read. A read after an error is a no-op returning the zero
// value, so a truncated message cannot panic the decoder mid-parse.
type reader struct {
	b   []byte
	err error
}

func (r *reader) fail(what string) {
	if r.err == nil {
		r.err = fmt.Errorf("wire: truncated or malformed %s", what)
	}
}

func (r *reader) flag() bool {
	if r.err != nil {
		return false
	}
	if len(r.b) < 1 {
		r.fail("present flag")
		return false
	}
	f := r.b[0]
	r.b = r.b[1:]
	return f != 0
}

func (r *reader) uvarint() uint64 {
	if r.err != nil {
		return 0
	}
	v, n := binary.Uvarint(r.b)
	if n <= 0 {
		r.fail("uvarint")
		return 0
	}
	r.b = r.b[n:]
	return v
}

func (r *reader) varint() int64 {
	if r.err != nil {
		return 0
	}
	v, n := binary.Varint(r.b)
	if n <= 0 {
		r.fail("varint")
		return 0
	}
	r.b = r.b[n:]
	return v
}

func (r *reader) string() string {
	if r.err != nil {
		return ""
	}
	n := r.uvarint()
	if r.err != nil {
		return ""
	}
	if uint64(len(r.b)) < n {
		r.fail("string body")
		return ""
	}
	s := string(r.b[:n])
	r.b = r.b[n:]
	return s
}

func (r *reader) stringSlice() []string {
	if !r.flag() {
		return nil
	}
	n := int(r.uvarint())
	if r.err != nil {
		return nil
	}
	ss := make([]string, n)
	for i := 0; i < n; i++ {
		ss[i] = r.string()
	}
	return ss
}

func (r *reader) u32() uint32 {
	if r.err != nil {
		return 0
	}
	if len(r.b) < 4 {
		r.fail("uint32")
		return 0
	}
	v := binary.LittleEndian.Uint32(r.b)
	r.b = r.b[4:]
	return v
}

func (r *reader) f32() float32 {
	return math.Float32frombits(r.u32())
}

func (r *reader) f64() float64 {
	if r.err != nil {
		return 0
	}
	if len(r.b) < 8 {
		r.fail("float64")
		return 0
	}
	v := binary.LittleEndian.Uint64(r.b)
	r.b = r.b[8:]
	return math.Float64frombits(v)
}
