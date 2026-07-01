// Package dense implements the query-side dense encoder doc 10 pins: the component that
// turns a query's analyzed terms into the dense-plane query vector the ANN index recalls
// against. Doc 10 names two encoders chosen by budget, a small int8 ONNX model for the
// real learned embedding and a static token-lookup embedder for tight budgets, behind one
// contract: terms in, a dense vector out, produced once per query at the broker.
//
// This package implements the static token-lookup path, the budget encoder, as a real
// working encoder and exposes the model path as the same interface a deployment plugs an
// ONNX runtime into. The static encoder is a precomputed table mapping each token to a
// vector with the query embedding the pooled mean of its terms' vectors, exactly the
// model doc 10 describes: microseconds, no transformer, bag-of-words quality with no
// contextual interaction between tokens. The build side embeds documents with the same
// encoder so a query and a document live in one space, the consistency the dense plane
// depends on, the dense-plane analog of the shared analyzer the lexical plane depends on.
package dense

import "math"

// Encoder turns a query's analyzed terms into a dense query vector at the kept dimension,
// the vector the broker ships to the shards as ParsedQuery.DenseVec. It is the one seam
// the query pipeline consumes; the static encoder below satisfies it, and an ONNX-backed
// encoder a deployment supplies satisfies it the same way. A nil or empty term list, or
// terms the table does not know, yield a zero vector, which the dense plane reads as no
// signal rather than a spurious neighbor.
type Encoder interface {
	// Encode returns the kept-dimension query vector for the analyzed terms.
	Encode(terms []string) []float32
	// Dim is the kept dimension of the vectors the encoder produces, the dimension the
	// shard's vectors were truncated to so a query and a document compare coordinate for
	// coordinate.
	Dim() int
}

// Table maps a token to its embedding, the lookup the static encoder pools over. A
// deployment with a real per-token embedding table derived from its model implements this
// and passes it to NewStatic; the package ships HashTable as the dependency-free default
// so a process has a working encoder with no embedding file to load.
type Table interface {
	// Lookup returns the token's vector, or nil for a token the table does not carry. A
	// returned vector must be exactly Dim long.
	Lookup(token string) []float32
	// Dim is the embedding dimension every vector the table returns has.
	Dim() int
}

// StaticEncoder is the static token-lookup embedder doc 10 specifies for tight budgets.
// The query embedding is the L2-normalized mean of its terms' table vectors, the standard
// pooled bag-of-words embedding: cheap, deterministic, and in the same space as the
// documents when they are embedded with the same table. It is immutable after
// construction and safe for concurrent use, so one instance serves every query.
type StaticEncoder struct {
	table Table
}

// NewStatic builds a static encoder over a token-embedding table.
func NewStatic(table Table) *StaticEncoder {
	return &StaticEncoder{table: table}
}

// DefaultNonzero and DefaultSeed are the random-indexing hash table parameters the
// default static encoder uses: how many nonzero entries each token's sparse vector
// carries, and the seed that fixes the table. They live here, not at each call site, so
// the build side that embeds documents and the query side that embeds queries construct
// the same table by construction and their vectors land in one comparable space. Changing
// either changes the space, so a collection built at one setting must be queried at the
// same one; pinning them as package defaults is what keeps the two sides from drifting.
const (
	DefaultNonzero = 8
	DefaultSeed    = 1
)

// NewDefault builds the default static encoder at the given kept dimension: a static
// token-lookup embedder over a random-indexing HashTable seeded with the package
// defaults. It is the one constructor both the collection build (document vectors) and
// the query pipeline (query vectors) call, so a document and a query at the same
// dimension are embedded by the identical table and their cosine is meaningful. The
// dimension is chosen per collection, so it is a parameter; the nonzero count and seed
// are fixed defaults so the two sides cannot disagree on anything but the dimension the
// shards already record.
func NewDefault(dim int) *StaticEncoder {
	return NewStatic(NewHashTable(dim, DefaultNonzero, DefaultSeed))
}

// Dim is the kept dimension, the table's embedding dimension.
func (e *StaticEncoder) Dim() int {
	if e == nil || e.table == nil {
		return 0
	}
	return e.table.Dim()
}

// Encode pools the terms' table vectors into one query vector. It sums the vector of
// every term the table knows, divides by the count of contributing terms to mean-pool,
// and L2-normalizes the result so it sits on the unit sphere the document vectors were
// normalized onto, the precondition for a meaningful cosine. A query with no known terms
// returns the zero vector, the dense plane's no-signal value.
func (e *StaticEncoder) Encode(terms []string) []float32 {
	if e == nil || e.table == nil {
		return nil
	}
	dim := e.table.Dim()
	acc := make([]float32, dim)
	contributing := 0
	for _, t := range terms {
		v := e.table.Lookup(t)
		if len(v) != dim {
			continue
		}
		for i, x := range v {
			acc[i] += x
		}
		contributing++
	}
	if contributing == 0 {
		return acc
	}
	inv := float32(1) / float32(contributing)
	for i := range acc {
		acc[i] *= inv
	}
	l2Normalize(acc)
	return acc
}

// l2Normalize scales v in place to unit length, leaving an all-zero vector untouched so a
// no-signal query stays the zero vector rather than dividing by zero.
func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}
