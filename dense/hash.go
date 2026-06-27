package dense

import (
	"sort"

	"github.com/cespare/xxhash/v2"
)

// HashTable is the dependency-free default Table: a deterministic token-embedding table
// built by random indexing rather than loaded from a model file. Random indexing is the
// classic way to get distributional vectors with no training: each token hashes to a
// small set of signed dimensions, so a token's vector is sparse and fixed, two tokens
// share a coordinate only by hash collision, and the whole table is a function of the
// token string with nothing to store. It gives the static encoder a real working vector
// space out of the box, the bag-of-words quality doc 10 ascribes to the static path, and
// it is the seam a deployment replaces with a real per-token embedding table from its
// model while keeping the encoder, the build side, and the wire format unchanged.
//
// The table is immutable and its Lookup is a pure function of the token, so one instance
// is safe across every query and the build side and the query side that share it embed
// identical tokens into identical vectors, the consistency the dense plane depends on.
type HashTable struct {
	dim     int
	nonzero int
	seed    uint64
}

// NewHashTable builds a random-indexing table of the given dimension. nonzero is how many
// signed coordinates each token's vector sets, the table's sparsity: a handful keeps
// distinct tokens near-orthogonal while a pooled query over several terms still fills
// enough of the space to compare. seed lets a deployment pin a private space; the same
// seed yields the same table.
func NewHashTable(dim, nonzero int, seed uint64) *HashTable {
	if dim < 1 {
		dim = 1
	}
	if nonzero < 1 {
		nonzero = 1
	}
	if nonzero > dim {
		nonzero = dim
	}
	return &HashTable{dim: dim, nonzero: nonzero, seed: seed}
}

// Dim is the table's embedding dimension.
func (t *HashTable) Dim() int { return t.dim }

// Lookup returns the token's fixed random-indexing vector. The token hashes to a base
// value; successive rehashes pick nonzero distinct coordinates, each set to +1 or -1 by a
// sign bit drawn from the same hash stream, and the vector is L2-normalized so every token
// is a unit vector and the encoder's mean-pool weights its terms equally. An empty token
// returns nil so the encoder skips it rather than pooling a spurious vector.
func (t *HashTable) Lookup(token string) []float32 {
	if token == "" {
		return nil
	}
	v := make([]float32, t.dim)
	h := xxhash.Sum64String(token) ^ t.seed
	set := 0
	// Walk a deterministic hash stream, taking distinct coordinates until nonzero are set.
	// Distinctness keeps a token's mass from piling onto one coordinate when two draws
	// collide, so the vector stays a clean sparse signature.
	for attempt := uint64(0); set < t.nonzero; attempt++ {
		hv := mix(h + attempt)
		idx := int(hv % uint64(t.dim))
		if v[idx] != 0 {
			continue
		}
		if hv&0x8000000000000000 != 0 {
			v[idx] = 1
		} else {
			v[idx] = -1
		}
		set++
	}
	l2Normalize(v)
	return v
}

// mix is a finalizer that scrambles a counter into a well-distributed 64-bit value, the
// splitmix64 finalizer, so consecutive attempts pick uncorrelated coordinates and signs.
func mix(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// Vocabulary is an optional Table built from an explicit token-to-vector map, the shape a
// deployment loads from a real embedding file. It is here so the build side and the query
// side can share an arbitrary fixed table through the same Table seam the HashTable
// satisfies, and so a test can pin exact vectors. Tokens absent from the map return nil,
// the encoder's skip signal.
type Vocabulary struct {
	dim     int
	vectors map[string][]float32
}

// NewVocabulary wraps a token-to-vector map. Vectors not exactly dim long are dropped so
// the table's Lookup contract, every returned vector is dim long, always holds.
func NewVocabulary(dim int, vectors map[string][]float32) *Vocabulary {
	clean := make(map[string][]float32, len(vectors))
	for tok, v := range vectors {
		if len(v) == dim {
			cp := make([]float32, dim)
			copy(cp, v)
			clean[tok] = cp
		}
	}
	return &Vocabulary{dim: dim, vectors: clean}
}

// Dim is the table's embedding dimension.
func (v *Vocabulary) Dim() int { return v.dim }

// Lookup returns a copy of the token's vector, or nil for an unknown token. It copies so a
// caller that mutates the result, as the encoder's pooling never does but a future one
// might, cannot corrupt the shared table.
func (v *Vocabulary) Lookup(token string) []float32 {
	stored, ok := v.vectors[token]
	if !ok {
		return nil
	}
	out := make([]float32, v.dim)
	copy(out, stored)
	return out
}

// Tokens returns the vocabulary's tokens in sorted order, a deterministic view a test or a
// dump uses without depending on map iteration order.
func (v *Vocabulary) Tokens() []string {
	toks := make([]string, 0, len(v.vectors))
	for tok := range v.vectors {
		toks = append(toks, tok)
	}
	sort.Strings(toks)
	return toks
}
