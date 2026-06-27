package vector

import (
	"fmt"
	"math"

	"github.com/tamnd/tsumugi/codec"
)

// defaultSeed is a fixed golden-ratio constant for the rotation and the layer
// assignment, so a build is reproducible.
const defaultSeed int64 = 0x1e3779b97f4a7c15

// Builder accumulates document embeddings and encodes them into a VEC1 region.
// Vectors arrive already truncated to the kept dimension (the Matryoshka cut is a
// model-side choice); the builder rotates, quantizes to one-bit and int8, builds
// the HNSW graph over the int8 dot, and frames the three parts.
type Builder struct {
	dim            int
	seed           int64
	m, m0          int
	efConstruction int
	normalized     bool
	rerank         bool
	vecs           [][]float32
}

// NewBuilder returns a builder for vectors of the given kept dimension.
func NewBuilder(dim int) *Builder {
	return &Builder{
		dim:            dim,
		seed:           defaultSeed,
		m:              DefaultM,
		m0:             DefaultM0,
		efConstruction: DefaultEfConstruction,
		normalized:     true,
		rerank:         true,
	}
}

// WithSeed sets the rotation and layer-assignment seed.
func (b *Builder) WithSeed(seed int64) *Builder { b.seed = seed; return b }

// WithHNSW overrides the graph parameters.
func (b *Builder) WithHNSW(m, m0, efConstruction int) *Builder {
	b.m, b.m0, b.efConstruction = m, m0, efConstruction
	return b
}

// WithRerank toggles the int8 rerank copy. With it off the region is the
// no-rerank one-bit form and the search scores with the RaBitQ estimator.
func (b *Builder) WithRerank(on bool) *Builder { b.rerank = on; return b }

// WithNormalized declares whether the input vectors are unit norm. The default is
// true, the common case for cosine-trained embeddings, and it lets the region drop
// the per-code norm field entirely (the norm is then a constant one). Pass false for
// vectors of varying magnitude, and the region stores each code's float16 norm so the
// estimator can scale by it.
func (b *Builder) WithNormalized(on bool) *Builder { b.normalized = on; return b }

// Add records a document's embedding. The dense docID is the call order.
func (b *Builder) Add(vec []float32) {
	cp := make([]float32, b.dim)
	copy(cp, vec)
	b.vecs = append(b.vecs, cp)
}

// Build rotates, quantizes, indexes, and frames the region. It returns an error
// only if the graph cannot be made fully reachable from the entry point, which the
// orphan repair below prevents on any normal corpus.
func (b *Builder) Build() ([]byte, error) {
	rot := newRotator(b.dim, b.seed)
	rdim := rot.rdim
	n := len(b.vecs)

	rotated := make([][]float32, n)
	codes := make([]oneBitCode, n)
	var maxAbs float64
	for i, v := range b.vecs {
		oRot := rot.rotate(v)
		rotated[i] = oRot
		codes[i] = encodeOneBit(oRot)
		for _, x := range oRot {
			if a := math.Abs(float64(x)); a > maxAbs {
				maxAbs = a
			}
		}
	}
	i8scale := float32(maxAbs / 127)
	if i8scale == 0 {
		i8scale = 1
	}
	iq := newInt8Quant(i8scale)

	words := rdim / 64
	// Each code row is a float16 scalar (2 bytes), an optional float16 norm (2 bytes,
	// dropped when the vectors are normalized because the norm is then a constant one),
	// then the words sign blocks. The sign words are read byte-wise (codec.Uint64), so
	// the row needs no eight-byte alignment and the half-width scalars are pure savings
	// against the four-byte float32 the budget would otherwise carry.
	codeHdr := 2
	if !b.normalized {
		codeHdr = 4
	}
	codeStride := codeHdr + words*8
	rerankStride := rdim // power of two, already 8-aligned

	codesPart := make([]byte, 0, n*codeStride)
	for i := 0; i < n; i++ {
		codesPart = codec.AppendUint16(codesPart, codec.Float16bits(codes[i].scalar))
		if !b.normalized {
			codesPart = codec.AppendUint16(codesPart, codec.Float16bits(codes[i].norm))
		}
		for _, w := range codes[i].bits {
			codesPart = codec.AppendUint64(codesPart, w)
		}
	}

	// The int8 rows are the build distance for the graph regardless of mode; they
	// are only written to the region when the rerank copy is kept.
	rows := make([][]int8, n)
	for i := 0; i < n; i++ {
		rows[i] = iq.encode(rotated[i])
	}
	var rerankPart []byte
	if b.rerank {
		rerankPart = make([]byte, 0, n*rerankStride)
		for i := 0; i < n; i++ {
			for _, c := range rows[i] {
				rerankPart = append(rerankPart, byte(c))
			}
		}
	}

	g := newHNSW(rows, b.m, b.m0, b.efConstruction, b.seed)
	// Guarantee the connectivity invariant the search relies on: every node must be
	// reachable from the entry, or the document it holds is invisible to dense search.
	// repair grafts orphans back in and a single pass reconnects the graph unless an
	// eviction re-orphans a node, so loop until the reachable count stops rising.
	prev := -1
	for {
		g.repair()
		reached := g.reachableCount()
		if reached == n {
			break
		}
		if reached <= prev {
			return nil, fmt.Errorf("%w: %d of %d nodes reachable", ErrUnreachable, reached, n)
		}
		prev = reached
	}
	linksPart := serializeLinks(g)

	flags := uint32(0)
	if b.normalized {
		flags |= flagNormalized
	}
	if b.rerank {
		flags |= flagHasRerank
	}
	entry := uint32(0)
	if g.entry >= 0 {
		entry = uint32(g.entry)
	}

	h := header{
		version:        regionVersion,
		flags:          flags,
		dimKept:        uint32(b.dim),
		rdim:           uint32(rdim),
		count:          uint32(n),
		m:              uint32(b.m),
		m0:             uint32(b.m0),
		entryPoint:     entry,
		maxLayer:       uint32(g.maxLayer),
		efConstruction: uint32(b.efConstruction),
		rotationSeed:   b.seed,
		i8scale:        i8scale,
		codeStride:     uint32(codeStride),
		rerankStride:   uint32(rerankStride),
		codesLen:       uint64(len(codesPart)),
		rerankLen:      uint64(len(rerankPart)),
		linksLen:       uint64(len(linksPart)),
	}
	region := h.encode()
	region = append(region, codesPart...)
	region = append(region, rerankPart...)
	region = append(region, linksPart...)
	return region, nil
}
