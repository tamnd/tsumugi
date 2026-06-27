package lexical

import "math"

// scoreScale is the fixed-point scale the traversal scores in. BM25F produces a
// real contribution per term-document; the engine multiplies it by this scale
// and rounds to an int32 so retrieval works in an integer domain. Integer
// addition is associative, so the pruned traversal and the exhaustive oracle sum
// the identical per-posting integers in any order and reach the identical total,
// which is what makes the oracle equality exact rather than float-fuzzy.
const scoreScale = 1 << 16

// Params are the BM25F tuning knobs. They are scoring parameters, not index
// structure, so a deployment can override them per query without a rebuild.
type Params struct {
	K1     float64            // term-frequency saturation
	Weight [numFields]float64 // per-field weight w_f
	B      [numFields]float64 // per-field length normalization b_f
}

// DefaultParams are sensible BM25F defaults: the title and anchor fields weigh
// more than the body, url-tokens least, with moderate length normalization.
func DefaultParams() Params {
	return Params{
		K1:     0.9,
		Weight: [numFields]float64{FieldTitle: 3.0, FieldBody: 1.0, FieldURL: 1.0, FieldAnchor: 2.0},
		B:      [numFields]float64{FieldTitle: 0.4, FieldBody: 0.75, FieldURL: 0.4, FieldAnchor: 0.5},
	}
}

// stats are the shard-wide constants BM25F needs, read once at shard open. The
// per-field average lengths and the document count are baked at build time; the
// scorer takes them and the per-query params.
type stats struct {
	docCount    uint32
	avgFieldLen [numFields]float64
}

// IDF is the Robertson-Sparck-Jones inverse document frequency for a term with
// document frequency df in a collection of n documents, always positive because of
// the +1 inside the log. It is exported so the broker can compute the collection-wide
// idf from the fleet's total document count and the term's df summed across shards,
// then push that one value down to every shard. Scoring a term against the same n and
// df everywhere is what makes the merged top-k the result a single index over the
// whole collection would give, rather than one biased by how the term happens to be
// distributed across shards.
func IDF(n, df uint64) float64 {
	return math.Log(1 + (float64(n)-float64(df)+0.5)/(float64(df)+0.5))
}

// idf is the shard-local idf, the value the build bakes nothing of and the
// single-shard path computes from the region's own document count.
func idf(n uint32, df uint32) float64 {
	return IDF(uint64(n), uint64(df))
}

// scaleBound applies a term's idf to an idf-free integer bound, rounding up so the
// result stays a true upper bound on any document's integer score. The block-max and
// list-max bounds are stored idf-free since M13, so the same posting lists can be
// scored against a shard-local idf or a pushed-down collection-wide idf; the cursor
// scales the stored bound by whichever idf the query carries. Rounding up is what
// keeps the scaled bound at or above the rounded per-document score, so a skip
// decision never drops a document the exhaustive scan would have kept.
func scaleBound(termIDF float64, bound int32) int32 {
	return int32(math.Ceil(termIDF * float64(bound)))
}

// contribution is bm25f(t, d): the field-weighted, length-normalized,
// saturation-capped score a single term contributes to one document. fieldTF
// holds the term's frequency in each field for this document (zero where the
// term is absent), and fieldLen holds this document's length in each field.
func contribution(termIDF float64, fieldTF *[numFields]uint32, fieldLen *[numFields]uint32, st *stats, p *Params) float64 {
	var tfF float64
	for f := 0; f < numFields; f++ {
		tf := float64(fieldTF[f])
		if tf == 0 {
			continue
		}
		avg := st.avgFieldLen[f]
		var norm float64 = 1
		if avg > 0 {
			norm = 1 - p.B[f] + p.B[f]*float64(fieldLen[f])/avg
		}
		tfF += p.Weight[f] * tf / norm
	}
	if tfF == 0 {
		return 0
	}
	return termIDF * tfF / (p.K1 + tfF)
}

// quantize maps a real contribution to the integer score domain, rounding to
// nearest. A block-max upper bound rounds up instead, via quantizeCeil, so the
// bound never under-reports a contribution.
func quantize(contrib float64) int32 {
	return int32(math.Round(contrib * scoreScale))
}

// quantizeCeil rounds a contribution up, used for upper bounds so a skip
// decision never drops a document whose true score the bound under-counted.
func quantizeCeil(contrib float64) int32 {
	return int32(math.Ceil(contrib * scoreScale))
}
