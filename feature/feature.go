// Package feature implements the .tsumugi feature region: a row-major,
// fixed-width matrix of query-independent ranking signals, one row per dense
// docID. Every value is quantized to one or two bytes, so a row is a handful of
// bytes reached by pure address arithmetic, region_base + docID*stride, with no
// per-row indirection. This is the layout the L1 linear scorer walks in its tight
// loop and the L2 reranker reads in full.
//
// A signal lands in this region exactly when it does not depend on the query:
// PageRank, host and domain authority, trust, spam mass, link counts, freshness,
// content quality, the field lengths, and the composite static rank. The
// query-dependent signals, BM25F and the dense cosine, are computed at query time
// and never stored here. The bytes are the region's FEA1 format, framed by the M0
// container as RegionFeature.
package feature

import (
	"github.com/cespare/xxhash/v2"

	"github.com/tamnd/tsumugi/codec"
)

const regionMagic = "FEA1"

const regionVersion = 1

// SchemaVersion is the version of the canonical feature schema this build writes
// and reads. It is the number the build stamps into every feature region and the
// model artifact records, distinct from regionVersion, which is the FEA1 byte
// format. A change to DefaultSchema that reorders, adds, or retypes a column bumps
// this so a shard or model built against the old layout is refused at load rather
// than read with its columns silently misaligned.
const SchemaVersion uint16 = 1

// FeatureID is the stable identity of a signal, constant across schema versions
// so a model trained against one schema can find its columns in a later one. A
// minor schema bump appends new ids; it never renumbers an existing one.
type FeatureID uint8

const (
	FeatStaticRank     FeatureID = 0 // composite, also drives posting order
	FeatPageRank       FeatureID = 1
	FeatHostRank       FeatureID = 2
	FeatDomainRank     FeatureID = 3
	FeatTrust          FeatureID = 4
	FeatSpamMass       FeatureID = 5
	FeatInDegree       FeatureID = 6 // two-byte log column
	FeatLinkingDomains FeatureID = 7
	FeatFreshness      FeatureID = 8
	FeatChangeRate     FeatureID = 9
	FeatContentQuality FeatureID = 10
	FeatBoilerplate    FeatureID = 11
	FeatNearDup        FeatureID = 12
	FeatDocLen         FeatureID = 13
	FeatTitleLen       FeatureID = 14
	FeatBodyLen        FeatureID = 15
	FeatURLFieldLen    FeatureID = 16
	FeatAnchorFieldLen FeatureID = 17
	FeatLanguage       FeatureID = 18
	FeatURLDepth       FeatureID = 19
	FeatURLLen         FeatureID = 20
	FeatHTTPS          FeatureID = 21
	FeatHostErrorRate  FeatureID = 22
)

// Quant is a column's quantization scheme. Linear suits bounded, roughly uniform
// signals; log suits heavy-tailed ones like PageRank and the lengths; signed
// suits zero-centered differences.
type Quant uint8

const (
	QuantLinear Quant = 0
	QuantLog    Quant = 1
	QuantSigned Quant = 2
)

// epsLog keeps log quantization defined at zero, since log(0) is undefined and
// many link signals are zero for most documents.
const epsLog = 1.0

// Column declares one feature column: which signal it holds, how wide its
// quantized value is, and how it is quantized. The byte offset within a row is
// derived at build time from the column order.
type Column struct {
	ID    FeatureID
	Width uint8 // 1 or 2 bytes
	Quant Quant
}

// DefaultSchema is the canonical M3 feature set: the link signals and lengths are
// log-quantized, in-degree gets two bytes because its flat middle has many
// distinct values, and the bounded quality and ratio signals are linear.
func DefaultSchema() []Column {
	return []Column{
		{FeatStaticRank, 1, QuantLinear},
		{FeatPageRank, 1, QuantLog},
		{FeatHostRank, 1, QuantLog},
		{FeatDomainRank, 1, QuantLog},
		{FeatTrust, 1, QuantLog},
		{FeatSpamMass, 1, QuantLinear},
		{FeatInDegree, 2, QuantLog},
		{FeatLinkingDomains, 1, QuantLog},
		{FeatFreshness, 1, QuantLinear},
		{FeatChangeRate, 1, QuantLinear},
		{FeatContentQuality, 1, QuantLinear},
		{FeatBoilerplate, 1, QuantLinear},
		{FeatNearDup, 1, QuantLinear},
		{FeatDocLen, 1, QuantLog},
		{FeatTitleLen, 1, QuantLinear},
		{FeatBodyLen, 1, QuantLog},
		{FeatURLFieldLen, 1, QuantLinear},
		{FeatAnchorFieldLen, 1, QuantLinear},
		{FeatLanguage, 1, QuantLinear},
		{FeatURLDepth, 1, QuantLinear},
		{FeatURLLen, 1, QuantLinear},
		{FeatHTTPS, 1, QuantLinear},
		{FeatHostErrorRate, 1, QuantLinear},
	}
}

// SchemaHash is a stable 64-bit fingerprint of a column layout: the column count
// followed by each column's running byte offset, feature id, width, and quant. Two
// schemas hash equal exactly when they place the same signals in the same order with
// the same widths and quantization, so a model trained against one schema and a
// shard built against another are caught even when both carry the same version
// number. The running offset is folded in so a width change that shifts later
// columns is caught even if no id, width, or quant byte differs at a given index.
func SchemaHash(cols []Column) uint64 {
	var h xxhash.Digest
	var hdr [2]byte
	codec.PutUint16(hdr[:], uint16(len(cols)))
	_, _ = h.Write(hdr[:])
	var off uint16
	for _, c := range cols {
		var b [5]byte
		codec.PutUint16(b[0:], off)
		b[2] = byte(c.ID)
		b[3] = c.Width
		b[4] = byte(c.Quant)
		_, _ = h.Write(b[:])
		off += uint16(c.Width)
	}
	return h.Sum64()
}

// DefaultSchemaHash is the SchemaHash of DefaultSchema, the fingerprint the build
// stamps and the loader checks against.
func DefaultSchemaHash() uint64 { return SchemaHash(DefaultSchema()) }

// colLayout is a column plus where it sits in a row and the dequant params the
// build computed for it.
type colLayout struct {
	Column
	offset uint16
	p0     float32
	p1     float32
	p2     float32
}

// maxLevel is the largest quantized value a column of the given width can hold.
func maxLevel(width uint8) float64 {
	if width == 2 {
		return 65535
	}
	return 255
}

// loadQuant reads a column's raw quantized value from a row.
func loadQuant(row []byte, offset uint16, width uint8) uint32 {
	if width == 2 {
		return uint32(codec.Uint16(row[offset:]))
	}
	return uint32(row[offset])
}
