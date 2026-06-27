// Package tsumugi implements the .tsumugi single-file shard: one self-describing
// container holding the inverted index, stored fields, the quantized feature
// matrix, the compressed link graph, and the quantized vectors for one slice of
// a corpus. The byte layout is pinned in spec 2067 doc 03; this package owns the
// container framing (header, region descriptors, footer, trailer, CRC, mmap),
// and the per-region internals live in the lexical, forward, feature, graph, and
// vector subpackages.
//
// A shard is written append-then-footer: every region's bytes land first, then
// the footer that points at them, then the trailer, and the header is rewritten
// last once the footer offset is known. A file with a valid trailing magic and a
// footer that passes its CRC is complete; anything else is a torn write and is
// rejected at open. This is the same single-write-moment discipline tatami's
// TAT1 container uses, with TSM1's extra region kinds.
package tsumugi

// Magic marks both ends of a shard. It is distinct from tatami's TAT1 so a tool
// never confuses the two formats.
const (
	Magic        = "TSM1"
	VersionMajor = 1
	VersionMinor = 0

	// HeaderSize is the fixed header length in bytes. Only the header (at offset
	// zero) and the trailer (at the end) sit at known offsets; everything else
	// is reached through the footer.
	HeaderSize = 64

	// TrailerSize is the fixed trailer length: footer_length (8) + footer_crc
	// (4) + magic (4).
	TrailerSize = 16
)

// Capability flag bits live in the header's flags word so a reader can tell what
// a shard contains from 64 bytes without parsing the footer.
const (
	FlagHasLexical     uint64 = 1 << 0 // an inverted index is present
	FlagHasForward     uint64 = 1 << 1 // stored fields are present
	FlagHasFeature     uint64 = 1 << 2 // a feature matrix is present
	FlagHasGraph       uint64 = 1 << 3 // a link graph is present
	FlagHasVector      uint64 = 1 << 4 // dense vectors are present
	FlagHasDictionary  uint64 = 1 << 5 // shared zstd dictionaries are present
	FlagSearchOnly     uint64 = 1 << 6 // the forward region dropped the body, snippet only
	FlagImpactPostings uint64 = 1 << 7 // postings carry learned-sparse impact weights, not tf
)

// RegionKind identifies a region in its footer descriptor. Regions are reached
// only through descriptors, so their physical order in the file is not load
// bearing and a new kind can be added without breaking an old reader.
type RegionKind uint8

const (
	RegionLexical    RegionKind = 1
	RegionForward    RegionKind = 2
	RegionFeature    RegionKind = 3
	RegionGraph      RegionKind = 4
	RegionVector     RegionKind = 5
	RegionDictionary RegionKind = 6
)

// String renders a region kind for inspect output.
func (k RegionKind) String() string {
	switch k {
	case RegionLexical:
		return "lexical"
	case RegionForward:
		return "forward"
	case RegionFeature:
		return "feature"
	case RegionGraph:
		return "graph"
	case RegionVector:
		return "vector"
	case RegionDictionary:
		return "dictionary"
	default:
		return "unknown"
	}
}

// Codec identifies how a region's bytes are stored on disk.
type Codec uint8

const (
	CodecNone     Codec = 0 // stored as is
	CodecZstd     Codec = 1 // zstd compressed, no shared dictionary
	CodecZstdDict Codec = 2 // zstd compressed against a shared trained dictionary
)

// String renders a codec for inspect output.
func (c Codec) String() string {
	switch c {
	case CodecNone:
		return "none"
	case CodecZstd:
		return "zstd"
	case CodecZstdDict:
		return "zstd+dict"
	default:
		return "unknown"
	}
}

// Footer section tags. The footer is a sequence of tagged, length-prefixed
// sections so a reader skips a section it does not understand.
const (
	sectionSchema  uint8 = 1
	sectionRegions uint8 = 2
	sectionStats   uint8 = 3
)
