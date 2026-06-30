package tsumugi

import (
	"encoding/binary"
	"math"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// RegionDescriptor locates one region and records how to read it. A region is
// found through its descriptor and nothing else, so the physical order of
// regions in the file is not load bearing. The on-disk descriptor is 36 bytes:
//
//	kind u8, codec u8, flags u16, offset u64, length u64,
//	raw_length u64, crc u32, dict_id u32
type RegionDescriptor struct {
	Kind      RegionKind
	Codec     Codec
	Flags     uint16
	Offset    uint64 // byte offset of the region in the file
	Length    uint64 // bytes on disk (compressed length when a codec is set)
	RawLength uint64 // bytes uncompressed (equals Length when CodecNone)
	CRC       uint32 // CRC32C of the on-disk bytes, validated before decompression
	DictID    uint32 // dictionary sub-id this region trained against, or 0
}

const regionDescriptorSize = 36

// Field describes one column of the forward store, carried in the footer schema
// section. The full forward layout lands with M2; M0 records the schema so the
// container framing is stable.
type Field struct {
	Name     string
	Type     uint8 // logical type tag, defined by the forward package
	Codec    Codec
	Nullable bool
}

// Stats holds the shard-level numbers a broker reads before touching any region:
// document and token counts, average lengths, the node-id range, and the
// dequantization bounds for the feature columns. Values are float64 so one map
// carries both counts and real-valued bounds; integer counts up to 2^53 are
// exact. Well-known keys are defined as Stat* constants.
type Stats map[string]float64

// Well-known statistics keys.
const (
	StatDocCount   = "doc_count"
	StatTokenCount = "token_count"
	StatAvgDocLen  = "avg_doc_len"
	StatTermCount  = "term_count"
	// Per-field token-count sums the build records so a broker can derive the fleet
	// average length of each field, the per-field BM25F length normalizer. A single
	// average-doc-length conflates the fields; the title is short and the body long, so
	// normalizing both against one mean misnormalizes both. These let the broker compute
	// the fleet average title, body, and url length separately and feed each field's BM25
	// its own denominator, the cross-shard normalization the merged top-k rests on.
	StatTitleTokenCount = "title_token_count"
	StatBodyTokenCount  = "body_token_count"
	StatURLTokenCount   = "url_token_count"
	StatEdgeCount       = "edge_count"
	StatNodeMin         = "node_min"
	StatNodeMax         = "node_max"
	StatVectorDim       = "vector_dim"
	// StatAnalyzerHash records the hash of the analyzer the shard was built with, the
	// consistency guard a broker checks before it queries the shard. It is a full
	// 64-bit value carried through the float64 stats map by AnalyzerHashStat, which
	// preserves the bit pattern rather than the numeric value a plain stat would lose
	// above 2^53.
	StatAnalyzerHash = "analyzer_hash"
	// StatBuildConfigHash records a 64-bit digest of the configuration the shard was
	// built under: the container format version, the feature schema version, the
	// routing index version, the analyzer hash, the shard size, and the curated trust
	// and spam seed lists. Two shards built under the same configuration carry the same
	// digest, so a reader can tell at a glance whether two shards are configuration-
	// compatible and a reproducibility check can assert a rebuild used the same inputs.
	// Like the analyzer hash it is a full 64-bit value carried losslessly through the
	// float64 stats map by AnalyzerHashStat.
	StatBuildConfigHash = "build_config_hash"
)

// AnalyzerHashStat encodes a 64-bit analyzer hash as the float64 with the identical
// bit pattern, the lossless way to carry a full uint64 through the float64 stats map.
// The footer round-trips float64 bits exactly (it serializes math.Float64bits), so the
// hash survives a write and read cycle unchanged where a numeric stat would round off
// any value above 2^53.
func AnalyzerHashStat(h uint64) float64 { return math.Float64frombits(h) }

// AnalyzerHashFromStat is the inverse of AnalyzerHashStat, recovering the 64-bit hash
// from its stored float64 bit pattern.
func AnalyzerHashFromStat(v float64) uint64 { return math.Float64bits(v) }

// Footer is the directory written last: the schema, a descriptor for every
// region, and the shard statistics. A complete footer means a complete file.
type Footer struct {
	Schema  []Field
	Regions []RegionDescriptor
	Stats   Stats
}

// encode serializes the footer as a sequence of tagged, length-prefixed
// sections so a reader can skip a section it does not understand.
func (f *Footer) encode() []byte {
	var out []byte
	out = appendSection(out, sectionSchema, f.encodeSchema())
	out = appendSection(out, sectionRegions, f.encodeRegions())
	out = appendSection(out, sectionStats, f.encodeStats())
	return out
}

func appendSection(out []byte, tag uint8, payload []byte) []byte {
	out = append(out, tag)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(payload)))
	return append(out, payload...)
}

func (f *Footer) encodeSchema() []byte {
	var b []byte
	b = codec.AppendUvarint(b, uint64(len(f.Schema)))
	for _, fld := range f.Schema {
		b = codec.AppendUvarint(b, uint64(len(fld.Name)))
		b = append(b, fld.Name...)
		b = append(b, fld.Type, byte(fld.Codec))
		if fld.Nullable {
			b = append(b, 1)
		} else {
			b = append(b, 0)
		}
	}
	return b
}

func (f *Footer) encodeRegions() []byte {
	b := make([]byte, 0, len(f.Regions)*regionDescriptorSize)
	for _, r := range f.Regions {
		var d [regionDescriptorSize]byte
		d[0] = byte(r.Kind)
		d[1] = byte(r.Codec)
		binary.LittleEndian.PutUint16(d[2:4], r.Flags)
		binary.LittleEndian.PutUint64(d[4:12], r.Offset)
		binary.LittleEndian.PutUint64(d[12:20], r.Length)
		binary.LittleEndian.PutUint64(d[20:28], r.RawLength)
		binary.LittleEndian.PutUint32(d[28:32], r.CRC)
		binary.LittleEndian.PutUint32(d[32:36], r.DictID)
		b = append(b, d[:]...)
	}
	return b
}

func (f *Footer) encodeStats() []byte {
	// Sort keys so the footer bytes are deterministic for a given build, which
	// keeps shard builds reproducible.
	keys := make([]string, 0, len(f.Stats))
	for k := range f.Stats {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b []byte
	b = codec.AppendUvarint(b, uint64(len(keys)))
	for _, k := range keys {
		b = codec.AppendUvarint(b, uint64(len(k)))
		b = append(b, k...)
		b = binary.LittleEndian.AppendUint64(b, math.Float64bits(f.Stats[k]))
	}
	return b
}

// decodeFooter parses the footer bytes. It tolerates unknown section tags by
// skipping them via their length prefix, which is how the format grows.
func decodeFooter(b []byte) (Footer, error) {
	var f Footer
	f.Stats = Stats{}
	for len(b) > 0 {
		if len(b) < 5 {
			return f, ErrCorruptFooter
		}
		tag := b[0]
		n := binary.LittleEndian.Uint32(b[1:5])
		b = b[5:]
		if uint64(len(b)) < uint64(n) {
			return f, ErrCorruptFooter
		}
		payload := b[:n]
		b = b[n:]
		switch tag {
		case sectionSchema:
			if err := f.decodeSchema(payload); err != nil {
				return f, err
			}
		case sectionRegions:
			if err := f.decodeRegions(payload); err != nil {
				return f, err
			}
		case sectionStats:
			if err := f.decodeStats(payload); err != nil {
				return f, err
			}
		default:
			// Unknown section: skip it. This is the forward-compatible path.
		}
	}
	return f, nil
}

func (f *Footer) decodeSchema(b []byte) error {
	n, k := codec.Uvarint(b)
	if k <= 0 {
		return ErrCorruptFooter
	}
	b = b[k:]
	f.Schema = make([]Field, 0, n)
	for i := uint64(0); i < n; i++ {
		nl, k := codec.Uvarint(b)
		if k <= 0 || uint64(len(b)-k) < nl {
			return ErrCorruptFooter
		}
		b = b[k:]
		name := string(b[:nl])
		b = b[nl:]
		if len(b) < 3 {
			return ErrCorruptFooter
		}
		fld := Field{Name: name, Type: b[0], Codec: Codec(b[1]), Nullable: b[2] != 0}
		b = b[3:]
		f.Schema = append(f.Schema, fld)
	}
	return nil
}

func (f *Footer) decodeRegions(b []byte) error {
	if len(b)%regionDescriptorSize != 0 {
		return ErrCorruptFooter
	}
	count := len(b) / regionDescriptorSize
	f.Regions = make([]RegionDescriptor, count)
	for i := 0; i < count; i++ {
		d := b[i*regionDescriptorSize : (i+1)*regionDescriptorSize]
		f.Regions[i] = RegionDescriptor{
			Kind:      RegionKind(d[0]),
			Codec:     Codec(d[1]),
			Flags:     binary.LittleEndian.Uint16(d[2:4]),
			Offset:    binary.LittleEndian.Uint64(d[4:12]),
			Length:    binary.LittleEndian.Uint64(d[12:20]),
			RawLength: binary.LittleEndian.Uint64(d[20:28]),
			CRC:       binary.LittleEndian.Uint32(d[28:32]),
			DictID:    binary.LittleEndian.Uint32(d[32:36]),
		}
	}
	return nil
}

func (f *Footer) decodeStats(b []byte) error {
	n, k := codec.Uvarint(b)
	if k <= 0 {
		return ErrCorruptFooter
	}
	b = b[k:]
	for i := uint64(0); i < n; i++ {
		kl, k := codec.Uvarint(b)
		if k <= 0 || uint64(len(b)-k) < kl {
			return ErrCorruptFooter
		}
		b = b[k:]
		key := string(b[:kl])
		b = b[kl:]
		if len(b) < 8 {
			return ErrCorruptFooter
		}
		f.Stats[key] = math.Float64frombits(binary.LittleEndian.Uint64(b[:8]))
		b = b[8:]
	}
	return nil
}

// region returns the descriptor for a kind, or false if absent.
func (f *Footer) region(kind RegionKind) (RegionDescriptor, bool) {
	for _, r := range f.Regions {
		if r.Kind == kind {
			return r, true
		}
	}
	return RegionDescriptor{}, false
}
