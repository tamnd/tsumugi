package tsumugi

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi/codec"
)

// Writer assembles a shard append-then-footer. Regions are written in the order
// AddRegion is called, each one's offset, length, and CRC recorded as it lands.
// Close writes the footer and trailer, rewrites the header now that the footer
// offset is known, fsyncs, and atomically renames the temp file into place, so
// the final path only ever appears on a complete, durable shard.
type Writer struct {
	finalPath string
	tmpPath   string
	f         *os.File
	w         *bufio.Writer
	off       uint64 // current write offset, also the next region's offset

	header  Header
	regions []RegionDescriptor
	schema  []Field
	stats   Stats
	enc     *zstd.Encoder
	closed  bool

	// Shared zstd dictionaries registered with AddDictionary, framed into the
	// RegionDictionary region on Close. dictByID holds each id's content for the
	// CodecZstdDict path; encDict caches a per-dictionary encoder.
	dicts    []dictEntry
	dictByID map[uint32][]byte
	encDict  map[uint32]*zstd.Encoder
}

// AddDictionary registers a shared zstd dictionary under a non-zero id. A region
// written with CodecZstdDict names a registered id in its descriptor, and the
// dictionaries are framed into the RegionDictionary region on Close so the reader
// recovers them. The id must be non-zero (0 means no dictionary) and unique, and
// the content is copied so the caller may reuse its buffer.
func (w *Writer) AddDictionary(id uint32, content []byte) error {
	if w.closed {
		return fmt.Errorf("tsumugi: AddDictionary after Close")
	}
	if id == 0 {
		return fmt.Errorf("tsumugi: dictionary id must be non-zero")
	}
	if len(content) == 0 {
		return fmt.Errorf("tsumugi: dictionary content is empty")
	}
	if _, dup := w.dictByID[id]; dup {
		return fmt.Errorf("tsumugi: dictionary id %d already registered", id)
	}
	cp := make([]byte, len(content))
	copy(cp, content)
	if w.dictByID == nil {
		w.dictByID = map[uint32][]byte{}
	}
	w.dictByID[id] = cp
	w.dicts = append(w.dicts, dictEntry{id: id, content: cp})
	return nil
}

// Create opens a new shard for writing at path. The shard is built in a sibling
// temp file and renamed into place on Close.
func Create(path string) (*Writer, error) {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	w := &Writer{
		finalPath: path,
		tmpPath:   tmp,
		f:         f,
		w:         bufio.NewWriterSize(f, 1<<20),
		stats:     Stats{},
		header:    Header{VersionMajor: VersionMajor, VersionMinor: VersionMinor},
	}
	// Reserve the header: write 64 placeholder bytes now, rewrite them on Close.
	if _, err := w.w.Write(make([]byte, HeaderSize)); err != nil {
		w.abort()
		return nil, err
	}
	w.off = HeaderSize
	return w, nil
}

// SetFlags sets the header capability flags (FlagHas*, FlagSearchOnly, ...).
func (w *Writer) SetFlags(flags uint64) { w.header.Flags |= flags }

// SetDocCount records N, the dense docID space size.
func (w *Writer) SetDocCount(n uint32) { w.header.DocCount = n }

// SetNodeBase records the global node id of dense docID 0 when the dense space
// maps to a contiguous run of node ids; zero otherwise.
func (w *Writer) SetNodeBase(base uint64) { w.header.NodeBase = base }

// SetBuildEpoch records the build timestamp in seconds. It is passed in, never
// read from a clock, so a build is reproducible.
func (w *Writer) SetBuildEpoch(epoch uint64) { w.header.BuildEpoch = epoch }

// SetSchema records the forward-store column schema in the footer.
func (w *Writer) SetSchema(fields []Field) { w.schema = fields }

// SetStat records a shard-level statistic.
func (w *Writer) SetStat(key string, value float64) { w.stats[key] = value }

// SetAnalyzerHash records the analyzer_hash, the hash of the analyzer the shard was
// built with, so a broker can refuse to query the shard with an incompatible analyzer.
// The hash is stored losslessly as a reinterpreted float64 because it is a full 64-bit
// value the numeric stats map would otherwise truncate.
func (w *Writer) SetAnalyzerHash(h uint64) { w.stats[StatAnalyzerHash] = AnalyzerHashStat(h) }

// AddRegion appends a region built from raw. When c is CodecZstd the bytes are
// compressed before they land; the descriptor records the on-disk length, the
// raw length, and the CRC of the on-disk bytes. AddRegion also sets the matching
// capability flag for the well-known region kinds.
func (w *Writer) AddRegion(kind RegionKind, c Codec, flags uint16, dictID uint32, raw []byte) error {
	if w.closed {
		return fmt.Errorf("tsumugi: AddRegion after Close")
	}
	onDisk := raw
	switch c {
	case CodecNone:
	case CodecZstd:
		if w.enc == nil {
			enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
			if err != nil {
				return err
			}
			w.enc = enc
		}
		onDisk = w.enc.EncodeAll(raw, make([]byte, 0, len(raw)/2+64))
	case CodecZstdDict:
		content, ok := w.dictByID[dictID]
		if !ok {
			return fmt.Errorf("tsumugi: codec zstd+dict needs a registered dictionary, id %d unknown", dictID)
		}
		enc := w.encDict[dictID]
		if enc == nil {
			e, err := zstd.NewWriter(nil,
				zstd.WithEncoderLevel(zstd.SpeedBetterCompression),
				zstd.WithEncoderDictRaw(dictID, content))
			if err != nil {
				return err
			}
			if w.encDict == nil {
				w.encDict = map[uint32]*zstd.Encoder{}
			}
			w.encDict[dictID] = e
			enc = e
		}
		onDisk = enc.EncodeAll(raw, make([]byte, 0, len(raw)/2+64))
	default:
		return fmt.Errorf("tsumugi: codec %d not supported by AddRegion", c)
	}
	desc := RegionDescriptor{
		Kind:      kind,
		Codec:     c,
		Flags:     flags,
		Offset:    w.off,
		Length:    uint64(len(onDisk)),
		RawLength: uint64(len(raw)),
		CRC:       codec.CRC32C(onDisk),
		DictID:    dictID,
	}
	if _, err := w.w.Write(onDisk); err != nil {
		return err
	}
	w.off += uint64(len(onDisk))
	w.regions = append(w.regions, desc)
	w.header.Flags |= flagForKind(kind)
	return nil
}

func flagForKind(kind RegionKind) uint64 {
	switch kind {
	case RegionLexical:
		return FlagHasLexical
	case RegionForward:
		return FlagHasForward
	case RegionFeature:
		return FlagHasFeature
	case RegionGraph:
		return FlagHasGraph
	case RegionVector:
		return FlagHasVector
	case RegionDictionary:
		return FlagHasDictionary
	default:
		return 0
	}
}

// Close finalizes the shard: it writes the footer and trailer, rewrites the
// header, fsyncs, and renames the temp file into place.
func (w *Writer) Close() error {
	if w.closed {
		return nil
	}
	// Frame the registered dictionaries into the RegionDictionary region before
	// the footer, while AddRegion still accepts writes, so its descriptor lands
	// in the footer like any other region.
	if len(w.dicts) > 0 {
		if err := w.AddRegion(RegionDictionary, CodecNone, 0, 0, encodeDictRegion(w.dicts)); err != nil {
			w.abort()
			return err
		}
	}
	w.closed = true
	if w.enc != nil {
		_ = w.enc.Close()
	}
	for _, e := range w.encDict {
		_ = e.Close()
	}

	footer := Footer{Schema: w.schema, Regions: w.regions, Stats: w.stats}
	w.stats[StatDocCount] = float64(w.header.DocCount)
	footerBytes := footer.encode()

	footerOffset := w.off
	if _, err := w.w.Write(footerBytes); err != nil {
		w.abort()
		return err
	}
	w.off += uint64(len(footerBytes))

	// Trailer: footer_length, footer_crc, magic.
	var trailer [TrailerSize]byte
	binary.LittleEndian.PutUint64(trailer[0:8], uint64(len(footerBytes)))
	binary.LittleEndian.PutUint32(trailer[8:12], codec.CRC32C(footerBytes))
	copy(trailer[12:16], Magic)
	if _, err := w.w.Write(trailer[:]); err != nil {
		w.abort()
		return err
	}
	if err := w.w.Flush(); err != nil {
		w.abort()
		return err
	}

	// Rewrite the header now that the footer offset and length are known.
	w.header.RegionCount = uint32(len(w.regions))
	w.header.FooterOffset = footerOffset
	w.header.FooterLength = uint64(len(footerBytes))
	if _, err := w.f.WriteAt(w.header.encode(), 0); err != nil {
		w.abort()
		return err
	}
	if err := w.f.Sync(); err != nil {
		w.abort()
		return err
	}
	if err := w.f.Close(); err != nil {
		return err
	}
	if err := os.Rename(w.tmpPath, w.finalPath); err != nil {
		return err
	}
	// fsync the directory so the rename is durable.
	if dir, err := os.Open(filepath.Dir(w.finalPath)); err == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return nil
}

// abort discards a half-written shard.
func (w *Writer) abort() {
	_ = w.f.Close()
	_ = os.Remove(w.tmpPath)
}
