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
	w.closed = true
	if w.enc != nil {
		_ = w.enc.Close()
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
