package tsumugi

import (
	"encoding/binary"
	"sync"

	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi/codec"
)

// Reader opens a shard for reading. Opening is one read of the tail (the
// trailer and footer) plus the header, independent of shard size, so a broker
// holds many shards open cheaply. Region bytes are reached lazily and validated
// against their CRC on first access.
type Reader struct {
	data    []byte // the whole file, memory-mapped
	mm      *mmap  // the mapping handle, nil when data is a plain read
	Header  Header
	Footer  Footer
	dec     *zstd.Decoder
	decOnce sync.Once

	// Shared dictionaries loaded eagerly from the RegionDictionary region at Open,
	// keyed by dict_id. decDict is one decoder with every raw dictionary
	// registered, built on first CodecZstdDict access; DecodeAll selects the right
	// dictionary by the id the frame carries.
	dictContent map[uint32][]byte
	decDict     *zstd.Decoder
	decDictOnce sync.Once

	mu        sync.Mutex
	validated map[RegionKind]bool // regions whose on-disk CRC has been checked
}

// Open memory-maps the shard at path and parses its header and footer.
func Open(path string) (*Reader, error) {
	mm, err := mmapOpen(path)
	if err != nil {
		return nil, err
	}
	r := &Reader{data: mm.data, mm: mm, validated: map[RegionKind]bool{}}
	if err := r.parse(); err != nil {
		_ = mm.Close()
		return nil, err
	}
	return r, nil
}

// OpenBytes parses a shard already held in memory. It is the mmap-free path used
// in tests and on platforms without mmap.
func OpenBytes(data []byte) (*Reader, error) {
	r := &Reader{data: data, validated: map[RegionKind]bool{}}
	if err := r.parse(); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *Reader) parse() error {
	n := len(r.data)
	if n < HeaderSize+TrailerSize {
		return ErrShortFile
	}
	// Trailer at the very end.
	tr := r.data[n-TrailerSize:]
	if string(tr[12:16]) != Magic {
		return ErrBadMagic
	}
	footerLen := binary.LittleEndian.Uint64(tr[0:8])
	footerCRC := binary.LittleEndian.Uint32(tr[8:12])
	footerEnd := uint64(n - TrailerSize)
	if footerLen > footerEnd {
		return ErrCorruptFooter
	}
	footerStart := footerEnd - footerLen
	footerBytes := r.data[footerStart:footerEnd]
	if codec.CRC32C(footerBytes) != footerCRC {
		return ErrFooterCRC
	}
	footer, err := decodeFooter(footerBytes)
	if err != nil {
		return err
	}
	r.Footer = footer

	// Header at the front.
	hdr, err := decodeHeader(r.data[:HeaderSize])
	if err != nil {
		return err
	}
	// Cross-check the header against the footer location.
	if hdr.FooterOffset != footerStart || hdr.FooterLength != footerLen {
		return ErrCorruptFooter
	}
	r.Header = hdr

	// Load shared dictionaries eagerly: the region is small and any CodecZstdDict
	// region needs them, so paying the parse once at Open is cheaper than racing
	// to build it on the first decode.
	if err := r.loadDictionaries(); err != nil {
		return err
	}
	return nil
}

// loadDictionaries reads the RegionDictionary region, if present, validates its
// CRC, and parses the shared dictionaries into dictContent keyed by dict_id.
func (r *Reader) loadDictionaries() error {
	desc, ok := r.Footer.region(RegionDictionary)
	if !ok {
		return nil
	}
	if desc.Offset+desc.Length > uint64(len(r.data)) {
		return ErrCorruptFooter
	}
	onDisk := r.data[desc.Offset : desc.Offset+desc.Length]
	if codec.CRC32C(onDisk) != desc.CRC {
		return ErrRegionCRC
	}
	m, err := decodeDictRegion(onDisk)
	if err != nil {
		return err
	}
	r.dictContent = m
	r.validated[RegionDictionary] = true
	return nil
}

// dictDecoder builds, once, a single decoder with every shared dictionary
// registered as raw content. DecodeAll then selects the dictionary by the id the
// compressed frame carries, so one decoder serves every CodecZstdDict region.
func (r *Reader) dictDecoder() (*zstd.Decoder, error) {
	var derr error
	r.decDictOnce.Do(func() {
		if len(r.dictContent) == 0 {
			return
		}
		opts := make([]zstd.DOption, 0, len(r.dictContent))
		for id, content := range r.dictContent {
			opts = append(opts, zstd.WithDecoderDictRaw(id, content))
		}
		r.decDict, derr = zstd.NewReader(nil, opts...)
	})
	return r.decDict, derr
}

// HasRegion reports whether a region kind is present.
func (r *Reader) HasRegion(kind RegionKind) bool {
	_, ok := r.Footer.region(kind)
	return ok
}

// RegionDesc returns the descriptor for a region kind.
func (r *Reader) RegionDesc(kind RegionKind) (RegionDescriptor, bool) {
	return r.Footer.region(kind)
}

// Region returns a region's logical (decompressed) bytes, validating the
// on-disk CRC on first access. The returned slice aliases the mapping when the
// region is stored uncompressed, so callers must not mutate it.
func (r *Reader) Region(kind RegionKind) ([]byte, error) {
	desc, ok := r.Footer.region(kind)
	if !ok {
		return nil, ErrNoRegion
	}
	onDisk := r.data[desc.Offset : desc.Offset+desc.Length]

	r.mu.Lock()
	if !r.validated[kind] {
		if codec.CRC32C(onDisk) != desc.CRC {
			r.mu.Unlock()
			return nil, ErrRegionCRC
		}
		r.validated[kind] = true
	}
	r.mu.Unlock()

	switch desc.Codec {
	case CodecNone:
		return onDisk, nil
	case CodecZstd:
		var derr error
		r.decOnce.Do(func() {
			r.dec, derr = zstd.NewReader(nil)
		})
		if derr != nil {
			return nil, derr
		}
		return r.dec.DecodeAll(onDisk, make([]byte, 0, desc.RawLength))
	case CodecZstdDict:
		dec, derr := r.dictDecoder()
		if derr != nil {
			return nil, derr
		}
		if dec == nil {
			return nil, ErrNoRegion
		}
		return dec.DecodeAll(onDisk, make([]byte, 0, desc.RawLength))
	default:
		return nil, ErrNoRegion
	}
}

// Dictionary returns a registered shared dictionary's raw content by id and
// whether it was present. The returned slice aliases the mapping; callers must
// not mutate it.
func (r *Reader) Dictionary(id uint32) ([]byte, bool) {
	c, ok := r.dictContent[id]
	return c, ok
}

// DocCount returns N, the dense docID space size.
func (r *Reader) DocCount() uint32 { return r.Header.DocCount }

// Stat returns a shard statistic and whether it was present.
func (r *Reader) Stat(key string) (float64, bool) {
	v, ok := r.Footer.Stats[key]
	return v, ok
}

// AnalyzerHash returns the recorded analyzer_hash and whether it was present. A shard
// built before the hash was recorded returns false, the unknown case a broker treats
// as a skipped check rather than a mismatch.
func (r *Reader) AnalyzerHash() (uint64, bool) {
	v, ok := r.Footer.Stats[StatAnalyzerHash]
	if !ok {
		return 0, false
	}
	return AnalyzerHashFromStat(v), true
}

// Close releases the mapping.
func (r *Reader) Close() error {
	if r.dec != nil {
		r.dec.Close()
	}
	if r.decDict != nil {
		r.decDict.Close()
	}
	if r.mm != nil {
		return r.mm.Close()
	}
	return nil
}
