package mph

import (
	"errors"

	"github.com/tamnd/tsumugi/codec"
)

// ErrCorrupt is returned when a serialized directory or hash cannot be parsed,
// either because the bytes are truncated or because a length field overruns the
// buffer. A caller that loads a directory from disk treats it the same as an
// absent file: rebuild from the source rather than trust a damaged structure.
var ErrCorrupt = errors.New("mph: corrupt serialized form")

// Append serializes the directory onto b and returns the extended slice. The form
// is the minimal perfect hash (its levels and the overflow table) followed by the
// fingerprint and value arrays, every integer little-endian so the bytes are
// byte-identical for the same directory on every platform. The per-level cumulative
// popcount is not stored because it is a pure function of the level bit array, so it
// is rebuilt on read; everything else round-trips exactly. This is what lets a build
// persist the canonical-URL directory as a loadable artifact instead of rebuilding
// it from a full shard scan on every recrawl.
func (d *Dir) Append(b []byte) []byte {
	b = appendMPH(b, d.mph)
	b = codec.AppendUint32(b, uint32(len(d.fp)))
	for _, f := range d.fp {
		b = codec.AppendUint64(b, f)
	}
	b = codec.AppendUint32(b, uint32(len(d.val)))
	for _, v := range d.val {
		b = codec.AppendUint32(b, v)
	}
	return b
}

// ReadDir parses a directory serialized by Append. It returns the directory and the
// number of bytes consumed, so a caller can frame the directory alongside other data;
// a truncated or inconsistent buffer returns ErrCorrupt rather than a partial value.
func ReadDir(b []byte) (*Dir, int, error) {
	m, off, err := readMPH(b)
	if err != nil {
		return nil, 0, err
	}
	p := &reader{b: b, off: off}
	fpLen := int(p.u32())
	fp := make([]uint64, fpLen)
	for i := range fp {
		fp[i] = p.u64()
	}
	valLen := int(p.u32())
	val := make([]uint32, valLen)
	for i := range val {
		val[i] = p.u32()
	}
	if p.err {
		return nil, 0, ErrCorrupt
	}
	return &Dir{mph: m, fp: fp, val: val}, p.off, nil
}

// appendMPH serializes a minimal perfect hash: the key count, then each level's
// size, seed, rank base, and bit words, then the overflow table.
func appendMPH(b []byte, m *MPH) []byte {
	b = codec.AppendUint64(b, m.n)
	b = codec.AppendUint32(b, uint32(len(m.levels)))
	for i := range m.levels {
		l := &m.levels[i]
		b = codec.AppendUint64(b, l.size)
		b = codec.AppendUint64(b, l.seed)
		b = codec.AppendUint64(b, l.rankBase)
		b = codec.AppendUint32(b, uint32(len(l.bits)))
		for _, w := range l.bits {
			b = codec.AppendUint64(b, w)
		}
	}
	b = codec.AppendUint32(b, uint32(len(m.overflow)))
	// The overflow table is empty on every real key set (it only catches the level
	// cap and duplicate keys), so its iteration order does not affect a real
	// artifact's bytes; it is serialized for completeness so the form is total.
	for k, id := range m.overflow {
		b = codec.AppendUvarint(b, uint64(len(k)))
		b = append(b, k...)
		b = codec.AppendUint64(b, id)
	}
	return b
}

// readMPH parses a minimal perfect hash and rebuilds each level's cumulative
// popcount, which Append does not store. It returns the hash and the offset past it.
func readMPH(b []byte) (*MPH, int, error) {
	p := &reader{b: b}
	m := &MPH{}
	m.n = p.u64()
	nLevels := int(p.u32())
	if nLevels < 0 || nLevels > maxLevels {
		return nil, 0, ErrCorrupt
	}
	m.levels = make([]level, nLevels)
	for i := 0; i < nLevels; i++ {
		l := &m.levels[i]
		l.size = p.u64()
		l.seed = p.u64()
		l.rankBase = p.u64()
		nWords := int(p.u32())
		if nWords < 0 || uint64(nWords)*64 < l.size || p.off+nWords*8 > len(b) {
			return nil, 0, ErrCorrupt
		}
		l.bits = make([]uint64, nWords)
		for w := range l.bits {
			l.bits[w] = p.u64()
		}
		l.buildRank()
	}
	nOverflow := int(p.u32())
	if nOverflow > 0 {
		m.overflow = make(map[string]uint64, nOverflow)
		for i := 0; i < nOverflow; i++ {
			kl := int(p.uvarint())
			if kl < 0 || p.off+kl > len(b) {
				return nil, 0, ErrCorrupt
			}
			k := string(b[p.off : p.off+kl])
			p.off += kl
			m.overflow[k] = p.u64()
		}
	}
	if p.err {
		return nil, 0, ErrCorrupt
	}
	return m, p.off, nil
}

// reader walks a byte slice, tracking an overrun in a sticky error flag so the parse
// path stays free of per-read error handling and a truncated buffer fails the final
// check rather than panicking on a slice bound.
type reader struct {
	b   []byte
	off int
	err bool
}

func (p *reader) u32() uint32 {
	if p.off+4 > len(p.b) {
		p.err = true
		return 0
	}
	v := codec.Uint32(p.b[p.off:])
	p.off += 4
	return v
}

func (p *reader) u64() uint64 {
	if p.off+8 > len(p.b) {
		p.err = true
		return 0
	}
	v := codec.Uint64(p.b[p.off:])
	p.off += 8
	return v
}

func (p *reader) uvarint() uint64 {
	if p.off >= len(p.b) {
		p.err = true
		return 0
	}
	v, n := codec.Uvarint(p.b[p.off:])
	if n <= 0 {
		p.err = true
		return 0
	}
	p.off += n
	return v
}
