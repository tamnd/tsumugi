package tsumugi

import (
	"fmt"

	"github.com/tamnd/tsumugi/codec"
)

// A shard can carry shared zstd dictionaries so that many small, similar values
// compress against a common context instead of each paying its own cold-start
// cost. The dictionaries live in one RegionDictionary region, addressed by a
// non-zero dict_id a region descriptor records, and a region stored CodecZstdDict
// is compressed and decompressed against the dictionary its descriptor names.
//
// The dictionaries are raw-content dictionaries: the shared context is a run of
// representative bytes that zstd references as if they sat immediately before the
// input. The pure-Go zstd trainer is unreliable on real corpora (it returns a
// zero-length dictionary on the ccrawl markdown columns), and the win the format
// is after, many small payloads sharing a context, is fully realized by a raw
// dictionary, so the mechanism standardizes on raw content. dict_id 0 is reserved
// to mean no dictionary, so a registered dictionary must use a non-zero id.

// dictMagic and dictVersion frame the RegionDictionary payload, a small container
// of (id, bytes) pairs distinct from the outer shard framing.
const (
	dictMagic   = "TSDC"
	dictVersion = uint32(1)
)

// dictEntry is one registered shared dictionary: its id and its raw content.
type dictEntry struct {
	id      uint32
	content []byte
}

// encodeDictRegion frames the registered dictionaries into the RegionDictionary
// payload: magic, version, count, then each entry as id, length, bytes.
func encodeDictRegion(dicts []dictEntry) []byte {
	size := 4 + 4 + 4
	for _, d := range dicts {
		size += 4 + 4 + len(d.content)
	}
	out := make([]byte, 0, size)
	out = append(out, dictMagic...)
	out = codec.AppendUint32(out, dictVersion)
	out = codec.AppendUint32(out, uint32(len(dicts)))
	for _, d := range dicts {
		out = codec.AppendUint32(out, d.id)
		out = codec.AppendUint32(out, uint32(len(d.content)))
		out = append(out, d.content...)
	}
	return out
}

// decodeDictRegion parses the RegionDictionary payload back into an id-to-content
// map. It aliases the region bytes; callers must not mutate the returned slices.
func decodeDictRegion(b []byte) (map[uint32][]byte, error) {
	if len(b) < 12 || string(b[0:4]) != dictMagic {
		return nil, fmt.Errorf("tsumugi: bad dictionary region")
	}
	if codec.Uint32(b[4:]) != dictVersion {
		return nil, fmt.Errorf("tsumugi: unsupported dictionary version")
	}
	count := int(codec.Uint32(b[8:]))
	off := 12
	m := make(map[uint32][]byte, count)
	for i := 0; i < count; i++ {
		if off+8 > len(b) {
			return nil, fmt.Errorf("tsumugi: truncated dictionary region")
		}
		id := codec.Uint32(b[off:])
		n := int(codec.Uint32(b[off+4:]))
		off += 8
		if off+n > len(b) {
			return nil, fmt.Errorf("tsumugi: truncated dictionary content")
		}
		if id == 0 {
			return nil, fmt.Errorf("tsumugi: dictionary id 0 is reserved")
		}
		if _, dup := m[id]; dup {
			return nil, fmt.Errorf("tsumugi: duplicate dictionary id %d", id)
		}
		m[id] = b[off : off+n]
		off += n
	}
	return m, nil
}

// DeriveDictionary builds a raw-content dictionary from sample values, the shared
// context a CodecZstdDict region compresses against. It walks the samples on an
// even stride so the dictionary captures variety across the whole set rather than
// only its first rows, and stops once maxBytes are gathered. It returns nil when
// there is nothing to derive, which a caller reads as "store these without a
// dictionary".
func DeriveDictionary(samples [][]byte, maxBytes int) []byte {
	if maxBytes <= 0 || len(samples) == 0 {
		return nil
	}
	step := len(samples) / 1024
	if step < 1 {
		step = 1
	}
	out := make([]byte, 0, maxBytes)
	for i := 0; i < len(samples) && len(out) < maxBytes; i += step {
		v := samples[i]
		if len(out)+len(v) > maxBytes {
			v = v[:maxBytes-len(out)]
		}
		out = append(out, v...)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
