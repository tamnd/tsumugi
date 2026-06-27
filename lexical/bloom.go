package lexical

import (
	"math"

	"github.com/tamnd/tsumugi/codec"
)

// bloom is the term membership prefilter that sits in front of the dictionary.
// A keyword query is mostly misses: a shard holds few of a query's terms, so
// rejecting an absent term here saves a dictionary lookup. The filter has no
// false negatives, so a "not present" answer is always true and skipping the
// lookup never drops a real posting list.
type bloom struct {
	bits []uint64
	m    uint64 // number of bits
	k    uint64 // number of hash probes
}

// newBloom sizes a filter for n terms at the target false-positive rate p,
// using the standard m = -n ln p / (ln 2)^2 and k = (m/n) ln 2.
func newBloom(n int, p float64) *bloom {
	if n < 1 {
		n = 1
	}
	m := bloomBits(n, p)
	k := bloomProbes(m, n)
	return &bloom{bits: make([]uint64, (m+63)/64), m: m, k: k}
}

func bloomBits(n int, p float64) uint64 {
	// -n ln p / (ln2)^2, with ln2^2 = 0.4804530139182014.
	m := uint64(-float64(n) * math.Log(p) / 0.4804530139182014)
	if m < 64 {
		m = 64
	}
	return m
}

func bloomProbes(m uint64, n int) uint64 {
	k := uint64(float64(m) / float64(n) * 0.6931471805599453)
	if k < 1 {
		k = 1
	}
	if k > 16 {
		k = 16
	}
	return k
}

// add inserts a term.
func (b *bloom) add(term string) {
	h1, h2 := codec.XXHash64Pair([]byte(term))
	for i := uint64(0); i < b.k; i++ {
		bit := (h1 + i*h2) % b.m
		b.bits[bit>>6] |= 1 << (bit & 63)
	}
}

// mayContain reports whether a term might be present. False means definitely
// absent; true means probably present, do the real dictionary lookup.
func (b *bloom) mayContain(term string) bool {
	h1, h2 := codec.XXHash64Pair([]byte(term))
	for i := uint64(0); i < b.k; i++ {
		bit := (h1 + i*h2) % b.m
		if b.bits[bit>>6]&(1<<(bit&63)) == 0 {
			return false
		}
	}
	return true
}

// encode serializes the filter: k, m, then the bit words.
func (b *bloom) encode() []byte {
	out := make([]byte, 0, 16+len(b.bits)*8)
	out = codec.AppendUvarint(out, b.k)
	out = codec.AppendUvarint(out, b.m)
	for _, w := range b.bits {
		out = codec.AppendUint64(out, w)
	}
	return out
}

// decodeBloom parses a filter from its bytes.
func decodeBloom(b []byte) (*bloom, error) {
	k, n := codec.Uvarint(b)
	if n <= 0 {
		return nil, errCorrupt
	}
	b = b[n:]
	m, n2 := codec.Uvarint(b)
	if n2 <= 0 {
		return nil, errCorrupt
	}
	b = b[n2:]
	words := (m + 63) / 64
	if uint64(len(b)) < words*8 {
		return nil, errCorrupt
	}
	bits := make([]uint64, words)
	for i := range bits {
		bits[i] = codec.Uint64(b[i*8:])
	}
	return &bloom{bits: bits, m: m, k: k}, nil
}
