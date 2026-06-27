package graph

import (
	"errors"
	"math/bits"

	"github.com/tamnd/tsumugi/codec"
)

// ef is an Elias-Fano coded monotone non-decreasing sequence. The adjacency
// stream has no separators between records, so to reach node x's record by
// random access we need the bit offset where it starts. Storing those N+1
// offsets as plain uint64s would be 64 bits a node; Elias-Fano stores them in
// about 2 + log2(universe/N) bits a node with O(1) random access, which is under
// a bit an edge of overhead. The sequential scan PageRank does never touches
// this; it decodes records back to back. Only random access uses it.
//
// Each value splits into a low part of l bits stored verbatim and a high part
// stored as a unary bucket bitvector: value i sets bit (value>>l)+i. Reading
// value i is the i-th set bit (select1) minus i, shifted up, or-ed with the low
// bits. select1 is sampled every 64 ones so it costs a short word scan.
type ef struct {
	n   int
	l   uint
	low []byte   // packed n*l low bits, MSB-first per value
	hi  []uint64 // high bucket bitvector, LSB bit order within each word

	sampleWord []uint32
	sampleOnes []uint32
	sampleStep int
}

// buildEF Elias-Fano codes a monotone non-decreasing slice.
func buildEF(vals []uint64) *ef {
	n := len(vals)
	e := &ef{n: n}
	if n == 0 {
		e.buildSamples()
		return e
	}
	maxv := vals[n-1]
	if maxv >= uint64(n) {
		e.l = uint(bits.Len64(maxv/uint64(n)) - 1)
	}

	lw := &bitWriter{}
	mask := (uint64(1) << e.l) - 1
	maxHigh := maxv >> e.l
	words := int((maxHigh+uint64(n-1))/64) + 1
	e.hi = make([]uint64, words)
	for i, v := range vals {
		lw.writeBits(v&mask, int(e.l))
		pos := (v >> e.l) + uint64(i)
		e.hi[pos>>6] |= uint64(1) << (pos & 63)
	}
	e.low = lw.finish()
	e.buildSamples()
	return e
}

// buildSamples records, for every 64th one-bit, the word it lives in and the
// number of ones before that word, so select1 starts its scan close to home.
func (e *ef) buildSamples() {
	e.sampleStep = 64
	e.sampleWord = e.sampleWord[:0]
	e.sampleOnes = e.sampleOnes[:0]
	running := 0
	target := 0
	for wi, word := range e.hi {
		pc := bits.OnesCount64(word)
		for target < e.n && target < running+pc {
			e.sampleWord = append(e.sampleWord, uint32(wi))
			e.sampleOnes = append(e.sampleOnes, uint32(running))
			target += e.sampleStep
		}
		running += pc
	}
}

// select1 returns the bit position of the i-th set bit, zero indexed.
func (e *ef) select1(i int) int {
	j := i / e.sampleStep
	wi := int(e.sampleWord[j])
	ones := int(e.sampleOnes[j])
	for {
		word := e.hi[wi]
		pc := bits.OnesCount64(word)
		if ones+pc > i {
			need := i - ones
			for ; need > 0; need-- {
				word &= word - 1
			}
			return wi*64 + bits.TrailingZeros64(word)
		}
		ones += pc
		wi++
	}
}

// get returns value i.
func (e *ef) get(i int) uint64 {
	high := uint64(e.select1(i) - i)
	var low uint64
	if e.l > 0 {
		low = newBitReader(e.low, uint64(i)*uint64(e.l)).readBits(int(e.l))
	}
	return (high << e.l) | low
}

// encode serializes the structure; the samples are rebuilt at decode, not stored.
func (e *ef) encode() []byte {
	b := make([]byte, 0, 16+len(e.low)+len(e.hi)*8)
	b = codec.AppendUint32(b, uint32(e.n))
	b = append(b, byte(e.l))
	b = codec.AppendUint32(b, uint32(len(e.low)))
	b = append(b, e.low...)
	b = codec.AppendUint32(b, uint32(len(e.hi)))
	for _, w := range e.hi {
		b = codec.AppendUint64(b, w)
	}
	return b
}

var errEF = errors.New("graph: corrupt offset index")

// decodeEF parses a serialized ef and rebuilds its select samples.
func decodeEF(b []byte) (*ef, error) {
	if len(b) < 9 {
		return nil, errEF
	}
	e := &ef{}
	e.n = int(codec.Uint32(b))
	e.l = uint(b[4])
	lowLen := int(codec.Uint32(b[5:]))
	off := 9
	if len(b) < off+lowLen+4 {
		return nil, errEF
	}
	e.low = b[off : off+lowLen]
	off += lowLen
	hiLen := int(codec.Uint32(b[off:]))
	off += 4
	if len(b) < off+hiLen*8 {
		return nil, errEF
	}
	e.hi = make([]uint64, hiLen)
	for i := 0; i < hiLen; i++ {
		e.hi[i] = codec.Uint64(b[off:])
		off += 8
	}
	e.buildSamples()
	return e, nil
}
