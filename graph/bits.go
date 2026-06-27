package graph

import "math/bits"

// This file holds the bit-level instantaneous codes the compressed adjacency
// lists are built from: unary, Elias gamma, and Boldi-Vigna zeta. They are the
// same family WebGraph uses, reimplemented here in pure Go so a .tsumugi shard
// decodes with no external dependency. Every code is self-delimiting, so an
// adjacency record is a back-to-back sequence of them with no separators.
//
// The naturals encoded here are >= 0. gamma and zeta are classically defined for
// x >= 1, so the natural n is mapped to n+1 before coding and back after. Signed
// gaps go through zig-zag first.

// bitWriter accumulates bits most-significant-first into a byte buffer.
type bitWriter struct {
	buf   []byte
	cur   byte
	nbits uint8  // bits filled in cur, 0..7
	bits  uint64 // total bits written so far, the running bit offset
}

func (w *bitWriter) writeBit(b uint64) {
	w.cur |= byte(b&1) << (7 - w.nbits)
	w.nbits++
	w.bits++
	if w.nbits == 8 {
		w.buf = append(w.buf, w.cur)
		w.cur = 0
		w.nbits = 0
	}
}

// writeBits writes the low n bits of v, most-significant first.
func (w *bitWriter) writeBits(v uint64, n int) {
	for i := n - 1; i >= 0; i-- {
		w.writeBit((v >> uint(i)) & 1)
	}
}

// finish flushes the partial byte and returns the buffer.
func (w *bitWriter) finish() []byte {
	if w.nbits > 0 {
		w.buf = append(w.buf, w.cur)
		w.cur = 0
		w.nbits = 0
	}
	return w.buf
}

// writeUnary writes n zeros then a terminating one. The decoder counts the zeros.
func (w *bitWriter) writeUnary(n int) {
	for i := 0; i < n; i++ {
		w.writeBit(0)
	}
	w.writeBit(1)
}

// writeGammaPos encodes x >= 1 in Elias gamma: floor(log2 x) zeros, then x with
// its leading one, which is l zeros, a one, and the low l bits.
func (w *bitWriter) writeGammaPos(x uint64) {
	l := bits.Len64(x) - 1
	w.writeUnary(l)
	w.writeBits(x&((uint64(1)<<uint(l))-1), l)
}

// writeGamma encodes the natural n >= 0 as gamma of n+1.
func (w *bitWriter) writeGamma(n uint64) { w.writeGammaPos(n + 1) }

// ceilLog2 returns the number of bits to address z values, ceil(log2 z).
func ceilLog2(z uint64) uint {
	if z <= 1 {
		return 0
	}
	return uint(bits.Len64(z - 1))
}

// writeMinimalBinary writes v in [0, z) in the shortened binary code: values
// below the threshold take ceil(log2 z)-1 bits, the rest take ceil(log2 z).
func (w *bitWriter) writeMinimalBinary(v, z uint64) {
	if z <= 1 {
		return
	}
	s := ceilLog2(z)
	t := (uint64(1) << s) - z
	if v < t {
		w.writeBits(v, int(s)-1)
	} else {
		w.writeBits(v+t, int(s))
	}
}

// writeZetaPos encodes x >= 1 in zeta with parameter k, the code Boldi and Vigna
// tuned for the power-law gap distribution of web-graph adjacency lists.
func (w *bitWriter) writeZetaPos(x uint64, k int) {
	m := bits.Len64(x) - 1 // floor(log2 x)
	h := m / k
	w.writeUnary(h)
	base := uint64(1) << uint(h*k)
	hi := uint64(1) << uint((h+1)*k)
	w.writeMinimalBinary(x-base, hi-base)
}

// writeZeta encodes the natural n >= 0 as zeta of n+1.
func (w *bitWriter) writeZeta(n uint64, k int) { w.writeZetaPos(n+1, k) }

// zigzag maps a signed gap to a natural so small magnitudes stay small.
func zigzag(g int64) uint64   { return uint64((g << 1) ^ (g >> 63)) }
func unzigzag(u uint64) int64 { return int64(u>>1) ^ -int64(u&1) }

func (w *bitWriter) writeSignedGamma(g int64)       { w.writeGamma(zigzag(g)) }
func (w *bitWriter) writeSignedZeta(g int64, k int) { w.writeZeta(zigzag(g), k) }

// bitReader reads the codes back from a byte buffer starting at a bit offset.
type bitReader struct {
	buf []byte
	pos uint64 // current bit offset
}

func newBitReader(buf []byte, bitPos uint64) *bitReader {
	return &bitReader{buf: buf, pos: bitPos}
}

func (r *bitReader) readBit() uint64 {
	b := (r.buf[r.pos>>3] >> (7 - (r.pos & 7))) & 1
	r.pos++
	return uint64(b)
}

func (r *bitReader) readBits(n int) uint64 {
	var v uint64
	for i := 0; i < n; i++ {
		v = (v << 1) | r.readBit()
	}
	return v
}

func (r *bitReader) readUnary() int {
	n := 0
	for r.readBit() == 0 {
		n++
	}
	return n
}

func (r *bitReader) readGammaPos() uint64 {
	l := r.readUnary()
	return (uint64(1) << uint(l)) | r.readBits(l)
}

func (r *bitReader) readGamma() uint64 { return r.readGammaPos() - 1 }

func (r *bitReader) readMinimalBinary(z uint64) uint64 {
	if z <= 1 {
		return 0
	}
	s := ceilLog2(z)
	t := (uint64(1) << s) - z
	x := r.readBits(int(s) - 1)
	if x < t {
		return x
	}
	x = (x << 1) | r.readBit()
	return x - t
}

func (r *bitReader) readZetaPos(k int) uint64 {
	h := r.readUnary()
	base := uint64(1) << uint(h*k)
	hi := uint64(1) << uint((h+1)*k)
	return base + r.readMinimalBinary(hi-base)
}

func (r *bitReader) readZeta(k int) uint64 { return r.readZetaPos(k) - 1 }

func (r *bitReader) readSignedGamma() int64     { return unzigzag(r.readGamma()) }
func (r *bitReader) readSignedZeta(k int) int64 { return unzigzag(r.readZeta(k)) }
