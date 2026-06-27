package lexical

import "github.com/tamnd/tsumugi/codec"

// A docID block stores its postings' docIDs as a stream of gaps, the delta from
// one docID to the next. The gap stream is the densest part of the index and the
// hottest to decode, so its integer codec is a real lever on both shard size and
// query latency. The codec is pluggable: a region records which one it used in its
// header, so a reader decodes with the same scheme the build wrote, and the format
// can move forward without a flag day.
//
// docCodec is the seam. encode turns a block's gaps into the on-disk stream;
// decode turns that stream back into the same gaps. The payload (field masks and
// term frequencies) is not the codec's concern; only the docID gap stream is.
type docCodec interface {
	// id is the selector written into the region header so a reader can pick the
	// matching codec back out.
	id() uint16
	// encode appends the encoded form of gaps to dst and returns the grown slice.
	encode(dst []byte, gaps []uint32) []byte
	// decode reads the gaps back from a stream encode produced. The stream is the
	// exact byte slice the block header delimits, so decode consumes all of it.
	decode(src []byte) ([]uint32, error)
}

// docID codec selectors, the values stored in the region header's docidCodec field.
const (
	// docCodecVarint is the plain LEB128 varint gap stream, one uvarint per gap with
	// no count prefix. It is the original format: the stream is self-delimiting, so
	// decode reads until the slice is exhausted. It is the default: on the dense,
	// small-gap posting lists web postings produce it is the densest option (a gap
	// under 128 is a single byte), and measured against StreamVByte in a scalar Go
	// decoder it is no slower, so there is nothing to trade away.
	docCodecVarint = 0
	// docCodecStreamVByte is StreamVByte: a uvarint count, then one control byte per
	// group of four gaps holding four 2-bit length codes, then the gap bytes packed
	// little-endian at the length each code names. Its decode reads a gap's length
	// from the control byte with a shift and mask rather than a per-byte
	// continuation-bit loop, which is a large win under a SIMD shuffle decode. In the
	// scalar Go decoder here that edge is small, and the control bytes cost about
	// 0.25 byte per gap, so on dense lists it is a few percent larger than varint for
	// a few percent faster decode. It is implemented, selectable, and proven to serve
	// identical results so a SIMD decode path or a workload of large gaps can adopt it
	// without a format change, but it is not the default.
	docCodecStreamVByte = 1
	// docCodecPFor is patched frame-of-reference bit packing, the FastPFOR family: a
	// block picks one bit width that covers most of its gaps, packs every gap's low
	// bits at that width, and stores the few gaps that overflow it as patched
	// exceptions (position and full value). On the dense posting lists web text
	// produces, where a common term's gaps are mostly 1, the chosen width is a
	// handful of bits, so the gap stream is several times smaller than varint's one
	// byte per gap. It is the density codec the spec's size budget leans on.
	docCodecPFor = 2
)

// Exported aliases of the codec selectors, the values Builder.WithDocCodec and
// SpimiBuilder.WithDocCodec take. They are public so a caller can pick the gap
// codec by name rather than by a bare integer.
const (
	CodecVarint      = docCodecVarint
	CodecStreamVByte = docCodecStreamVByte
	CodecPFor        = docCodecPFor
)

// codecByID resolves a header selector to its codec, refusing an unknown id so a
// region written by a newer format is rejected cleanly rather than misread.
func codecByID(id uint16) (docCodec, error) {
	switch id {
	case docCodecVarint:
		return varintCodec{}, nil
	case docCodecStreamVByte:
		return streamVByteCodec{}, nil
	case docCodecPFor:
		return pforCodec{}, nil
	default:
		return nil, errCorrupt
	}
}

// varintCodec is the original LEB128 gap stream. Its bytes are identical to the
// pre-codec format, so a region built with it is byte-for-byte what the engine
// wrote before the codec seam existed.
type varintCodec struct{}

func (varintCodec) id() uint16 { return docCodecVarint }

func (varintCodec) encode(dst []byte, gaps []uint32) []byte {
	for _, g := range gaps {
		dst = codec.AppendUvarint(dst, uint64(g))
	}
	return dst
}

func (varintCodec) decode(src []byte) ([]uint32, error) {
	var gaps []uint32
	for off := 0; off < len(src); {
		g, n := codec.Uvarint(src[off:])
		if n <= 0 {
			return nil, errCorrupt
		}
		off += n
		gaps = append(gaps, uint32(g))
	}
	return gaps, nil
}

// streamVByteCodec implements StreamVByte over the gap stream. The layout is a
// uvarint count, then the control bytes (ceil(count/4) of them, two bits per gap
// naming its byte length 1..4), then the gap data bytes little-endian. Separating
// the control bytes from the data is the StreamVByte shape: a decoder reads a
// gap's length from the control stream with a shift and mask, then copies that
// many data bytes, with no per-byte branch.
type streamVByteCodec struct{}

func (streamVByteCodec) id() uint16 { return docCodecStreamVByte }

// svLen returns the 2-bit length code for a gap and the byte count it names, 1..4.
func svLen(g uint32) (code byte, n int) {
	switch {
	case g < 1<<8:
		return 0, 1
	case g < 1<<16:
		return 1, 2
	case g < 1<<24:
		return 2, 3
	default:
		return 3, 4
	}
}

func (streamVByteCodec) encode(dst []byte, gaps []uint32) []byte {
	dst = codec.AppendUvarint(dst, uint64(len(gaps)))
	if len(gaps) == 0 {
		return dst
	}
	nctrl := (len(gaps) + 3) / 4
	ctrlAt := len(dst)
	dst = append(dst, make([]byte, nctrl)...)
	for i, g := range gaps {
		code, n := svLen(g)
		dst[ctrlAt+i/4] |= code << uint((i%4)*2)
		for b := 0; b < n; b++ {
			dst = append(dst, byte(g>>(8*b)))
		}
	}
	return dst
}

func (streamVByteCodec) decode(src []byte) ([]uint32, error) {
	count, n := codec.Uvarint(src)
	if n <= 0 {
		return nil, errCorrupt
	}
	off := n
	c := int(count)
	if c == 0 {
		return nil, nil
	}
	nctrl := (c + 3) / 4
	if off+nctrl > len(src) {
		return nil, errCorrupt
	}
	ctrl := src[off : off+nctrl]
	data := src[off+nctrl:]
	gaps := make([]uint32, c)
	pos := 0
	for i := 0; i < c; i++ {
		code := (ctrl[i/4] >> uint((i%4)*2)) & 3
		nbytes := int(code) + 1
		if pos+nbytes > len(data) {
			return nil, errCorrupt
		}
		var g uint32
		for b := 0; b < nbytes; b++ {
			g |= uint32(data[pos+b]) << (8 * b)
		}
		gaps[i] = g
		pos += nbytes
	}
	if pos != len(data) {
		return nil, errCorrupt
	}
	return gaps, nil
}

// pforCodec implements patched frame-of-reference bit packing over the gap stream.
// The layout is a uvarint gap count, a one-byte bit width b, a uvarint exception
// count, that many (uvarint position, uvarint full value) exception pairs, then the
// low b bits of every gap packed little-endian. A gap that does not fit in b bits is
// an exception: its packed low bits are written like any other and overwritten with
// the full value on decode, so the packing stays a fixed b bits per gap. The width
// is chosen to minimize the encoded size, trading wider packing against fewer
// exceptions, which is where the density comes from on the small gaps dense posting
// lists produce.
type pforCodec struct{}

func (pforCodec) id() uint16 { return docCodecPFor }

// uvarintLen is the number of bytes AppendUvarint writes for x, used to cost a
// candidate bit width exactly rather than estimate.
func uvarintLen(x uint64) int {
	n := 1
	for x >= 0x80 {
		x >>= 7
		n++
	}
	return n
}

// pforChooseWidth returns the bit width that minimizes the encoded gap-stream size.
// For each candidate width it counts the gaps that overflow it (the exceptions) and
// sizes the packed bits plus the exception list, and keeps the cheapest. Widths run
// 0..32; at 32 every uint32 fits so there are no exceptions, the safe upper bound.
func pforChooseWidth(gaps []uint32) int {
	bestB, bestCost := 32, int(^uint(0)>>1)
	for b := 0; b <= 32; b++ {
		limit := uint32(0)
		if b < 32 {
			limit = uint32(1) << uint(b)
		}
		excBytes := 0
		excCount := 0
		for i, g := range gaps {
			if b < 32 && g >= limit {
				excCount++
				excBytes += uvarintLen(uint64(i)) + uvarintLen(uint64(g))
			}
		}
		packedBytes := (b*len(gaps) + 7) / 8
		cost := 1 + uvarintLen(uint64(excCount)) + excBytes + packedBytes
		if cost < bestCost {
			bestCost, bestB = cost, b
		}
	}
	return bestB
}

// bitpack appends the low b bits of each value to dst, little-endian across byte
// boundaries. b ranges 0..32; b of 0 writes nothing.
func bitpack(dst []byte, vals []uint32, b int) []byte {
	if b == 0 {
		return dst
	}
	mask := (uint64(1) << uint(b)) - 1
	var acc uint64
	var nbits int
	for _, v := range vals {
		acc |= (uint64(v) & mask) << uint(nbits)
		nbits += b
		for nbits >= 8 {
			dst = append(dst, byte(acc))
			acc >>= 8
			nbits -= 8
		}
	}
	if nbits > 0 {
		dst = append(dst, byte(acc))
	}
	return dst
}

// bitunpack reads count values of b bits each from src into out, the inverse of
// bitpack. src must hold exactly ceil(b*count/8) bytes.
func bitunpack(src []byte, out []uint32, b int) {
	if b == 0 {
		return
	}
	mask := (uint64(1) << uint(b)) - 1
	var acc uint64
	var nbits int
	si := 0
	for i := range out {
		for nbits < b {
			acc |= uint64(src[si]) << uint(nbits)
			si++
			nbits += 8
		}
		out[i] = uint32(acc & mask)
		acc >>= uint(b)
		nbits -= b
	}
}

func (pforCodec) encode(dst []byte, gaps []uint32) []byte {
	dst = codec.AppendUvarint(dst, uint64(len(gaps)))
	if len(gaps) == 0 {
		return dst
	}
	b := pforChooseWidth(gaps)
	dst = append(dst, byte(b))

	var exc []int
	if b < 32 {
		limit := uint32(1) << uint(b)
		for i, g := range gaps {
			if g >= limit {
				exc = append(exc, i)
			}
		}
	}
	dst = codec.AppendUvarint(dst, uint64(len(exc)))
	for _, i := range exc {
		dst = codec.AppendUvarint(dst, uint64(i))
		dst = codec.AppendUvarint(dst, uint64(gaps[i]))
	}
	return bitpack(dst, gaps, b)
}

func (pforCodec) decode(src []byte) ([]uint32, error) {
	count, n := codec.Uvarint(src)
	if n <= 0 {
		return nil, errCorrupt
	}
	off := n
	c := int(count)
	if c == 0 {
		return nil, nil
	}
	if off >= len(src) {
		return nil, errCorrupt
	}
	b := int(src[off])
	off++
	if b > 32 {
		return nil, errCorrupt
	}
	excCount, n := codec.Uvarint(src[off:])
	if n <= 0 {
		return nil, errCorrupt
	}
	off += n
	type patch struct {
		pos int
		val uint32
	}
	exc := make([]patch, excCount)
	for i := range exc {
		p, n := codec.Uvarint(src[off:])
		if n <= 0 {
			return nil, errCorrupt
		}
		off += n
		v, n := codec.Uvarint(src[off:])
		if n <= 0 {
			return nil, errCorrupt
		}
		off += n
		if int(p) >= c {
			return nil, errCorrupt
		}
		exc[i] = patch{pos: int(p), val: uint32(v)}
	}
	need := (b*c + 7) / 8
	if off+need != len(src) {
		return nil, errCorrupt
	}
	gaps := make([]uint32, c)
	bitunpack(src[off:], gaps, b)
	for _, e := range exc {
		gaps[e.pos] = e.val
	}
	return gaps, nil
}
