package search

import (
	"math"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// The routing index is keyed on a front-coded term dictionary with a bloom-filter front,
// the same structure a shard's lexical region uses, because the fleet vocabulary it holds
// is the scale-critical part of serving: at 100,000 shards over billions of documents the
// vocabulary is hundreds of millions of terms, and a Go map of string keys to shard-id
// slices carries a map bucket, a string header, and a slice header for every one of them,
// which is the single largest resident cost on the broker (doc 11 lines 101, 148). The
// front-coded dictionary stores the sorted vocabulary as shared-prefix-coded blocks with no
// per-term Go overhead, and the bloom front rejects a term absent from the fleet without
// even a dictionary binary search, so routing stays in-memory and small relative to the
// shards it routes to.

// routingBloom is the bloom-filter front over the fleet vocabulary, a term-absent reject
// that never needs to touch the front-coded blocks. It is the lexical region's bloom logic
// reused at the routing tier: double-hashing over codec.XXHash64Pair, sized for a target
// false-positive rate. A false positive only costs one wasted dictionary lookup that
// returns absent, never a wrong route, because the dictionary lookup is authoritative.
type routingBloom struct {
	bits []uint64
	m    uint64 // number of bits
	k    uint64 // number of hash probes
}

// newRoutingBloom sizes a filter for n terms at false-positive rate p, the standard
// m = -n ln(p) / (ln 2)^2 bits and k = (m/n) ln 2 probes, clamped so a tiny or huge
// vocabulary still gets a sane probe count.
func newRoutingBloom(n int, p float64) *routingBloom {
	if n <= 0 {
		n = 1
	}
	m := uint64(math.Ceil(-float64(n) * math.Log(p) / (math.Ln2 * math.Ln2)))
	if m < 64 {
		m = 64
	}
	k := uint64(math.Round(float64(m) / float64(n) * math.Ln2))
	if k < 1 {
		k = 1
	}
	if k > 16 {
		k = 16
	}
	return &routingBloom{bits: make([]uint64, (m+63)/64), m: m, k: k}
}

func (b *routingBloom) add(term string) {
	h1, h2 := codec.XXHash64Pair([]byte(term))
	for i := uint64(0); i < b.k; i++ {
		bit := (h1 + i*h2) % b.m
		b.bits[bit>>6] |= 1 << (bit & 63)
	}
}

// mayContain reports whether the term might be in the set: false means definitely absent
// (skip the dictionary), true means probably present (the dictionary lookup decides). A nil
// filter has nothing to reject, so it returns true and the lookup runs.
func (b *routingBloom) mayContain(term string) bool {
	if b == nil {
		return true
	}
	h1, h2 := codec.XXHash64Pair([]byte(term))
	for i := uint64(0); i < b.k; i++ {
		bit := (h1 + i*h2) % b.m
		if b.bits[bit>>6]&(1<<(bit&63)) == 0 {
			return false
		}
	}
	return true
}

// routingBlockSize is the number of terms per front-coded block, the same 16 the lexical
// dictionary uses: a binary search lands on a block's anchor, then a short forward scan
// over at most this many terms finds the target, so the lookup is logarithmic in the block
// count and constant in the block.
const routingBlockSize = 16

// routingDict is a front-coded, bloom-fronted dictionary over a sorted term set that maps a
// term to its ordinal position (0..n-1) in sorted order, the index into the shard-set table.
// It holds no postings, only the vocabulary, so it is small relative to the shards.
type routingDict struct {
	bloom *routingBloom

	// anchorTerms holds the first term of each block in full, the only terms stored
	// uncoded, so a binary search over the anchors finds the block a target lives in
	// without decoding any block. blockOff is each block's byte offset into blocks.
	anchorTerms []string
	blockOff    []uint32

	// blocks is the front-coded vocabulary: per term, the shared-prefix length with the
	// previous term in its block (uvarint, zero for an anchor), the suffix length (uvarint),
	// and the suffix bytes, so a term costs only the bytes it does not share with its
	// predecessor.
	blocks []byte

	n int
}

// newRoutingDict front-codes a sorted, unique term set into a bloom-fronted dictionary. The
// terms must be sorted ascending and deduplicated; the builder takes them in that order, so
// each term's ordinal is its position in the slice.
func newRoutingDict(sortedTerms []string) *routingDict {
	d := &routingDict{n: len(sortedTerms)}
	if d.n == 0 {
		return d
	}
	d.bloom = newRoutingBloom(d.n, 0.01)
	for i, term := range sortedTerms {
		d.bloom.add(term)
		if i%routingBlockSize == 0 {
			// Block boundary: record the anchor in full and the block's byte offset, and
			// start the block with the anchor coded as a zero-shared full suffix.
			d.anchorTerms = append(d.anchorTerms, term)
			d.blockOff = append(d.blockOff, uint32(len(d.blocks)))
			d.blocks = appendCodedTerm(d.blocks, "", term)
			continue
		}
		d.blocks = appendCodedTerm(d.blocks, sortedTerms[i-1], term)
	}
	return d
}

// appendCodedTerm front-codes term against prev (the previous term in the same block) and
// appends [shared uvarint][suffixLen uvarint][suffix] to b.
func appendCodedTerm(b []byte, prev, term string) []byte {
	shared := commonPrefixLen(prev, term)
	suffix := term[shared:]
	b = codec.AppendUvarint(b, uint64(shared))
	b = codec.AppendUvarint(b, uint64(len(suffix)))
	return append(b, suffix...)
}

func commonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

func (d *routingDict) len() int { return d.n }

// lookup returns the ordinal of term in the sorted vocabulary, or -1 if absent. It rejects
// a fleet-absent term at the bloom front, then binary-searches the anchors for the block the
// term could be in and scans that block forward, reconstructing each front-coded term until
// it finds the target or passes where it would sort.
func (d *routingDict) lookup(term string) int {
	if d.n == 0 || !d.bloom.mayContain(term) {
		return -1
	}
	// Largest anchor <= term: sort.Search finds the first anchor strictly greater, so the
	// block is the one before it. A term below the first anchor cannot be present.
	bi := sort.Search(len(d.anchorTerms), func(i int) bool { return d.anchorTerms[i] > term }) - 1
	if bi < 0 {
		return -1
	}
	if d.anchorTerms[bi] == term {
		return bi * routingBlockSize
	}

	off := d.blockOff[bi]
	end := uint32(len(d.blocks))
	if bi+1 < len(d.blockOff) {
		end = d.blockOff[bi+1]
	}
	// Reconstruct the block forward. The anchor is index 0 and already compared above, so
	// start the scan at the second term, rebuilding each from the prior term's prefix.
	prev := d.anchorTerms[bi]
	pos := off
	for j := 0; j < routingBlockSize && pos < end; j++ {
		shared, n1 := codec.Uvarint(d.blocks[pos:])
		pos += uint32(n1)
		slen, n2 := codec.Uvarint(d.blocks[pos:])
		pos += uint32(n2)
		suffix := string(d.blocks[pos : pos+uint32(slen)])
		pos += uint32(slen)
		if j == 0 {
			// The anchor entry, already handled by the anchorTerms compare above.
			continue
		}
		cur := prev[:shared] + suffix
		if cur == term {
			return bi*routingBlockSize + j
		}
		if cur > term {
			return -1 // passed where the term would sort: absent
		}
		prev = cur
	}
	return -1
}

// sizeBytes estimates the dictionary's resident size, the bloom bits plus the front-coded
// blocks plus the anchor index, for the size measurement the routing index reports against a
// Go map of the same vocabulary.
func (d *routingDict) sizeBytes() int {
	total := len(d.blocks) + len(d.blockOff)*4
	if d.bloom != nil {
		total += len(d.bloom.bits) * 8
	}
	for _, a := range d.anchorTerms {
		total += len(a) + 16 // bytes plus the string header
	}
	return total
}
