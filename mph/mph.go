// Package mph implements a minimal perfect hash over a set of byte-string keys,
// the global node id construction doc 02 specifies for canonical URLs. A minimal
// perfect hash maps each of K distinct keys to a distinct integer in [0, K) with
// no collisions and no gaps, using a small structure of a few bits a key, and it
// is a pure function of the key so the same URL hashes to the same id in every
// shard and every build of the same corpus, which is exactly what cross-shard
// edges and cross-crawl dedup need.
//
// The construction is BBHash (Limasset, Rizk, Chikhi, Salikhov): at each level
// the still-unplaced keys are hashed into a bit array sized gamma times their
// count, every key that lands alone in its slot is placed at that level, and the
// keys that collided fall through to the next level, which repeats until no keys
// remain. A lookup walks the levels, stops at the first level whose slot bit is
// set, and returns the global rank of that bit, the count of set bits before it
// across all levels, which is the key's index in [0, K). At gamma 2 this lands
// near three bits a key.
//
// An MPH says nothing useful about a key that was not in the build set: it returns
// some index in [0, K) for any input. A caller that must distinguish members from
// non-members, which the link directory does because a link target is often a page
// the crawl never captured, pairs the MPH with a fingerprint check; see Dir.
package mph

import "math/bits"

// DefaultGamma is the bit-array load factor. Larger gamma places more keys per
// level so the structure has fewer levels and a faster lookup, at more bits a key;
// 2.0 is the BBHash default and the space/speed knee.
const DefaultGamma = 2.0

// maxLevels caps the BBHash level chain. Distinct keys diverge under the per-level
// seed so the survivor count falls geometrically and a real key set exhausts in
// well under this many levels; the cap only bounds the pathological case, where
// the leftover keys spill into an explicit overflow table.
const maxLevels = 32

// MPH is a built minimal perfect hash. It is read-only after Build and safe for
// concurrent lookups.
type MPH struct {
	levels   []level
	overflow map[string]uint64 // keys the level chain did not place, by explicit id
	n        uint64            // number of keys, the size of the output range
}

// level is one BBHash level: the bit array of placed slots, its size, the seed
// that hashed keys into it, the global rank of its first bit (the count of placed
// bits in all earlier levels), and a per-word cumulative popcount for O(1) rank.
type level struct {
	bits     []uint64
	size     uint64
	seed     uint64
	rankBase uint64
	wordRank []uint64 // wordRank[w] = set bits in bits[0:w]
}

// Build constructs a minimal perfect hash over keys, which must be distinct. A
// gamma below one falls back to the default. Duplicate keys would never separate
// across levels and are caught by the overflow table rather than looping.
func Build(keys [][]byte, gamma float64) *MPH {
	if gamma < 1 {
		gamma = DefaultGamma
	}
	m := &MPH{}
	remaining := keys
	var cumBits uint64
	for lvl := 0; len(remaining) > 0 && lvl < maxLevels; lvl++ {
		n := uint64(len(remaining))
		words := uint64(float64(n)*gamma)/64 + 1
		size := words * 64
		seed := levelSeed(lvl)

		a := make([]uint64, words)
		c := make([]uint64, words)
		for _, k := range remaining {
			i := hashKey(seed, k) % size
			if getbit(a, i) {
				setbit(c, i)
			} else {
				setbit(a, i)
			}
		}
		// A slot keeps its bit only if exactly one key landed there; a collided
		// slot is cleared and its keys fall to the next level.
		for w := range a {
			a[w] &^= c[w]
		}

		l := level{bits: a, size: size, seed: seed, rankBase: cumBits}
		l.buildRank()
		cumBits += l.wordRank[len(l.wordRank)-1] + uint64(bits.OnesCount64(a[len(a)-1]))
		m.levels = append(m.levels, l)

		next := remaining[:0:0]
		for _, k := range remaining {
			if getbit(c, hashKey(seed, k)%size) {
				next = append(next, k)
			}
		}
		remaining = next
	}
	// Anything the level chain could not place (the cap, or duplicates) gets an
	// explicit id past the placed range so the output stays a bijection over the
	// distinct keys.
	if len(remaining) > 0 {
		m.overflow = make(map[string]uint64, len(remaining))
		for _, k := range remaining {
			s := string(k)
			if _, dup := m.overflow[s]; dup {
				continue
			}
			m.overflow[s] = cumBits
			cumBits++
		}
	}
	m.n = cumBits
	return m
}

// Len is the number of keys, the size of the output range [0, Len).
func (m *MPH) Len() uint64 { return m.n }

// Lookup returns the index of key in [0, Len). The result is meaningful only for a
// key that was in the build set; a non-member returns some index in range, which is
// the MPH contract, so callers that take untrusted keys verify with a fingerprint.
func (m *MPH) Lookup(key []byte) uint64 {
	for li := range m.levels {
		l := &m.levels[li]
		i := hashKey(l.seed, key) % l.size
		if getbit(l.bits, i) {
			return l.rankBase + l.rank(i)
		}
	}
	if m.overflow != nil {
		if id, ok := m.overflow[string(key)]; ok {
			return id
		}
	}
	return 0
}

// Bits is the total size of the level bit arrays, the structure's storage cost
// before the overflow table, which is empty on a real key set.
func (m *MPH) Bits() uint64 {
	var t uint64
	for i := range m.levels {
		t += m.levels[i].size
	}
	return t
}

// BitsPerKey is the average structure size a key costs, the density measure BBHash
// is tuned for; near three at the default gamma.
func (m *MPH) BitsPerKey() float64 {
	if m.n == 0 {
		return 0
	}
	return float64(m.Bits()) / float64(m.n)
}

// buildRank fills the per-word cumulative popcount so rank is a word lookup plus
// one partial-word popcount.
func (l *level) buildRank() {
	l.wordRank = make([]uint64, len(l.bits))
	var c uint64
	for w := range l.bits {
		l.wordRank[w] = c
		c += uint64(bits.OnesCount64(l.bits[w]))
	}
}

// rank returns the number of set bits before position i in this level.
func (l *level) rank(i uint64) uint64 {
	w := i / 64
	off := i % 64
	r := l.wordRank[w]
	if off > 0 {
		r += uint64(bits.OnesCount64(l.bits[w] & ((uint64(1) << off) - 1)))
	}
	return r
}

func getbit(b []uint64, i uint64) bool { return b[i/64]&(uint64(1)<<(i%64)) != 0 }
func setbit(b []uint64, i uint64)      { b[i/64] |= uint64(1) << (i % 64) }

// levelSeed derives a per-level seed so the same key hashes independently at each
// level, which is what lets collided keys separate as they fall through.
func levelSeed(level int) uint64 {
	return 0x9e3779b97f4a7c15 * (uint64(level) + 1)
}

// hashKey is a seeded 64-bit hash: FNV-1a over the key bytes with the seed mixed
// into the basis, finished with the murmur3 64-bit finalizer for avalanche. The
// MPH only needs the hash to be deterministic and well-distributed; a poor hash
// costs extra levels, never correctness.
func hashKey(seed uint64, b []byte) uint64 {
	const (
		offset64 = 1469598103934665603
		prime64  = 1099511628211
	)
	h := offset64 ^ seed
	for _, c := range b {
		h ^= uint64(c)
		h *= prime64
	}
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	h ^= h >> 33
	h *= 0xc4ceb9fe1a85ec53
	h ^= h >> 33
	return h
}
