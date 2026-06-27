package mph

// Dir is a compact canonical-URL to node-id directory built on a minimal perfect
// hash. It replaces the build's plain map[string]int with a structure that costs a
// few bits a key for the hash plus a fixed fingerprint and value a key, instead of
// the map's tens of bytes a key of string storage and bucket overhead, which is
// what the two-billion-URL scale needs.
//
// An MPH alone cannot tell a member from a non-member, and the link directory must,
// because most link targets are pages the crawl never captured. Dir pairs each slot
// with a 64-bit fingerprint of its key, so a lookup confirms the key actually owns
// the slot before trusting the value; a non-member lands on some member's slot and
// is rejected when the fingerprints disagree, with a false-positive rate of one in
// two-to-the-sixty-four, which is zero for any real corpus.
type Dir struct {
	mph *MPH
	fp  []uint64
	val []uint32
}

// BuildDir constructs the directory from parallel url and id slices. A url that
// appears more than once keeps its first id, the same first-occurrence rule the
// build's map used, so the directory resolves a repeated canonical URL to one node.
func BuildDir(urls [][]byte, ids []uint32, gamma float64) *Dir {
	m := Build(urls, gamma)
	d := &Dir{mph: m, fp: make([]uint64, m.Len()), val: make([]uint32, m.Len())}
	seen := make([]bool, m.Len())
	for i, u := range urls {
		slot := m.Lookup(u)
		if seen[slot] {
			continue
		}
		seen[slot] = true
		d.fp[slot] = fingerprint(u)
		d.val[slot] = ids[i]
	}
	return d
}

// Lookup returns the node id stored for url and whether url is a member of the
// directory. A non-member returns false, which is how the link resolver tells a
// link to a crawled page from a link to a page the crawl never saw.
func (d *Dir) Lookup(url []byte) (uint32, bool) {
	slot := d.mph.Lookup(url)
	if slot >= uint64(len(d.fp)) || d.fp[slot] != fingerprint(url) {
		return 0, false
	}
	return d.val[slot], true
}

// Len is the number of distinct keys the directory holds.
func (d *Dir) Len() uint64 { return d.mph.Len() }

// BitsPerKey is the minimal perfect hash's own density, the spec's headline number;
// the fingerprint and value arrays add a fixed per-key cost on top.
func (d *Dir) BitsPerKey() float64 { return d.mph.BitsPerKey() }

// fingerprint is an independent 64-bit hash of the key for the membership check. It
// uses a seed disjoint from any level seed so it does not correlate with where the
// MPH placed the key.
func fingerprint(key []byte) uint64 {
	return hashKey(0xd1b54a32d192ed03, key)
}
