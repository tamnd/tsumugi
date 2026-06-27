package collection

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/klauspost/compress/zstd"
	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/codec"
	"github.com/tamnd/tsumugi/lexical"
)

// The artifact body is zstd compressed on disk. The routing index is the bulk of it and
// is a sorted run of terms with small delta-coded shard lists, which compresses well, so
// compression is what keeps the artifact from growing to a sizeable fraction of the
// shards it indexes. EncodeAll and DecodeAll are safe for concurrent use, so a single
// shared encoder and decoder serve every build and load.
var (
	indexEnc, _ = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	indexDec, _ = zstd.NewReader(nil)
)

// IndexName is the collection-level artifact a build writes next to the shards. It
// holds the manifest, the fleet-wide statistics, and the routing index, so a serve
// reads one file instead of opening and scanning every shard to rebuild them. At
// 100,000 shards that startup scan would read every term dictionary in the fleet,
// which does not hold, so the artifact is what lets serve start in time proportional
// to the query vocabulary rather than the corpus.
const IndexName = "index.tsm"

// indexMagic marks the artifact, distinct from a shard's TSM1.
const indexMagic = "TSMI"

const indexVersion = 1

// Stats are the fleet-wide collection statistics, summed across every shard. A single
// shard describes only its slice of the collection, so any scoring that normalizes by
// a collection-level term, the average document length above all, reads these rather
// than any one shard's numbers.
type Stats struct {
	DocCount   uint64
	TokenCount float64
	AvgDocLen  float64
}

// Index is a loaded collection artifact: the manifest of shards, the fleet-wide
// statistics, and the routing index that maps a term to the shards that hold it. It is
// read-only after loading and safe for concurrent reads.
type Index struct {
	BuildEpoch uint64
	Stats      Stats
	Shards     []ShardInfo

	// routing maps a term to the shard indices, into Shards, that carry a posting for
	// it. A query routes to the union of its terms' shards, so fan-out is proportional
	// to the query vocabulary, not the fleet size.
	routing map[string][]int32
	// always lists the shards that must see every query because their vocabulary could
	// not be enumerated, the impact-quantized shards whose dictionary the artifact does
	// not walk. Routing always includes these so no candidate is missed.
	always []int32
	numShd int
}

// BuildEpoch sentinel passed to WriteIndex when the caller has no clock to stamp the
// artifact with; the artifact stays valid, it just carries a zero epoch.
const NoEpoch uint64 = 0

// WriteIndex scans a collection's shards once and writes the index artifact. It reads
// each shard's node base, document count, size, and token count for the manifest and
// the statistics, and walks each lexical shard's term dictionary for the routing
// index. The write is atomic: it renders into a temporary file and renames it into
// place, so a reader only ever sees a complete artifact. An impact-quantized shard,
// whose dictionary this scan does not enumerate, is recorded as an always-routed shard
// so routing never drops a candidate it cannot see.
func WriteIndex(dir string, epoch uint64) error {
	infos, err := List(dir)
	if err != nil {
		return err
	}
	if len(infos) == 0 {
		return fmt.Errorf("collection: no shards in %s to index", dir)
	}

	ix := &Index{
		BuildEpoch: epoch,
		Shards:     infos,
		routing:    make(map[string][]int32),
		numShd:     len(infos),
	}
	for si, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			return fmt.Errorf("open %s: %w", filepath.Base(info.Path), err)
		}
		ix.Stats.DocCount += uint64(r.DocCount())
		if v, ok := r.Stat(tsumugi.StatTokenCount); ok {
			ix.Stats.TokenCount += v
		}
		if err := ix.indexShardTerms(r, si); err != nil {
			_ = r.Close()
			return fmt.Errorf("index %s: %w", filepath.Base(info.Path), err)
		}
		_ = r.Close()
	}
	if ix.Stats.DocCount > 0 {
		ix.Stats.AvgDocLen = ix.Stats.TokenCount / float64(ix.Stats.DocCount)
	}

	buf := ix.encode()
	tmp := filepath.Join(dir, IndexName+".tmp")
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, IndexName))
}

// indexShardTerms records, for one shard, the terms it routes for. A classic BM25
// shard has its vocabulary walked term by term; an impact-quantized shard, whose
// dictionary the artifact does not enumerate, is added to the always-routed set so it
// still sees every query.
func (ix *Index) indexShardTerms(r *tsumugi.Reader, si int) error {
	if !r.HasRegion(tsumugi.RegionLexical) {
		return nil
	}
	if r.Header.Has(tsumugi.FlagImpactPostings) {
		ix.always = append(ix.always, int32(si))
		return nil
	}
	b, err := r.Region(tsumugi.RegionLexical)
	if err != nil {
		return err
	}
	reg, err := lexical.Open(b)
	if err != nil {
		return err
	}
	n := reg.TermCount()
	for id := uint32(0); id < n; id++ {
		term, ok := reg.Term(id)
		if !ok {
			continue
		}
		lst := ix.routing[term]
		if len(lst) == 0 || lst[len(lst)-1] != int32(si) {
			ix.routing[term] = append(lst, int32(si))
		}
	}
	return nil
}

// LoadIndex reads a collection's index artifact. It returns os.ErrNotExist, wrapped,
// when the artifact is absent, so a caller can fall back to scanning the shards.
func LoadIndex(dir string) (*Index, error) {
	b, err := os.ReadFile(filepath.Join(dir, IndexName))
	if err != nil {
		return nil, err
	}
	return decodeIndex(b)
}

// Route returns the shard indices a query over the given analyzed terms should fan out
// to: the union of the shards holding any term, plus the always-routed shards. A query
// with no terms routes everywhere, since the routing index says nothing about which
// shards carry relevant vectors or impacts.
func (ix *Index) Route(terms []string) []int {
	if len(terms) == 0 {
		return ix.allShards()
	}
	seen := make([]bool, ix.numShd)
	out := make([]int, 0, 16)
	add := func(si int32) {
		if !seen[si] {
			seen[si] = true
			out = append(out, int(si))
		}
	}
	for _, si := range ix.always {
		add(si)
	}
	for _, t := range terms {
		for _, si := range ix.routing[t] {
			add(si)
		}
	}
	return out
}

func (ix *Index) allShards() []int {
	out := make([]int, ix.numShd)
	for i := range out {
		out[i] = i
	}
	return out
}

// NumShards is the number of shards the artifact describes.
func (ix *Index) NumShards() int { return ix.numShd }

// RoutingMap returns the term-to-shards routing as a plain map, the shape the search
// broker's routing index loads from. The shard ids are the manifest order, so they line
// up with shards opened in that order. RoutingMap carries only the per-term lists; the
// shards that must see every query are reported separately by AlwaysRouted.
func (ix *Index) RoutingMap() map[string][]int {
	out := make(map[string][]int, len(ix.routing))
	for t, lst := range ix.routing {
		ids := make([]int, len(lst))
		for i, si := range lst {
			ids[i] = int(si)
		}
		out[t] = ids
	}
	return out
}

// AlwaysRouted returns the shards that must see every query because their vocabulary
// was not enumerable at index time, the impact-quantized shards.
func (ix *Index) AlwaysRouted() []int {
	out := make([]int, len(ix.always))
	for i, si := range ix.always {
		out[i] = int(si)
	}
	return out
}

// encode renders the artifact to bytes: a small fixed frame of magic, version, and the
// uncompressed body length, then the zstd-compressed body, closed by a CRC over
// everything before it. The body holds the build epoch, the fleet-wide statistics, the
// shard manifest, and the routing index. The routing terms are written in sorted order
// so the body, and therefore the whole artifact, is byte-identical for the same
// collection regardless of map iteration order.
func (ix *Index) encode() []byte {
	body := ix.encodeBody()
	comp := indexEnc.EncodeAll(body, nil)

	b := make([]byte, 0, len(comp)+24)
	b = append(b, indexMagic...)
	b = codec.AppendUint32(b, indexVersion)
	b = codec.AppendUint64(b, uint64(len(body)))
	b = append(b, comp...)
	b = codec.AppendUint32(b, codec.CRC32C(b))
	return b
}

// encodeBody renders the uncompressed artifact body, the part compression and the CRC
// cover.
func (ix *Index) encodeBody() []byte {
	b := make([]byte, 0, 1<<16)
	b = codec.AppendUint64(b, ix.BuildEpoch)

	b = codec.AppendUint64(b, ix.Stats.DocCount)
	b = codec.AppendUint64(b, math.Float64bits(ix.Stats.TokenCount))
	b = codec.AppendUint64(b, math.Float64bits(ix.Stats.AvgDocLen))

	b = codec.AppendUint32(b, uint32(len(ix.Shards)))
	for _, s := range ix.Shards {
		name := filepath.Base(s.Path)
		b = codec.AppendUvarint(b, uint64(len(name)))
		b = append(b, name...)
		b = codec.AppendUint32(b, s.NodeBase)
		b = codec.AppendUint32(b, s.DocCount)
		b = codec.AppendUint64(b, uint64(s.Bytes))
	}

	b = codec.AppendUint32(b, uint32(len(ix.always)))
	for _, si := range ix.always {
		b = codec.AppendUint32(b, uint32(si))
	}

	terms := make([]string, 0, len(ix.routing))
	for t := range ix.routing {
		terms = append(terms, t)
	}
	sort.Strings(terms)
	b = codec.AppendUint32(b, uint32(len(terms)))
	for _, t := range terms {
		b = codec.AppendUvarint(b, uint64(len(t)))
		b = append(b, t...)
		lst := ix.routing[t]
		b = codec.AppendUvarint(b, uint64(len(lst)))
		var prev int32 = -1
		for _, si := range lst {
			b = codec.AppendUvarint(b, uint64(si-prev-1))
			prev = si
		}
	}
	return b
}

var errBadIndex = errors.New("collection: corrupt index artifact")

// decodeIndex parses an artifact rendered by encode, verifying the magic, the version,
// and the trailing CRC, then decompressing the body, before reading the fields, so a
// torn or stale artifact is refused rather than read as garbage.
func decodeIndex(b []byte) (*Index, error) {
	const frame = 4 + 4 + 8 // magic + version + body length
	if len(b) < frame+4 || string(b[0:4]) != indexMagic {
		return nil, errBadIndex
	}
	if codec.Uint32(b[len(b)-4:]) != codec.CRC32C(b[:len(b)-4]) {
		return nil, errBadIndex
	}
	if codec.Uint32(b[4:]) != indexVersion {
		return nil, errBadIndex
	}
	bodyLen := codec.Uint64(b[8:])
	body, err := indexDec.DecodeAll(b[frame:len(b)-4], make([]byte, 0, bodyLen))
	if err != nil || uint64(len(body)) != bodyLen {
		return nil, errBadIndex
	}

	p := &parser{b: body, off: 0}
	ix := &Index{routing: make(map[string][]int32)}
	ix.BuildEpoch = p.u64()
	ix.Stats.DocCount = p.u64()
	ix.Stats.TokenCount = math.Float64frombits(p.u64())
	ix.Stats.AvgDocLen = math.Float64frombits(p.u64())

	nShard := int(p.u32())
	ix.Shards = make([]ShardInfo, nShard)
	for i := 0; i < nShard; i++ {
		name := p.str()
		ix.Shards[i] = ShardInfo{
			Path:     name,
			NodeBase: p.u32(),
			DocCount: p.u32(),
			Bytes:    int64(p.u64()),
		}
	}
	ix.numShd = nShard

	nAlways := int(p.u32())
	ix.always = make([]int32, nAlways)
	for i := 0; i < nAlways; i++ {
		ix.always[i] = int32(p.u32())
	}

	nTerms := int(p.u32())
	for i := 0; i < nTerms; i++ {
		t := p.str()
		nl := int(p.uvarint())
		lst := make([]int32, nl)
		var prev int32 = -1
		for j := 0; j < nl; j++ {
			prev += int32(p.uvarint()) + 1
			lst[j] = prev
		}
		ix.routing[t] = lst
	}
	if p.err {
		return nil, errBadIndex
	}
	return ix, nil
}

// parser walks a byte slice, tracking an overrun in a sticky error flag so the decode
// path stays free of per-read error handling and a truncated artifact fails the final
// check rather than panicking on a slice bound.
type parser struct {
	b   []byte
	off int
	err bool
}

func (p *parser) u32() uint32 {
	if p.off+4 > len(p.b) {
		p.err = true
		return 0
	}
	v := codec.Uint32(p.b[p.off:])
	p.off += 4
	return v
}

func (p *parser) u64() uint64 {
	if p.off+8 > len(p.b) {
		p.err = true
		return 0
	}
	v := codec.Uint64(p.b[p.off:])
	p.off += 8
	return v
}

func (p *parser) uvarint() uint64 {
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

func (p *parser) str() string {
	n := int(p.uvarint())
	if n < 0 || p.off+n > len(p.b) {
		p.err = true
		return ""
	}
	s := string(p.b[p.off : p.off+n])
	p.off += n
	return s
}
