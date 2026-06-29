package collection

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/codec"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/mph"
)

// recrawlName is the collection-level re-crawl membership directory, a sibling of the
// shards. It is doc 02's minimal perfect hash over the collection's canonical URLs,
// the few-bits-a-key map a later crawl checks to tell a page it already holds from a
// new one without rescanning every shard's forward store. At 100,000 shards a per-add
// rescan to learn the collection's membership reads every shard, which does not hold;
// the directory is loaded once and answers each membership check in a hash probe.
const recrawlName = "recrawl.tsm"

// recrawlMagic marks the artifact, distinct from a shard's TSM1 and the index's TSMI.
const recrawlMagic = "TSMD"

// recrawlVersion is bumped when the on-disk form changes so a stale artifact triggers
// a rebuild rather than a misread.
const recrawlVersion = 1

// recrawlPath returns the membership directory's path inside a collection directory.
func recrawlPath(dir string) string { return filepath.Join(dir, recrawlName) }

// RecrawlDir is a loaded re-crawl membership directory. It answers, for a raw URL,
// whether the collection already holds the page that URL names (by its canonical
// form) and the global document id it was assigned. It is read-only after loading and
// safe for concurrent reads.
type RecrawlDir struct {
	dir *mph.Dir
}

// Lookup reports whether the collection holds the page raw names and, if so, its
// global document id. The URL is canonicalized first, so two spellings of one page
// (a trailing slash, a tracking parameter) resolve to the same identity the build
// stored, which is what makes the membership check fold aliases the way doc 02's
// canonical identity requires. A URL with no canonical form is never a member.
func (rd *RecrawlDir) Lookup(rawURL string) (uint32, bool) {
	cu, ok := analyze.CanonicalURL(rawURL)
	if !ok {
		return 0, false
	}
	return rd.dir.Lookup([]byte(cu))
}

// Len is the number of distinct pages the directory holds.
func (rd *RecrawlDir) Len() uint64 { return rd.dir.Len() }

// WriteRecrawlDir scans a collection's shards once, reads every document's canonical
// URL from its forward store, and writes the membership directory keyed on those URLs
// with each one's global document id as the value. The write is atomic: it renders
// into a temporary file and renames it into place, so a reader only ever sees a
// complete artifact. It is called after a build or an add so the directory always
// reflects the union of every shard in the directory, not just the slice a single add
// wrote. The scan reads forward stores rather than reusing the build's in-memory
// directory because an add's in-memory directory covers only the new crawl, while the
// recrawl check must see the whole collection.
func WriteRecrawlDir(dir string) error {
	infos, err := List(dir)
	if err != nil {
		return err
	}
	if len(infos) == 0 {
		return fmt.Errorf("collection: no shards in %s to index", dir)
	}

	var urls [][]byte
	var ids []uint32
	for _, info := range infos {
		r, err := tsumugi.Open(info.Path)
		if err != nil {
			return fmt.Errorf("open %s: %w", filepath.Base(info.Path), err)
		}
		b, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			_ = r.Close()
			return fmt.Errorf("forward region %s: %w", filepath.Base(info.Path), err)
		}
		fwd, err := forward.Open(b)
		if err != nil {
			_ = r.Close()
			return fmt.Errorf("open forward %s: %w", filepath.Base(info.Path), err)
		}
		for id := uint32(0); id < fwd.DocCount(); id++ {
			raw, _ := fwd.Column("url", id)
			cu, ok := analyze.CanonicalURL(string(raw))
			if !ok {
				continue
			}
			urls = append(urls, []byte(cu))
			// The value is the global document id, the shard's node base plus the row,
			// the handle a recrawl uses to find the page it already holds.
			ids = append(ids, info.NodeBase+id)
		}
		_ = r.Close()
	}

	d := mph.BuildDir(urls, ids, mph.DefaultGamma)
	buf := encodeRecrawl(d)
	tmp := recrawlPath(dir) + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, recrawlPath(dir))
}

// LoadRecrawlDir reads a collection's membership directory. It returns a wrapped
// os.ErrNotExist when the artifact is absent, so a caller (an add against a collection
// built before this artifact existed) can fall back to skipping the cross-crawl dedup
// rather than failing.
func LoadRecrawlDir(dir string) (*RecrawlDir, error) {
	b, err := os.ReadFile(recrawlPath(dir))
	if err != nil {
		return nil, err
	}
	return decodeRecrawl(b)
}

// encodeRecrawl frames the directory: a magic and version, the serialized directory,
// and a trailing CRC over everything before it, so a torn or stale artifact is refused
// rather than read as garbage. The directory bytes are not compressed: the fingerprint
// array is one independent 64-bit hash a key and does not compress, and the value array
// is small next to it, so compression would buy little for the decode cost it adds.
func encodeRecrawl(d *mph.Dir) []byte {
	b := make([]byte, 0, 1<<16)
	b = append(b, recrawlMagic...)
	b = codec.AppendUint32(b, recrawlVersion)
	b = d.Append(b)
	b = codec.AppendUint32(b, codec.CRC32C(b))
	return b
}

// decodeRecrawl parses an artifact rendered by encodeRecrawl, verifying the magic, the
// version, and the trailing CRC before reading the directory.
func decodeRecrawl(b []byte) (*RecrawlDir, error) {
	const frame = 4 + 4 // magic + version
	if len(b) < frame+4 || string(b[0:4]) != recrawlMagic {
		return nil, errBadRecrawl
	}
	if codec.Uint32(b[len(b)-4:]) != codec.CRC32C(b[:len(b)-4]) {
		return nil, errBadRecrawl
	}
	if codec.Uint32(b[4:]) != recrawlVersion {
		return nil, errBadRecrawl
	}
	d, n, err := mph.ReadDir(b[frame : len(b)-4])
	if err != nil {
		return nil, errBadRecrawl
	}
	if n != len(b)-4-frame {
		return nil, errBadRecrawl
	}
	return &RecrawlDir{dir: d}, nil
}

var errBadRecrawl = fmt.Errorf("collection: corrupt recrawl directory")

// dropRecrawled removes from docs every document the collection already holds, keyed
// on doc 02's canonical URL identity through the loaded membership directory. It is the
// cross-crawl half of the dedup the intra-crawl dedupByIdentity does not cover: a later
// crawl that re-fetched a page the collection already indexed drops the re-fetch rather
// than building a second copy of the page. The existing copy stays because the shards
// are immutable, which is what makes an add safe to run against a live directory; an
// add never rewrites a shard, so the freshest content of an already-held page waits for
// a compaction rather than landing as a duplicate. A document whose URL has no
// canonical form has no identity to match and is kept. The survivors stay in their
// input order so the build that follows is reproducible. The second return is the count
// dropped, for the caller to report.
func dropRecrawled(docs []convert.Document, rd *RecrawlDir) ([]convert.Document, int) {
	out := make([]convert.Document, 0, len(docs))
	dropped := 0
	for _, d := range docs {
		if _, held := rd.Lookup(d.URL); held {
			dropped++
			continue
		}
		out = append(out, d)
	}
	return out, dropped
}

// countHosts counts the distinct hosts in docs, the figure a build reports after the
// cross-crawl dedup has changed which documents survive.
func countHosts(docs []convert.Document) int {
	hosts := map[string]struct{}{}
	for _, d := range docs {
		hosts[d.Host] = struct{}{}
	}
	return len(hosts)
}
