// Package collection builds and manages a directory of .tsumugi shards as one served
// collection. It is the orchestration layer the CLI drives: Build turns a crawl
// export into shards, Add brings a later crawl into an existing collection, List
// reports what is there, and Compact merges shards back down. Every operation rests
// on the file format's single-write-moment immutability, so a shard is only ever
// written once and never mutated, which is what makes add and compact safe to run
// against a live directory.
package collection

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/tamnd/tsumugi"
)

// shardGlob matches the shard files a collection holds.
const shardGlob = "shard-*.tsumugi"

// docColumns is the forward-store schema every shard carries: the url and title for
// display, and the body so a shard holds the text it was built from, which is what
// lets Compact rebuild merged shards from the documents alone.
func docColumns() []forwardColumn {
	return []forwardColumn{
		{Name: "url"},
		{Name: "title"},
		{Name: "body", Blob: true},
	}
}

// ShardInfo is one shard's place in a collection: its file, the global id of its
// first document, and its document count.
type ShardInfo struct {
	Path     string
	NodeBase uint32
	DocCount uint32
	Bytes    int64
}

// List returns the shards in a collection directory ordered by node base, the global
// document order, so the listing reads as the collection's id layout.
func List(dir string) ([]ShardInfo, error) {
	paths, err := filepath.Glob(filepath.Join(dir, shardGlob))
	if err != nil {
		return nil, err
	}
	infos := make([]ShardInfo, 0, len(paths))
	for _, p := range paths {
		r, err := tsumugi.Open(p)
		if err != nil {
			return nil, fmt.Errorf("open %s: %w", filepath.Base(p), err)
		}
		st, err := os.Stat(p)
		if err != nil {
			_ = r.Close()
			return nil, err
		}
		infos = append(infos, ShardInfo{
			Path:     p,
			NodeBase: uint32(r.Header.NodeBase),
			DocCount: r.DocCount(),
			Bytes:    st.Size(),
		})
		_ = r.Close()
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].NodeBase < infos[j].NodeBase })
	return infos, nil
}

// nextBase returns the next free global document id in a collection, the id a new
// shard's first document takes, computed as the highest existing base plus its count.
// A fresh collection starts at zero.
func nextBase(dir string) (uint32, error) {
	infos, err := List(dir)
	if err != nil {
		return 0, err
	}
	var next uint32
	for _, s := range infos {
		if end := s.NodeBase + s.DocCount; end > next {
			next = end
		}
	}
	return next, nil
}

// nextIndex returns the next shard file index in a collection, so Add names its
// shards after the existing ones rather than colliding with them.
func nextIndex(dir string) (int, error) {
	paths, err := filepath.Glob(filepath.Join(dir, shardGlob))
	if err != nil {
		return 0, err
	}
	return len(paths), nil
}

func shardPath(dir string, index int) string {
	return filepath.Join(dir, fmt.Sprintf("shard-%05d.tsumugi", index))
}
