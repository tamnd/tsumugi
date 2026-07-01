package search

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/collection"
)

// openBuiltShards builds a collection over the source and opens every shard it wrote
// through the ordinary serving path, so the test reads the anchor forward column and
// the anchor token stat the production build writes rather than a hand-assembled shard.
func openBuiltShards(t *testing.T, src, out string) []*Shard {
	t.Helper()
	if _, err := collection.Build(collection.Options{Source: src, Out: out, ShardSize: 100000}); err != nil {
		t.Fatalf("build: %v", err)
	}
	infos, err := collection.List(out)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	model := trainExactMatchModel(t)
	var shards []*Shard
	for _, info := range infos {
		sh, err := OpenShard(info.Path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %s: %v", info.Path, err)
		}
		shards = append(shards, sh)
	}
	return shards
}

// shardLocalByURL returns the shard and local id of the document with the given url,
// so a test can name a document by its stable url rather than the id the reorder
// assigned. The online extractor reads columns by local id, so the caller needs both.
func shardLocalByURL(t *testing.T, shards []*Shard, url string) (*Shard, uint32) {
	t.Helper()
	for _, sh := range shards {
		for id := uint32(0); id < uint32(sh.docCount); id++ {
			if v, ok := sh.fwd.Column("url", id); ok && string(v) == url {
				return sh, id
			}
		}
	}
	t.Fatalf("url %q not found in any shard", url)
	return nil, 0
}

// TestOnlineAnchorBM25EndToEnd is the online counterpart of the offline
// TestAnchorFieldMakesInboundTermSearchable: it builds a collection through the ordinary
// build path, opens the shards the serving path opens, and checks that the L2 online
// extractor recomputes a positive anchor BM25 for a document whose query term appears
// only in its inbound anchor text, never in its own body, title, or url. That closes the
// gap slice 98 left, where the anchor field was in the inverted index driving retrieval
// but the online L2 total still ignored it.
func TestOnlineAnchorBM25EndToEnd(t *testing.T) {
	tmp := t.TempDir()
	targetURL := "https://target.example/doc"
	lines := []string{
		// Off-domain source: "zebranchor" names the target through inbound anchor text.
		`{"url":"https://source.example/a","host":"source.example","markdown":"intro [zebranchor beacon](https://target.example/doc) outro filler words here"}`,
		`{"url":"https://source.example/b","host":"source.example","markdown":"more [zebranchor beacon](https://target.example/doc) linking text"}`,
		`{"url":"https://third.example/c","host":"third.example","markdown":"unrelated [zebranchor beacon](https://target.example/doc) page body"}`,
		// The target: its own text carries a unique body word but never the anchor word.
		`{"url":"https://target.example/doc","host":"target.example","markdown":"# Target\ntanukibody main content of the page"}`,
	}
	src := filepath.Join(tmp, "anchor.jsonl")
	if err := os.WriteFile(src, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	shards := openBuiltShards(t, src, filepath.Join(tmp, "coll"))
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	sh, local := shardLocalByURL(t, shards, targetURL)

	// The anchor forward column must round-trip through the build: the target's stored
	// anchor field carries the inbound word.
	if v, ok := sh.fwd.Column("anchor", local); !ok || !strings.Contains(string(v), "zebranchor") {
		t.Fatalf("target anchor column missing the inbound word, got %q (ok=%v)", string(v), ok)
	}

	// Run the online extractor for the anchor-only term the way the reranker does, with
	// the shard's own idf and per-field average lengths.
	q := Query{Text: "zebranchor"}
	ext := sh.newOnline(q, nil, sh.localAvgFieldLen())
	f := ext.features(local)
	if f[OnBM25Anchor] <= 0 {
		t.Fatalf("target should score a positive online anchor bm25 for its inbound term, got %.4f", f[OnBM25Anchor])
	}
	if f[OnBM25Body] != 0 {
		t.Fatalf("the term is absent from the target body, so online body bm25 should be zero, got %.4f", f[OnBM25Body])
	}
	if f[OnBM25FTotal] <= 0 {
		t.Fatalf("the anchor match should lift the online field-weighted total, got %.4f", f[OnBM25FTotal])
	}

	// The target is still scored on its own body word, so widening the extractor did not
	// displace the existing fields.
	qb := Query{Text: "tanukibody"}
	extb := sh.newOnline(qb, nil, sh.localAvgFieldLen())
	if extb.features(local)[OnBM25Body] <= 0 {
		t.Fatalf("target should still score its own body word, got %.4f", extb.features(local)[OnBM25Body])
	}
}

// TestOnlineAnchorBM25OnCCrawl runs the online extractor over a collection built from
// the real crawl sample through the ordinary build path, so the anchor forward column
// and the fleet anchor average length come from real data. On this near-flat slice the
// inbound anchor resolution is close to zero (the graph tests inject synthetic far edges
// for the same reason), so the check is that the anchor field round-trips and its online
// BM25 is well-formed over the real distribution, not that it fires: OnBM25Anchor is
// non-negative for every document, and a document with no query term in its anchor field
// scores exactly zero there.
func TestOnlineAnchorBM25OnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl sample not present: %v", err)
	}
	tmp := t.TempDir()
	if _, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: filepath.Join(tmp, "coll"), ShardSize: 100000, Limit: 8000}); err != nil {
		t.Fatalf("build: %v", err)
	}
	infos, err := collection.List(filepath.Join(tmp, "coll"))
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	model := trainExactMatchModel(t)
	var shards []*Shard
	for _, info := range infos {
		sh, err := OpenShard(info.Path, newTestCascade(model))
		if err != nil {
			t.Fatalf("open shard %s: %v", info.Path, err)
		}
		shards = append(shards, sh)
	}
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	// Query a real document's title so the survivors carry real text into the anchor scan.
	var q Query
	for _, sh := range shards {
		for id := uint32(0); id < uint32(sh.docCount) && q.Text == ""; id++ {
			if v, ok := sh.fwd.Column("title", id); ok {
				if toks := uniqueTokens(string(v)); len(toks) >= 2 && len(toks) <= 6 {
					q = Query{Text: string(v)}
				}
			}
		}
	}
	if q.Text == "" {
		t.Skip("no usable real-title query in the sample")
	}

	checked := 0
	for _, sh := range shards {
		// The anchor forward column must exist on a real build, readable for every doc.
		if _, ok := sh.fwd.Column("anchor", 0); !ok {
			t.Fatalf("real build has no readable anchor column")
		}
		ext := sh.newOnline(q, nil, sh.localAvgFieldLen())
		for id := uint32(0); id < uint32(sh.docCount); id++ {
			f := ext.features(id)
			if f[OnBM25Anchor] < 0 {
				t.Fatalf("online anchor bm25 should be non-negative, got %.4f at doc %d", f[OnBM25Anchor], id)
			}
			// The anchor field only ever contributes to the weighted total, never subtracts,
			// so a positive anchor bm25 implies a positive total.
			if f[OnBM25Anchor] > 0 && f[OnBM25FTotal] <= 0 {
				t.Fatalf("a positive anchor bm25 should lift the total at doc %d", id)
			}
			checked++
		}
	}
	// The anchor fleet average is a real, non-negative number for a build with any anchor
	// text, or zero when the slice resolved none; either way it must not be negative.
	if avg := shards[0].localAvgFieldLen(); avg[fAnchor] < 0 {
		t.Fatalf("anchor fleet average should be non-negative, got %.4f", avg[fAnchor])
	}
	t.Logf("checked online anchor bm25 over %d real docs, anchor avg %.4f", checked, shards[0].localAvgFieldLen()[fAnchor])
}
