package cli

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/lexical"
)

// TestBuildImpactCCrawl proves the impact build end to end on real crawl data: a collection
// built with Options.Impact carries impact-ordered lexical regions whose per-document impact
// is the real composite static rank quantized, and the served impact traversal returns
// exactly the coverage-times-impact top-k. The oracle is recovered from the region itself
// through the public search API, so the gate rests on no private accessor: a single-term
// query returns every document that carries the term scored by its coverage of one times its
// impact, which is the impact byte, so the loop below rebuilds both the inverted index and
// the per-document impact from single-term queries, then checks multi-term queries against
// the coverage-times-impact oracle those recover.
func TestBuildImpactCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 2000, Limit: 6000, Impact: true, BuildEpoch: 1})
	if err != nil {
		t.Fatalf("impact build from ccrawl: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards, got %d", res.Shards)
	}

	infos, err := collection.List(out)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}

	checkedShards, spread := 0, false
	for _, in := range infos {
		reg := openShardLexical(t, in.Path)
		if !reg.IsImpact() {
			t.Fatalf("shard %s did not build impact-ordered", filepath.Base(in.Path))
		}
		docCount := int(reg.DocCount())

		// Recover the inverted index and the per-document impact from single-term queries.
		var vocab []string
		reg.ForEachTerm(func(term string, _ uint32) { vocab = append(vocab, term) })
		if len(vocab) < 8 {
			continue
		}
		inv := make(map[string]map[uint32]bool, len(vocab))
		impact := map[uint32]uint8{}
		for _, term := range vocab {
			cands, err := reg.SearchImpactTerms([]string{term}, docCount)
			if err != nil {
				t.Fatalf("single-term search %q: %v", term, err)
			}
			set := make(map[uint32]bool, len(cands))
			for _, c := range cands {
				if c.Score < 0 || c.Score > 255 {
					t.Fatalf("single-term score out of impact range: term %q doc %d score %d", term, c.DocID, c.Score)
				}
				set[c.DocID] = true
				impact[c.DocID] = uint8(c.Score)
			}
			inv[term] = set
		}

		// The impacts must spread across the byte range: a build feeding the real static
		// rank produces many distinct impacts, where a constant stand-in would collapse them.
		lo, hi := uint8(255), uint8(0)
		for _, v := range impact {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		if hi > lo {
			spread = true
		}

		// Draw multi-term queries from the recovered vocab and check the served impact
		// traversal against the coverage-times-impact oracle the recovery built.
		const k = 50
		for qi := 0; qi < 120; qi++ {
			n := 1 + (qi % 4)
			terms := make([]string, n)
			for i := range terms {
				terms[i] = vocab[(qi*7+i*101)%len(vocab)]
			}
			got, err := reg.SearchImpactTerms(terms, k)
			if err != nil {
				t.Fatalf("multi-term search %v: %v", terms, err)
			}
			want := coverageImpactOracle(terms, inv, impact, k)
			if len(got) != len(want) {
				t.Fatalf("shard %s query %v: got %d results, want %d", filepath.Base(in.Path), terms, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("shard %s query %v result %d: got %+v want %+v", filepath.Base(in.Path), terms, i, got[i], want[i])
				}
			}
		}
		checkedShards++
	}
	if checkedShards == 0 {
		t.Fatal("no shard had enough vocabulary to check")
	}
	if !spread {
		t.Fatal("impacts did not spread across the byte range; static rank may not be feeding the build")
	}
	t.Logf("impact build served the coverage-times-impact top-k over %d shards from real static rank", checkedShards)
}

// openShardLexical opens a shard file and returns its lexical region, failing the test if the
// shard has none, the reader a per-shard impact check reads through.
func openShardLexical(t *testing.T, path string) *lexical.Region {
	t.Helper()
	r, err := tsumugi.Open(path)
	if err != nil {
		t.Fatalf("open shard %s: %v", filepath.Base(path), err)
	}
	t.Cleanup(func() { _ = r.Close() })
	if !r.HasRegion(tsumugi.RegionLexical) {
		t.Fatalf("shard %s has no lexical region", filepath.Base(path))
	}
	b, err := r.Region(tsumugi.RegionLexical)
	if err != nil {
		t.Fatalf("read lexical region: %v", err)
	}
	reg, err := lexical.Open(b)
	if err != nil {
		t.Fatalf("open lexical region: %v", err)
	}
	return reg
}

// coverageImpactOracle computes the coverage-times-impact top-k the impact traversal serves:
// a document's score is the number of distinct query terms it carries times its impact, and
// the top-k is ordered score-descending then docID-ascending, the same order the region uses.
func coverageImpactOracle(terms []string, inv map[string]map[uint32]bool, impact map[uint32]uint8, k int) []lexical.Candidate {
	distinct := map[string]bool{}
	cov := map[uint32]int32{}
	for _, term := range terms {
		if distinct[term] {
			continue
		}
		distinct[term] = true
		for doc := range inv[term] {
			cov[doc]++
		}
	}
	var cands []lexical.Candidate
	for doc, c := range cov {
		cands = append(cands, lexical.Candidate{DocID: doc, Score: c * int32(impact[doc])})
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Score != cands[j].Score {
			return cands[i].Score > cands[j].Score
		}
		return cands[i].DocID < cands[j].DocID
	})
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands
}
