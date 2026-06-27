package lexical_test

import (
	"fmt"
	"sort"
	"testing"

	"github.com/tamnd/tsumugi/lexical"
)

// The M13 claim is that scoring every shard against one collection-wide idf, gathered
// across the shards, reproduces the ranking a single index over all the documents would
// produce. These tests prove it at the lexical layer, where the idf is the only moving
// part: the documents are built with identical per-field token counts so every shard and
// the combined index share the same average field lengths, which leaves the inverse
// document frequency as the sole cross-shard difference. That isolation is deliberate.
// The idf factors out of the BM25F contribution linearly, so the build can store it out
// of the block-max bound and the query can scale by whichever idf it is handed; the
// length-normalization average does not factor out that way and stays shard-local by
// design, so holding it constant is what makes the comparison a clean test of the idf
// path rather than a test of two things at once.

// idfDoc is one document with a fixed shape: each field carries a fixed number of tokens
// so the average field length is the same constant in every region built from these docs,
// in a shard and in the combined index alike.
type idfDoc struct {
	title string
	body  string
}

func (d idfDoc) fields() map[lexical.Field]string {
	return map[lexical.Field]string{
		lexical.FieldTitle: d.title,
		lexical.FieldBody:  d.body,
	}
}

// buildRegion builds a lexical region over docs assigned dense ids start, start+1, ...
func buildRegion(t *testing.T, docs []idfDoc, start uint32) *lexical.Region {
	t.Helper()
	b := lexical.NewBuilder(lexical.DefaultParams())
	for i, d := range docs {
		b.AddDoc(start+uint32(i), d.fields())
	}
	r, err := lexical.Open(b.Build())
	if err != nil {
		t.Fatalf("open region: %v", err)
	}
	return r
}

// gatherGlobalIDF sums each query term's document frequency across the shards and turns
// it into the collection-wide idf, the value the broker pushes down. n is the total
// document count across the shards, the N a single index would divide by.
func gatherGlobalIDF(query string, n uint64, shards ...*lexical.Region) map[string]float64 {
	df := map[string]uint32{}
	for _, r := range shards {
		for t, f := range r.DocFreqs(query) {
			df[t] += f
		}
	}
	idf := make(map[string]float64, len(df))
	for t, f := range df {
		idf[t] = lexical.IDF(n, uint64(f))
	}
	return idf
}

// shifted is a candidate moved into the combined-index id space by its shard's base.
type shifted struct {
	id    uint32
	score int32
}

// mergeShards runs each shard's search and merges the results into one list ordered the
// way a single index orders its top-k: descending score, then ascending global id. bases
// gives each shard's first global id, so a shard's local ids land where the combined
// index placed those same documents.
func mergeShards(t *testing.T, query string, k int, idfOf map[string]float64, regions []*lexical.Region, bases []uint32) []shifted {
	t.Helper()
	var all []shifted
	for si, r := range regions {
		cands, err := r.SearchWithIDF(query, k, idfOf)
		if err != nil {
			t.Fatalf("search shard %d: %v", si, err)
		}
		for _, c := range cands {
			all = append(all, shifted{id: bases[si] + c.DocID, score: c.Score})
		}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].id < all[j].id
	})
	if len(all) > k {
		all = all[:k]
	}
	return all
}

// combinedTopK is the single-index reference: the top-k a region over every document
// produces, in the same shifted shape as a merge.
func combinedTopK(t *testing.T, combined *lexical.Region, query string, k int) []shifted {
	t.Helper()
	cands, err := combined.Search(query, k)
	if err != nil {
		t.Fatalf("search combined: %v", err)
	}
	out := make([]shifted, len(cands))
	for i, c := range cands {
		out[i] = shifted{id: c.DocID, score: c.Score}
	}
	return out
}

func sameShifted(a, b []shifted) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// skewedCorpus builds two shards where a term is common in one shard but rare across the
// collection, the case shard-local idf gets wrong. group1 fills shard A, group2 fills
// shard B, and the combined index is group1 followed by group2, so a shard's local id i
// is the combined id base+i.
func skewedCorpus() (group1, group2 []idfDoc) {
	// "alpha" is dense in shard A and absent from shard B, so its shard-local idf in A
	// is computed against A's 40 documents while its true collection idf is against 80.
	for i := 0; i < 40; i++ {
		body := "common common common alpha rare"
		if i >= 36 {
			body = "common common common beta rare"
		}
		group1 = append(group1, idfDoc{title: fmt.Sprintf("doc a%d", i), body: body})
	}
	// Shard B never mentions alpha, so the term is rare across the whole collection even
	// though it saturates shard A.
	for i := 0; i < 40; i++ {
		body := "common common common gamma rare"
		if i >= 38 {
			body = "common common common beta rare"
		}
		group2 = append(group2, idfDoc{title: fmt.Sprintf("doc b%d", i), body: body})
	}
	return group1, group2
}

// TestGlobalIDFMatchesCombinedIndex proves the merge of per-shard searches scored with
// the gathered collection-wide idf equals the single combined index's top-k, document id
// and integer score, across a spread of queries.
func TestGlobalIDFMatchesCombinedIndex(t *testing.T) {
	group1, group2 := skewedCorpus()
	all := append(append([]idfDoc{}, group1...), group2...)
	n := uint64(len(all))

	shardA := buildRegion(t, group1, 0)
	shardB := buildRegion(t, group2, 0)
	combined := buildRegion(t, all, 0)
	regions := []*lexical.Region{shardA, shardB}
	bases := []uint32{0, uint32(len(group1))}

	k := len(all)
	queries := []string{
		"alpha", "beta", "gamma", "rare", "common",
		"alpha rare", "common beta", "alpha beta gamma", "missingterm", "alpha missingterm",
	}
	for _, q := range queries {
		idfOf := gatherGlobalIDF(q, n, shardA, shardB)
		got := mergeShards(t, q, k, idfOf, regions, bases)
		want := combinedTopK(t, combined, q, k)
		if !sameShifted(got, want) {
			t.Errorf("query %q: global-idf merge does not match combined index\n got %v\nwant %v", q, got, want)
		}
	}
}

// TestShardLocalIDFDivergesFromCombined is the negative control: without the gathered
// idf, scoring each shard against its local statistics, the merge disagrees with the
// combined index on the skewed term. It proves the fix is load-bearing and not masking a
// corpus where local and global idf happen to coincide.
func TestShardLocalIDFDivergesFromCombined(t *testing.T) {
	group1, group2 := skewedCorpus()
	all := append(append([]idfDoc{}, group1...), group2...)

	shardA := buildRegion(t, group1, 0)
	shardB := buildRegion(t, group2, 0)
	combined := buildRegion(t, all, 0)
	regions := []*lexical.Region{shardA, shardB}
	bases := []uint32{0, uint32(len(group1))}

	k := len(all)
	// A multi-term query that mixes the shard-dense term with a shard-rare term is where
	// local idf misranks: shard A overweights alpha because it sees it as common only
	// locally, so a local-idf merge orders alpha documents differently than the combined
	// index does.
	q := "alpha rare beta"
	localMerge := mergeShards(t, q, k, nil, regions, bases)
	want := combinedTopK(t, combined, q, k)
	if sameShifted(localMerge, want) {
		t.Fatalf("expected shard-local idf merge to diverge from the combined index for %q, but they matched; the corpus is not exercising the skew", q)
	}

	// And the gathered global idf closes exactly that gap.
	idfOf := gatherGlobalIDF(q, uint64(len(all)), shardA, shardB)
	globalMerge := mergeShards(t, q, k, idfOf, regions, bases)
	if !sameShifted(globalMerge, want) {
		t.Fatalf("global-idf merge should match the combined index for %q after the local-idf merge diverged", q)
	}
}
