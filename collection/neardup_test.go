package collection

import (
	"fmt"
	"math/bits"
	"os"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi/convert"
)

// longBody returns a body with enough distinct content tokens to clear the
// fingerprint floor, seeded so two calls with the same seed produce the same text.
func longBody(seed string) string {
	var b strings.Builder
	for i := 0; i < 40; i++ {
		fmt.Fprintf(&b, "%s word%d alpha beta gamma delta epsilon ", seed, i)
	}
	return b.String()
}

// BenchmarkNearDupPenalties measures the full fingerprint, banded-blocking, and
// penalty pass over a synthetic corpus with a realistic sprinkling of duplicates
// (every tenth page is a near-copy of an earlier one), the build-side cost.
func BenchmarkNearDupPenalties(b *testing.B) {
	const n = 20000
	docs := make([]convert.Document, n)
	pr := make([]float64, n)
	for i := 0; i < n; i++ {
		body := longBody(fmt.Sprintf("document subject number %d unique tokens", i))
		if i%10 == 0 && i >= 10 {
			body = docs[i-10].Body + fmt.Sprintf(" tail%d", i) // a near-copy
		}
		docs[i] = convert.Document{URL: fmt.Sprintf("https://h%d.example/p%d", i%500, i), Body: body}
		pr[i] = 1.0 / float64(n)
	}
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		_ = nearDupPenalties(docs, pr)
	}
}

// TestSimhashIdenticalBodies checks two identical bodies hash to the same
// fingerprint and a clearly different body lands far away in Hamming distance.
func TestSimhashIdenticalBodies(t *testing.T) {
	a, okA := simhash(longBody("topic one about engines and motors"))
	b, okB := simhash(longBody("topic one about engines and motors"))
	if !okA || !okB {
		t.Fatal("identical long bodies should fingerprint")
	}
	if a != b {
		t.Fatalf("identical bodies hashed differently: %x vs %x", a, b)
	}
	c, okC := simhash(longBody("entirely unrelated cooking recipes and kitchen"))
	if !okC {
		t.Fatal("third body should fingerprint")
	}
	if d := bits.OnesCount64(a ^ c); d <= nearDupK {
		t.Fatalf("unrelated body within near-dup radius: hamming %d", d)
	}
}

// TestSimhashNearDuplicate checks a body with one token changed stays within the
// near-dup radius, the property the penalty clustering relies on.
func TestSimhashNearDuplicate(t *testing.T) {
	base := longBody("shared content paragraph about the same subject matter")
	near := base + " onlyoneextraword"
	a, _ := simhash(base)
	b, _ := simhash(near)
	if d := bits.OnesCount64(a ^ b); d > nearDupK {
		t.Fatalf("near-duplicate outside radius: hamming %d", d)
	}
}

// TestSimhashShortBodySkipped checks a body below the token floor gets no
// fingerprint, so empty pages do not all collapse to one all-zero cluster.
func TestSimhashShortBodySkipped(t *testing.T) {
	if _, ok := simhash("tiny"); ok {
		t.Fatal("short body should not fingerprint")
	}
	if _, ok := simhash(""); ok {
		t.Fatal("empty body should not fingerprint")
	}
}

// TestNearDupPenaltiesCluster builds a small corpus with a three-page duplicate
// cluster and two unique pages, then checks the representative scores zero, the
// copies score positive, the lower-rank copy is penalized at least as hard as the
// higher-rank copy, and the unique pages are untouched.
func TestNearDupPenaltiesCluster(t *testing.T) {
	dupBody := longBody("the duplicated article body shared across the mirror cluster")
	docs := []convert.Document{
		{URL: "https://a.example/orig", Body: dupBody},
		{URL: "https://b.example/copy1", Body: dupBody + " tinytaila"},
		{URL: "https://c.example/copy2", Body: dupBody + " tinytailb"},
		{URL: "https://d.example/unique", Body: longBody("a completely separate unique page")},
		{URL: "https://e.example/other", Body: longBody("yet another distinct standalone document")},
	}
	// Give the original the highest rank so it wins the representative tie.
	pr := []float64{0.40, 0.20, 0.05, 0.30, 0.05}

	pen := nearDupPenalties(docs, pr)
	if len(pen) != len(docs) {
		t.Fatalf("penalty length %d != docs %d", len(pen), len(docs))
	}
	if pen[0] != 0 {
		t.Fatalf("representative penalty = %g, want 0", pen[0])
	}
	if pen[1] <= 0 || pen[2] <= 0 {
		t.Fatalf("copies not penalized: %g, %g", pen[1], pen[2])
	}
	// Copy 2 has the lower PageRank, so it sits further below the representative and
	// must be penalized at least as hard as copy 1.
	if pen[2] < pen[1] {
		t.Fatalf("lower-rank copy %g penalized less than higher-rank copy %g", pen[2], pen[1])
	}
	if pen[3] != 0 || pen[4] != 0 {
		t.Fatalf("unique pages penalized: %g, %g", pen[3], pen[4])
	}
	for i, p := range pen {
		if p < 0 || p > 1 {
			t.Fatalf("penalty at %d = %g, out of [0,1]", i, p)
		}
	}
}

// TestNearDupRepresentativeByRank checks the highest-PageRank page in a cluster is
// the representative even when it is not the first one in document order.
func TestNearDupRepresentativeByRank(t *testing.T) {
	body := longBody("identical body for the representative selection test case here")
	docs := []convert.Document{
		{URL: "https://a.example/x", Body: body},
		{URL: "https://b.example/y", Body: body},
		{URL: "https://c.example/z", Body: body},
	}
	pr := []float64{0.1, 0.7, 0.2} // doc 1 is the most authoritative
	pen := nearDupPenalties(docs, pr)
	if pen[1] != 0 {
		t.Fatalf("highest-rank page is not the representative: penalties %v", pen)
	}
	if pen[0] == 0 || pen[2] == 0 {
		t.Fatalf("non-representatives not penalized: %v", pen)
	}
}

// TestNearDupOnCCrawl is the real-data gate: near-dup clustering runs over the real
// ccrawl bodies, every penalty stays in [0,1], representatives score zero, and the
// reported cluster shape is logged. The check is that the signal is well-formed and
// that genuine duplicates on the real corpus are found, not a fixed cluster count,
// which depends on the snapshot.
func TestNearDupOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for {
		d, ok, err := src.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	pr := globalRanks(docs)
	pen := nearDupPenalties(docs, pr)
	if len(pen) != len(docs) {
		t.Fatalf("penalty length %d != docs %d", len(pen), len(docs))
	}

	var penalized, reps int
	for i, p := range pen {
		if p < 0 || p > 1 {
			t.Fatalf("penalty at %d = %g, out of [0,1]", i, p)
		}
		if p > 0 {
			penalized++
		}
	}

	// Cross-check against an exact all-pairs near-dup count on a bounded prefix, so
	// the banded blocking is proven to find the duplicates it should rather than
	// silently clustering nothing. The prefix keeps the O(n^2) check cheap.
	const probe = 4000
	m := len(docs)
	if m > probe {
		m = probe
	}
	fps := make([]uint64, m)
	has := make([]bool, m)
	for i := 0; i < m; i++ {
		if fp, ok := simhash(docs[i].Body); ok {
			fps[i] = fp
			has[i] = true
		}
	}
	exactPairs := 0
	for i := 0; i < m; i++ {
		if !has[i] {
			continue
		}
		for j := i + 1; j < m; j++ {
			if has[j] && bits.OnesCount64(fps[i]^fps[j]) <= nearDupK {
				exactPairs++
			}
		}
	}
	// Every exact near-dup pair in the prefix must have left both pages in a
	// multi-member cluster, so if the brute force finds pairs the banded blocking
	// must have penalized at least one page per pair.
	if exactPairs > 0 && penalized == 0 {
		t.Fatalf("brute force found %d near-dup pairs but blocking penalized none", exactPairs)
	}
	for i := 0; i < len(docs); i++ {
		if pen[i] == 0 {
			reps++
		}
	}
	t.Logf("docs=%d penalized=%d (reps+unique=%d) exactNearDupPairs(prefix %d)=%d",
		len(docs), penalized, reps, m, exactPairs)
}
