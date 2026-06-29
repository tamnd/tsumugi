package collection

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/langid"
)

// TestLanguageIDsOnCCrawl runs the detection over the real crawl export and checks the
// categorical id column is well-formed and reads the corpus. Every id is one the langid
// table assigns; a placed page carries a nonzero id and an unplaceable one carries zero;
// English dominates this mostly-English web sample but is not the only language, so more
// than one nonzero id appears; and the set of id-unknown pages is exactly the set the
// consistency signal scores neutral, the invariant that the two signals read one shared
// detection pass rather than two that could drift.
func TestLanguageIDsOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	docs := readCCrawlDocs(t, 8000)
	if len(docs) == 0 {
		t.Skip("no documents in parquet")
	}

	lang, confident := detectLanguages(docs, langid.New())
	ids := languageIDsFrom(lang, confident)
	cons := languageConsistencyFrom(docs, lang, confident)

	// Every id must be a known one (the table only mints ids the model knows).
	known := map[uint32]bool{0: true}
	for _, l := range []string{"en", "es", "fr", "de", "it", "pt", "nl", "zh", "ja", "ko", "ru", "ar", "he", "hi", "th", "el"} {
		known[uint32(langid.LanguageID(l))] = true
	}

	var unknown int
	distinct := map[uint32]int{}
	for i, id := range ids {
		if !known[id] {
			t.Fatalf("doc %d carries id %d, not in the language table", i, id)
		}
		// id-unknown exactly when the page was not placed, which is exactly neutral consistency.
		placed := id != 0
		neutral := cons[i] == languageNeutral
		if placed == neutral {
			t.Fatalf("doc %d id=%d (placed=%v) but consistency neutral=%v; the two signals disagree on placement", i, id, placed, neutral)
		}
		if id == 0 {
			unknown++
		} else {
			distinct[id]++
		}
	}

	if len(distinct) < 2 {
		t.Fatalf("only %d distinct nonzero language ids on a broad sample; the column is not reading the language", len(distinct))
	}
	en := uint32(langid.LanguageID("en"))
	if top := plurality(distinct); top != en {
		t.Fatalf("plurality language id is %d, want English %d, on a mostly-English web sample", top, en)
	}
	if unknown == 0 {
		t.Fatal("no id-unknown page; the confidence floor is not gating short or mixed pages")
	}
	t.Logf("docs=%d placed=%d unknown=%d distinctLangs=%d english=%d", len(docs), total(distinct), unknown, len(distinct), distinct[en])
}

func total(m map[uint32]int) int {
	var n int
	for _, v := range m {
		n += v
	}
	return n
}

// BenchmarkLanguageIDs measures the language-id column build: the shared detection pass
// plus the id mapping, over a real sample. The detection dominates and is the same pass
// the consistency signal reads, so the marginal cost of the id column over the consistency
// signal is just the table lookup the mapping adds.
func BenchmarkLanguageIDs(b *testing.B) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		b.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlGraphParquet)
	if err != nil {
		b.Fatalf("open source: %v", err)
	}
	var docs []convert.Document
	for len(docs) < 2000 {
		d, ok, err := src.Next()
		if err != nil || !ok {
			break
		}
		docs = append(docs, d)
	}
	_ = src.Close()
	if len(docs) == 0 {
		b.Skip("no documents in parquet")
	}
	det := langid.New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lang, confident := detectLanguages(docs, det)
		_ = languageIDsFrom(lang, confident)
	}
}

// plurality returns the id with the largest count, ties broken on the smaller id so the
// result is deterministic. It is the most common language among the placed pages.
func plurality(m map[uint32]int) uint32 {
	var best uint32
	var bestN int
	for id, n := range m {
		if n > bestN || (n == bestN && id < best) {
			best, bestN = id, n
		}
	}
	return best
}

// TestLanguageColumnBuiltOnCCrawl builds a shard from the real export and checks the
// FeatLanguage column in the persisted feature region holds the real categorical ids the
// detection produced, not the old latin-ratio stand-in: every stored value is an integer
// id the language table mints, more than one language appears, and English is the
// plurality. This is the end-to-end gate that the categorical column survives the build,
// the quantizer, and the round-trip through the .tsumugi shard.
func TestLanguageColumnBuiltOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlGraphParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "col")
	res, err := Build(Options{Source: ccrawlGraphParquet, Out: out, ShardSize: 100000, Limit: 8000})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	shards, err := List(out)
	if err != nil {
		t.Fatalf("list shards: %v", err)
	}
	r, err := tsumugi.Open(shards[0].Path)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer func() { _ = r.Close() }()
	fb, err := r.Region(tsumugi.RegionFeature)
	if err != nil {
		t.Fatalf("feature region: %v", err)
	}
	fr, err := feature.Open(fb)
	if err != nil {
		t.Fatalf("open feature region: %v", err)
	}

	known := map[uint32]bool{0: true}
	for _, l := range []string{"en", "es", "fr", "de", "it", "pt", "nl", "zh", "ja", "ko", "ru", "ar", "he", "hi", "th", "el"} {
		known[uint32(langid.LanguageID(l))] = true
	}
	distinct := map[uint32]int{}
	for doc := 0; doc < res.Docs; doc++ {
		v, ok := fr.Value(uint32(doc), feature.FeatLanguage)
		if !ok {
			t.Fatalf("doc %d has no language value", doc)
		}
		if v != float64(uint32(v)) {
			t.Fatalf("doc %d language value %g is not an integer id", doc, v)
		}
		id := uint32(v)
		if !known[id] {
			t.Fatalf("doc %d stored id %d not in the language table", doc, id)
		}
		if id != 0 {
			distinct[id]++
		}
	}
	if len(distinct) < 2 {
		t.Fatalf("built language column has only %d distinct languages; categorical id did not survive the build", len(distinct))
	}
	en := uint32(langid.LanguageID("en"))
	if top := plurality(distinct); top != en {
		t.Fatalf("plurality language id in the built column is %d, want English %d", top, en)
	}
	t.Logf("docs=%d distinctLangs=%d english=%d", res.Docs, len(distinct), distinct[en])
}
