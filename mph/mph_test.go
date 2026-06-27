package mph

import (
	"fmt"
	"os"
	"testing"

	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
)

const ccrawlParquet = "/Users/apple/data/ccrawl/markdown/CC-MAIN-2026-25/000000.parquet"

// assertBijection checks the core MPH property: every key maps to a distinct index
// and the indices exactly fill [0, len(keys)), with no gap and no collision.
func assertBijection(t *testing.T, m *MPH, keys [][]byte) {
	t.Helper()
	if m.Len() != uint64(len(keys)) {
		t.Fatalf("Len = %d, want %d", m.Len(), len(keys))
	}
	seen := make([]bool, len(keys))
	for _, k := range keys {
		idx := m.Lookup(k)
		if idx >= uint64(len(keys)) {
			t.Fatalf("index %d out of range [0,%d) for key %q", idx, len(keys), k)
		}
		if seen[idx] {
			t.Fatalf("index %d collided: two keys map to it", idx)
		}
		seen[idx] = true
	}
	for i, ok := range seen {
		if !ok {
			t.Fatalf("index %d was never produced; the map has a gap", i)
		}
	}
}

func TestMPHBijectionSynthetic(t *testing.T) {
	for _, n := range []int{0, 1, 2, 100, 10000} {
		keys := make([][]byte, n)
		for i := range keys {
			keys[i] = []byte(fmt.Sprintf("https://host%d.example/page/%d", i%997, i))
		}
		m := Build(keys, DefaultGamma)
		assertBijection(t, m, keys)
	}
}

func TestMPHDuplicatesGoToOverflow(t *testing.T) {
	// Duplicate keys never separate across levels; the overflow table must still
	// keep the output a bijection over the distinct keys.
	keys := [][]byte{[]byte("a"), []byte("a"), []byte("b")}
	m := Build(keys, DefaultGamma)
	// Two distinct keys, so the range is [0,2) and a and b get distinct indices.
	if m.Len() != 2 {
		t.Fatalf("Len = %d, want 2 distinct keys", m.Len())
	}
	if m.Lookup([]byte("a")) == m.Lookup([]byte("b")) {
		t.Fatal("distinct keys a and b collided")
	}
}

func TestDirRejectsNonMembers(t *testing.T) {
	var urls [][]byte
	var ids []uint32
	for i := 0; i < 5000; i++ {
		urls = append(urls, []byte(fmt.Sprintf("https://site%d.example/a", i)))
		ids = append(ids, uint32(i))
	}
	d := BuildDir(urls, ids, DefaultGamma)
	for i, u := range urls {
		got, ok := d.Lookup(u)
		if !ok || got != uint32(i) {
			t.Fatalf("member %q resolved to (%d,%v), want (%d,true)", u, got, ok, i)
		}
	}
	// None of these were inserted; every one must be rejected.
	var falsePos int
	for i := 0; i < 50000; i++ {
		if _, ok := d.Lookup([]byte(fmt.Sprintf("https://absent%d.example/x", i))); ok {
			falsePos++
		}
	}
	if falsePos != 0 {
		t.Fatalf("%d non-members passed the membership check; the 64-bit fingerprint should reject all", falsePos)
	}
}

// TestMPHOnCCrawl builds the MPH over the real corpus's distinct canonical URLs,
// proves the bijection on real keys, and records the bits-per-key density the spec
// is tuned for.
func TestMPHOnCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	src, err := convert.OpenSource(ccrawlParquet)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	seen := map[string]struct{}{}
	var keys [][]byte
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
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			continue
		}
		if _, dup := seen[cu]; dup {
			continue
		}
		seen[cu] = struct{}{}
		keys = append(keys, []byte(cu))
	}
	_ = src.Close()
	if len(keys) == 0 {
		t.Skip("no canonical URLs in parquet")
	}

	m := Build(keys, DefaultGamma)
	assertBijection(t, m, keys)
	t.Logf("keys=%d bits/key=%.3f levels=%d", m.Len(), m.BitsPerKey(), len(m.levels))
}
