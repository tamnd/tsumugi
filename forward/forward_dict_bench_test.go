package forward

import (
	"fmt"
	"testing"
)

// benchColumn builds a forward region of n url-like and title-like rows under
// per-value shared-dictionary compression, the encoding collection writes.
func benchColumn(n int) []byte {
	b := NewBuilder([]Column{
		{Name: "url", Type: ColString, Codec: CodecZstdDict},
		{Name: "title", Type: ColString, Codec: CodecZstdDict},
	})
	for i := 0; i < n; i++ {
		b.Set(uint32(i), "url", []byte(fmt.Sprintf("https://www.example.com/articles/2026/06/%06d/index.html", i)))
		b.Set(uint32(i), "title", []byte(fmt.Sprintf("Example article number %d about a familiar topic", i)))
	}
	return b.Build()
}

// BenchmarkForwardDictWrite measures the per-value dictionary build cost for a
// 20k-row two-column region, the offline build path.
func BenchmarkForwardDictWrite(b *testing.B) {
	const n = 20000
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = benchColumn(n)
	}
}

// BenchmarkForwardDictRead measures a single random-access decode, the online L2
// path: open once, then read one value per iteration so the timing is the per-
// value decode against the shared dictionary, not the open.
func BenchmarkForwardDictRead(b *testing.B) {
	const n = 20000
	r, err := Open(benchColumn(n))
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := uint32(i % n)
		if v, ok := r.Column("url", id); !ok || len(v) == 0 {
			b.Fatalf("read %d failed", id)
		}
	}
}
