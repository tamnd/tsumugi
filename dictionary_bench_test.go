package tsumugi

import (
	"bytes"
	"fmt"
	"path/filepath"
	"testing"
)

// dictColumn builds a url-like column and a dictionary derived from it, the shape
// the shared-dictionary mechanism is meant for.
func dictColumn(rows int) (column []byte, dict []byte, samples [][]byte) {
	samples = make([][]byte, rows)
	for i := 0; i < rows; i++ {
		u := []byte(fmt.Sprintf("https://host%d.example.com/section/page-%d?ref=feed&id=%d", i%64, i, i))
		samples[i] = u
		column = append(column, u...)
		column = append(column, '\n')
	}
	dict = DeriveDictionary(samples, 16<<10)
	return column, dict, samples
}

// BenchmarkDictRegionWrite times compressing a column against a shared dictionary,
// the build-time cost of the mechanism.
func BenchmarkDictRegionWrite(b *testing.B) {
	column, dict, _ := dictColumn(20000)
	dir := b.TempDir()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(dir, fmt.Sprintf("w%d.tsumugi", i))
		w, err := Create(path)
		if err != nil {
			b.Fatal(err)
		}
		if err := w.AddDictionary(1, dict); err != nil {
			b.Fatal(err)
		}
		if err := w.AddRegion(RegionForward, CodecZstdDict, 0, 1, column); err != nil {
			b.Fatal(err)
		}
		if err := w.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDictRegionRead times Open plus decoding a dictionary-compressed region,
// the serving hot path that has to stay well inside the 10ms shard budget.
func BenchmarkDictRegionRead(b *testing.B) {
	column, dict, _ := dictColumn(20000)
	path := filepath.Join(b.TempDir(), "r.tsumugi")
	w, err := Create(path)
	if err != nil {
		b.Fatal(err)
	}
	if err := w.AddDictionary(1, dict); err != nil {
		b.Fatal(err)
	}
	if err := w.AddRegion(RegionForward, CodecZstdDict, 0, 1, column); err != nil {
		b.Fatal(err)
	}
	if err := w.Close(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := Open(path)
		if err != nil {
			b.Fatal(err)
		}
		got, err := r.Region(RegionForward)
		if err != nil {
			b.Fatal(err)
		}
		if !bytes.Equal(got, column) {
			b.Fatal("mismatch")
		}
		_ = r.Close()
	}
}
