package spell

import (
	"fmt"
	"testing"
)

// synthDict builds a dictionary of n distinct lowercase terms with descending
// frequencies, a stand-in for a corpus vocabulary when measuring the corrector's cost
// independent of the index build.
func synthDict(n int) []TermFreq {
	const alpha = "abcdefghijklmnopqrstuvwxyz"
	terms := make([]TermFreq, 0, n)
	for i := 0; i < n; i++ {
		// A base-26 expansion of i padded to six characters gives n distinct terms of a
		// realistic word length.
		x := i
		var b [6]byte
		for k := 5; k >= 0; k-- {
			b[k] = alpha[x%26]
			x /= 26
		}
		terms = append(terms, TermFreq{Term: string(b[:]), Freq: uint64(n - i)})
	}
	return terms
}

// BenchmarkLookup measures one query-term correction against a resident index, the hot
// path cost the spec budgets at single-digit microseconds: a few dozen hash probes into
// the delete index plus a few short Damerau-Levenshtein verifications.
func BenchmarkLookup(b *testing.B) {
	for _, n := range []int{10_000, 100_000} {
		ix := Build(synthDict(n), 2)
		b.Run(fmt.Sprintf("terms=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				// A near-miss of an existing term, the case correction actually runs on.
				if s := ix.Lookup("abcdeg"); len(s) == 0 {
					b.Fatal("no candidates")
				}
			}
		})
	}
}

// BenchmarkBuild measures the offline delete-index construction, the once-per-collection
// cost paid at the broker so the online lookup is hash probes only.
func BenchmarkBuild(b *testing.B) {
	dict := synthDict(50_000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if ix := Build(dict, 2); ix.Len() == 0 {
			b.Fatal("empty index")
		}
	}
}
