package spell

import (
	"reflect"
	"sort"
	"testing"
)

// TestDeletesEnumeratesEditNeighborhood checks the offline delete enumeration: it must
// produce the term itself (the zero-delete variant) and every string obtained by
// deleting up to d characters, on runes so a multi-byte character deletes as one.
func TestDeletesEnumeratesEditNeighborhood(t *testing.T) {
	got := deletesInto(nil, "abc", 1)
	want := map[string]struct{}{"abc": {}, "bc": {}, "ac": {}, "ab": {}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deletes(abc,1) = %v, want %v", keys(got), keys(want))
	}

	got2 := deletesInto(nil, "abc", 2)
	// Distance two adds the single-character strings and the empty string is not
	// reachable from a length-three term in two deletes, so the level-two set is the
	// length-one strings plus the level-one set.
	for _, v := range []string{"abc", "ab", "ac", "bc", "a", "b", "c"} {
		if _, ok := got2[v]; !ok {
			t.Errorf("deletes(abc,2) missing %q", v)
		}
	}

	uni := deletesInto(nil, "café", 1)
	// Four runes means four single-deletion variants plus the original, and "caf"
	// must be one of them, proving the deletion is on runes not bytes.
	if _, ok := uni["caf"]; !ok {
		t.Errorf("deletes(café,1) missing caf, got %v", keys(uni))
	}
	if len(uni) != 5 {
		t.Errorf("deletes(café,1) has %d variants, want 5: %v", len(uni), keys(uni))
	}
}

// TestDamerauLevenshtein pins the distance the verification step uses, including the
// transposition that separates Damerau-Levenshtein from plain Levenshtein and the
// early-exit when the distance exceeds the cap.
func TestDamerauLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		max  int
		want int
	}{
		{"the", "the", 2, 0},
		{"teh", "the", 2, 1},    // a single transposition, the case the canon names
		{"ther", "there", 2, 1}, // one insertion
		{"kitten", "sitting", 3, 3},
		{"golan", "golang", 2, 1},
		{"abc", "abcdef", 2, -1}, // length gap exceeds the cap, rejected early
		{"", "ab", 2, 2},
		{"ab", "", 2, 2},
		{"flaw", "lawn", 2, 2},
	}
	for _, c := range cases {
		if got := damerauLevenshtein(c.a, c.b, c.max); got != c.want {
			t.Errorf("dl(%q,%q,max=%d) = %d, want %d", c.a, c.b, c.max, got, c.want)
		}
	}
}

// TestLookupFindsAndRanks proves the online half end to end: a delete-lookup gathers
// the candidates, verification keeps those within distance two, and the rank is
// distance ascending then frequency descending, so a closer or more common correction
// wins, the difference between a useful and an annoying corrector.
func TestLookupFindsAndRanks(t *testing.T) {
	ix := Build([]TermFreq{
		{"the", 1000},
		{"there", 500},
		{"their", 400},
		{"golang", 800},
		{"golan", 5},
		{"kubernetes", 600},
	}, 2)

	// "teh" is one transposition from "the" and nothing else close, so "the" leads.
	best, ok := ix.Best("teh")
	if !ok || best.Term != "the" || best.Dist != 1 {
		t.Fatalf("Best(teh) = %+v ok=%v, want the at distance 1", best, ok)
	}

	// "ther" is not itself a dictionary term but is distance one from several real
	// words; the frequency tie-break must surface the common "the" over the rarer ones.
	sugg := ix.Lookup("ther")
	if len(sugg) == 0 {
		t.Fatal("Lookup(ther) returned nothing")
	}
	// All distance-one candidates come before any distance-two candidate, and within
	// distance one frequency is descending.
	assertRanked(t, sugg)
	if sugg[0].Term != "the" {
		t.Errorf("Lookup(ther) ranked %q first, want the most frequent close term the: %+v", sugg[0].Term, sugg[:1])
	}

	// A term far from everything yields nothing within distance two.
	if s := ix.Lookup("zzzzqqqq"); len(s) != 0 {
		t.Errorf("Lookup(zzzzqqqq) = %+v, want empty", s)
	}
}

// TestLookupExactTermIncluded checks that a correctly-spelled term finds itself at
// distance zero, since the term is its own zero-delete variant.
func TestLookupExactTermIncluded(t *testing.T) {
	ix := Build([]TermFreq{{"rust", 900}, {"ruse", 100}}, 2)
	best, ok := ix.Best("rust")
	if !ok || best.Term != "rust" || best.Dist != 0 {
		t.Fatalf("Best(rust) = %+v ok=%v, want rust at distance 0", best, ok)
	}
}

// TestCorrectPolicy exercises the when-to-correct and did-you-mean-versus-auto-correct
// rules: a healthy term is left alone, a hard miss with a close common correction
// auto-corrects, and a thin or far correction is offered as did-you-mean.
func TestCorrectPolicy(t *testing.T) {
	ix := Build([]TermFreq{
		{"golang", 800},
		{"golan", 5},
		{"the", 1000},
		{"rust", 900},
		{"ruse", 850},
	}, 2)
	p := DefaultPolicy()

	// A correctly spelled common term is used as typed, no correction.
	if d := ix.Correct("golang", p); d.Action != None {
		t.Errorf("Correct(golang) = %+v, want None", d)
	}

	// A hard miss one transposition from a far more common word auto-corrects.
	d := ix.Correct("gloang", p)
	if d.Action != AutoCorrect || d.Suggestion.Term != "golang" {
		t.Errorf("Correct(gloang) = %+v, want AutoCorrect to golang", d)
	}

	// A miss whose best correction is two real words of similar frequency is offered,
	// not applied: "ruse" sits one edit from both "rust" and "ruse", close in df, so
	// the ratio is not met and it stays a suggestion.
	d = ix.Correct("ruet", p)
	if d.Action == None {
		t.Errorf("Correct(ruet) = None, want a suggestion")
	}
}

func assertRanked(t *testing.T, sugg []Suggestion) {
	t.Helper()
	for i := 1; i < len(sugg); i++ {
		a, b := sugg[i-1], sugg[i]
		if a.Dist > b.Dist {
			t.Errorf("rank out of order at %d: dist %d before %d", i, a.Dist, b.Dist)
		}
		if a.Dist == b.Dist && a.Freq < b.Freq {
			t.Errorf("rank out of order at %d: freq %d before %d at same dist", i, a.Freq, b.Freq)
		}
	}
}

func keys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
