package expand

import (
	"reflect"
	"testing"

	"github.com/tamnd/tsumugi/lexical"
)

// an is the analysis chain the tests build the table with: the default lexical analyzer,
// lowercase-and-tokenize, the same chain the build and the query share.
var an = lexical.Analyze

// TestBuildKeysSingleTokenForms checks the core build rule: a single-token form keys
// onto the other forms in its group, and a multi-token form does not get its own key
// because it cannot be matched against one analyzed query term.
func TestBuildKeysSingleTokenForms(t *testing.T) {
	tab := Build([]Group{{"nyc", "new york city"}}, an)

	if got := tab.Expand("nyc"); !reflect.DeepEqual(got, []string{"new york city"}) {
		t.Errorf("Expand(nyc) = %v, want [new york city]", got)
	}
	// The multi-word side is not a key: a three-token phrase is not one query term.
	if got := tab.Expand("new york city"); got != nil {
		t.Errorf("Expand(new york city) = %v, want nil (multi-token form is not a key)", got)
	}
	// A term not in the table expands to nothing, the common case.
	if got := tab.Expand("golang"); got != nil {
		t.Errorf("Expand(golang) = %v, want nil", got)
	}
}

// TestBuildBidirectionalSingleTokens checks that a group of two single-token forms is
// bidirectional, the synonym case: "color" finds "colour" and the reverse.
func TestBuildBidirectionalSingleTokens(t *testing.T) {
	tab := Build([]Group{{"color", "colour"}}, an)
	if got := tab.Expand("color"); !reflect.DeepEqual(got, []string{"colour"}) {
		t.Errorf("Expand(color) = %v, want [colour]", got)
	}
	if got := tab.Expand("colour"); !reflect.DeepEqual(got, []string{"color"}) {
		t.Errorf("Expand(colour) = %v, want [color]", got)
	}
}

// TestBuildMergesAndSortsAcrossGroups checks that a key appearing in two groups merges
// its alternatives, deduplicated and sorted for a stable expansion.
func TestBuildMergesAndSortsAcrossGroups(t *testing.T) {
	tab := Build([]Group{
		{"js", "javascript"},
		{"js", "ecmascript"},
		{"js", "javascript"}, // duplicate form, must not double
	}, an)
	got := tab.Expand("js")
	want := []string{"ecmascript", "javascript"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Expand(js) = %v, want %v (merged, deduped, sorted)", got, want)
	}
}

// TestBuildSkretchesDegenerateGroups checks that a group that analyzes to fewer than two
// distinct forms contributes no entry, so an empty or single-form group is a no-op.
func TestBuildSkipsDegenerateGroups(t *testing.T) {
	tab := Build([]Group{
		{"solo"},       // one form, nothing to expand to
		{"!!!", "???"}, // both analyze to nothing
		{"dup", "dup"}, // two forms, one distinct
		{"x", "y"},     // a real pair, to prove the table is otherwise built
	}, an)
	if tab.Expand("solo") != nil || tab.Expand("dup") != nil {
		t.Error("a group with fewer than two distinct analyzed forms must contribute no entry")
	}
	if got := tab.Expand("x"); !reflect.DeepEqual(got, []string{"y"}) {
		t.Errorf("Expand(x) = %v, want [y]", got)
	}
}

// TestDefaultTable checks the shipped curated table expands a representative acronym and
// a representative spelling variant, and leaves an ambiguous short word it deliberately
// omits alone.
func TestDefaultTable(t *testing.T) {
	tab := Default(an)
	if tab.Len() == 0 {
		t.Fatal("default table is empty")
	}
	if got := tab.Expand("nyc"); !reflect.DeepEqual(got, []string{"new york city"}) {
		t.Errorf("Expand(nyc) = %v, want [new york city]", got)
	}
	if got := tab.Expand("k8s"); !reflect.DeepEqual(got, []string{"kubernetes"}) {
		t.Errorf("Expand(k8s) = %v, want [kubernetes]", got)
	}
	if got := tab.Expand("gray"); !reflect.DeepEqual(got, []string{"grey"}) {
		t.Errorf("Expand(gray) = %v, want [grey]", got)
	}
	// "us" is deliberately not in the table: it collides with the pronoun, so a wrong
	// expansion would cost more than the missing one.
	if got := tab.Expand("us"); got != nil {
		t.Errorf("Expand(us) = %v, want nil (ambiguous short acronym omitted)", got)
	}
}

// TestNilTable checks the zero value and a nil table are safe to query, so a broker with
// no expansion configured calls through without a guard.
func TestNilTable(t *testing.T) {
	var tab *Table
	if tab.Expand("nyc") != nil || tab.Len() != 0 {
		t.Error("a nil table must expand to nothing and report length zero")
	}
}
