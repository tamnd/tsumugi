package cli

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestInspectSizes runs the inspect command with --sizes over a small shard and checks the
// report carries the doc 13 compactness columns: a per-region bytes-per-document number and
// the budget cell the RegionStats helper supplies, plus a total line. It is the CLI-side
// gate that the --sizes view is wired to the production budget table, complementing the
// helper's own unit test.
func TestInspectSizes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.tsumugi")
	writeShard(t, path, []string{
		"the brown bear forages in the forest",
		"a brown bear sleeps through winter",
		"brown bear tracks cross the fresh snow",
	}, 0)

	cmd := newInspectCmd()
	var out strings.Builder
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--sizes", path})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("inspect --sizes: %v", err)
	}
	got := out.String()

	for _, want := range []string{"sizes:", "bytes/doc", "budget", "lexical", "feature", "total"} {
		if !strings.Contains(got, want) {
			t.Errorf("inspect --sizes output missing %q:\n%s", want, got)
		}
	}
	// The lexical region carries a doc 13 budget cell, so its row must show the range.
	if !strings.Contains(got, "120-260") {
		t.Errorf("inspect --sizes missing the lexical budget cell:\n%s", got)
	}
}

// TestInspectDefaultNoSizes checks the default inspect view still prints the original
// regions table and does not emit the sizes report unless --sizes is passed.
func TestInspectDefaultNoSizes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.tsumugi")
	writeShard(t, path, []string{"brown bear forages"}, 0)

	cmd := newInspectCmd()
	var out strings.Builder
	cmd.SetOut(&out)
	cmd.SetArgs([]string{path})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("inspect: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "regions:") {
		t.Errorf("default inspect missing the regions table:\n%s", got)
	}
	if strings.Contains(got, "bytes/doc") {
		t.Errorf("default inspect should not print the sizes report:\n%s", got)
	}
}
