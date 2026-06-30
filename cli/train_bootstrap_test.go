package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadQueries(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "queries.txt")
	content := "" +
		"# a comment line is skipped\n" +
		"\n" +
		"plain query text\n" +
		"myid\ttab separated query\n" +
		"   spaced query   \n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	queries, err := readQueries(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(queries) != 3 {
		t.Fatalf("got %d queries, want 3: %v", len(queries), queries)
	}
	// The tab-separated line keeps its explicit id and trims both sides.
	if queries["myid"] != "tab separated query" {
		t.Fatalf("tab line parsed to %q", queries["myid"])
	}
	// A plain line takes a line-numbered id; line 3 is the first plain query.
	if queries["q0003"] != "plain query text" {
		t.Fatalf("plain line parsed to id q0003=%q, got %v", queries["q0003"], queries)
	}
	// The spaced line on line 5 is trimmed.
	if queries["q0005"] != "spaced query" {
		t.Fatalf("spaced line not trimmed: %q", queries["q0005"])
	}
}

func TestReadQueriesMissingFile(t *testing.T) {
	if _, err := readQueries(filepath.Join(t.TempDir(), "nope.txt")); err == nil {
		t.Fatal("want an error opening a missing query file, got nil")
	}
}
