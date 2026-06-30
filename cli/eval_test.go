package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFile drops content at a temp path and returns it, the fixture the eval command
// reads its run and qrels from.
func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return p
}

func TestEvalCmdScoresRunAgainstQrels(t *testing.T) {
	dir := t.TempDir()
	// Two queries, the relevant document ranked first in each, a perfect ranking.
	runPath := writeFile(t, dir, "run.txt",
		"q1 Q0 d1 1 5.0 sys\nq1 Q0 d2 2 1.0 sys\nq2 Q0 d3 1 5.0 sys\nq2 Q0 d4 2 1.0 sys\n")
	qrelsPath := writeFile(t, dir, "qrels.txt",
		"q1 0 d1 3\nq1 0 d2 0\nq2 0 d3 3\nq2 0 d4 0\n")

	cmd := newEvalCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--run", runPath, "--qrels", qrelsPath, "--ndcg", "10", "--recall", "100"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "2 scored") {
		t.Errorf("expected 2 scored queries:\n%s", got)
	}
	if !strings.Contains(got, "NDCG@10:   1.0000") {
		t.Errorf("a perfect ranking should report NDCG@10 1.0:\n%s", got)
	}
}

func TestEvalCmdGoldAgreement(t *testing.T) {
	dir := t.TempDir()
	runPath := writeFile(t, dir, "run.txt", "q1 Q0 d1 1 5.0 sys\n")
	// The qrels (the judge's labels) match the gold exactly, so the agreement passes.
	qrelsPath := writeFile(t, dir, "qrels.txt",
		"q1 0 d1 3\nq1 0 d2 0\nq2 0 d3 2\nq2 0 d4 1\nq3 0 d5 3\nq3 0 d6 0\n")
	goldPath := writeFile(t, dir, "gold.txt",
		"q1 0 d1 3\nq1 0 d2 0\nq2 0 d3 2\nq2 0 d4 1\nq3 0 d5 3\nq3 0 d6 0\n")

	cmd := newEvalCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--run", runPath, "--qrels", qrelsPath, "--gold", goldPath})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "gold-set agreement") {
		t.Errorf("expected the agreement section:\n%s", got)
	}
	if !strings.Contains(got, "pass") {
		t.Errorf("matching labels should pass the trust gate:\n%s", got)
	}
	if !strings.Contains(got, "confusion") {
		t.Errorf("expected the confusion matrix:\n%s", got)
	}
}

func TestEvalCmdRequiresFiles(t *testing.T) {
	cmd := newEvalCmd()
	cmd.SetArgs([]string{"--run", "only-run.txt"})
	cmd.SilenceUsage = true
	cmd.SilenceErrors = true
	if err := cmd.Execute(); err == nil {
		t.Fatal("eval without --qrels must error")
	}
}
