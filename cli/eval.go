package cli

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"github.com/tamnd/tsumugi/eval"
)

func newEvalCmd() *cobra.Command {
	var (
		runPath   string
		qrelsPath string
		goldPath  string
		ndcgArg   string
		recallArg string
	)
	cmd := &cobra.Command{
		Use:   "eval",
		Short: "Score a run against qrels and report NDCG, MRR, and recall",
		Long: "eval reads a TREC run and a graded qrels file, joins them per query, and reports\n" +
			"NDCG at the page cutoffs, MRR, and recall at the deep cutoffs, the quality numbers\n" +
			"doc 14's gates are read off. With a gold qrels subset it also runs the gold-set\n" +
			"agreement check, the Kendall tau and per-grade confusion matrix that say whether the\n" +
			"qrels' labels are trustworthy, since a quality number over bad labels is worse than\n" +
			"no number.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if runPath == "" || qrelsPath == "" {
				return fmt.Errorf("a run and a qrels file are required: pass --run and --qrels")
			}
			ndcg, err := parseCutoffs(ndcgArg)
			if err != nil {
				return fmt.Errorf("--ndcg: %w", err)
			}
			recall, err := parseCutoffs(recallArg)
			if err != nil {
				return fmt.Errorf("--recall: %w", err)
			}
			run, err := readRun(runPath)
			if err != nil {
				return err
			}
			qrels, err := readQrels(qrelsPath)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			rep := eval.Evaluate(run, qrels, ndcg, recall)
			_, _ = fmt.Fprintf(out, "queries: %d scored, %d skipped (no relevant judgment)\n", rep.NumQueries, rep.NumSkipped)
			for _, k := range ndcg {
				_, _ = fmt.Fprintf(out, "NDCG@%d:   %.4f\n", k, rep.MeanNDCG[k])
			}
			_, _ = fmt.Fprintf(out, "MRR:       %.4f\n", rep.MeanMRR)
			for _, k := range recall {
				_, _ = fmt.Fprintf(out, "Recall@%d: %.4f\n", k, rep.MeanRecall[k])
			}
			if goldPath != "" {
				gold, err := readQrels(goldPath)
				if err != nil {
					return err
				}
				ag := eval.CompareToGold(qrels, gold)
				printAgreement(out, ag)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&runPath, "run", "", "TREC run file to score")
	cmd.Flags().StringVar(&qrelsPath, "qrels", "", "TREC qrels file with graded judgments")
	cmd.Flags().StringVar(&goldPath, "gold", "", "human-labeled gold qrels subset to check the labels against")
	cmd.Flags().StringVar(&ndcgArg, "ndcg", "10,20", "comma-separated NDCG cutoffs")
	cmd.Flags().StringVar(&recallArg, "recall", "100,1000", "comma-separated recall cutoffs")
	return cmd
}

// printAgreement renders the gold-set agreement check: the Kendall tau against the trust
// gate, the pairs compared, and the per-grade confusion matrix, so an operator sees not
// just whether the labels pass but where the judge disagrees with the human.
func printAgreement(out io.Writer, ag eval.Agreement) {
	verdict := "FAIL"
	if ag.Passes() {
		verdict = "pass"
	}
	_, _ = fmt.Fprintf(out, "\ngold-set agreement over %d pairs (%d missing)\n", ag.N, ag.Missing)
	_, _ = fmt.Fprintf(out, "Kendall tau: %.4f (gate %.2f: %s)\n", ag.KendallTau, eval.UmbrelaTauGate, verdict)
	_, _ = fmt.Fprintf(out, "confusion [human rows x judge cols]:\n")
	header := "      "
	for j := 0; j <= eval.MaxGrade; j++ {
		header += fmt.Sprintf("  j%d", j)
	}
	_, _ = fmt.Fprintf(out, "%s\n", header)
	for h := 0; h <= eval.MaxGrade && h < len(ag.Confusion); h++ {
		row := fmt.Sprintf("  h%d  ", h)
		for j := 0; j <= eval.MaxGrade && j < len(ag.Confusion[h]); j++ {
			row += fmt.Sprintf("%4d", ag.Confusion[h][j])
		}
		_, _ = fmt.Fprintf(out, "%s\n", row)
	}
}

// parseCutoffs parses a comma-separated list of positive integer cutoffs, the form the
// --ndcg and --recall flags take, returning nil for an empty string so the harness falls
// back to its doc 14 defaults.
func parseCutoffs(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var out []int
	for _, part := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("cutoff %q is not an integer", part)
		}
		if n <= 0 {
			return nil, fmt.Errorf("cutoff %d must be positive", n)
		}
		out = append(out, n)
	}
	sort.Ints(out)
	return out, nil
}

// readRun opens a TREC run file and parses it, the engine output eval scores.
func readRun(path string) (eval.Run, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return eval.ParseRun(f)
}

// readQrels opens a TREC qrels file and parses it, the judgments eval scores against.
func readQrels(path string) (eval.Qrels, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return eval.ParseQrels(f)
}
