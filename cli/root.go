// Package cli wires tsumugi's command surface: the cobra tree and the
// fang-rendered help and errors. The engine work lives in the tsumugi package
// and its subpackages; this layer parses flags and prints results.
package cli

import (
	"context"
	"fmt"

	"github.com/charmbracelet/fang"
	"github.com/spf13/cobra"
)

// Execute builds the root command and runs it through fang. It returns the
// process exit code.
func Execute(ctx context.Context) int {
	root := newRoot()
	opts := []fang.Option{fang.WithVersion(Version)}
	if err := fang.Execute(ctx, root, opts...); err != nil {
		return 1
	}
	return 0
}

func newRoot() *cobra.Command {
	root := &cobra.Command{
		Use:   "tsumugi",
		Short: "A web-scale search and ranking engine on compact single-file shards",
		Long: "tsumugi weaves a crawl into compact .tsumugi shards and serves ranked\n" +
			"results in milliseconds. Each shard is one self-describing file holding the\n" +
			"inverted index, stored fields, a quantized feature matrix, the link graph,\n" +
			"and quantized vectors. This CLI builds, inspects, and serves them.",
		Version:       fmt.Sprintf("%s (commit %s, built %s)", Version, Commit, Date),
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.AddCommand(newInspectCmd())
	return root
}
