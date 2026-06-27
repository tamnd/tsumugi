package cli

import (
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"github.com/tamnd/tsumugi/collection"
)

func newBuildCmd() *cobra.Command {
	var opts collection.Options
	cmd := &cobra.Command{
		Use:   "build",
		Short: "Build a collection of shards from a crawl export",
		Long: "build reads a crawl export, a Parquet or newline-delimited JSON file, orders\n" +
			"the documents by host for locality, and writes them into .tsumugi shards under\n" +
			"the output directory. The result is a collection the serve command can answer\n" +
			"queries over directly.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if opts.Source == "" {
				return fmt.Errorf("a crawl export is required: pass --source")
			}
			if opts.Out == "" {
				return fmt.Errorf("an output directory is required: pass --out")
			}
			start := time.Now()
			res, err := collection.Build(opts)
			if err != nil {
				return err
			}
			printBuildResult(cmd, res, time.Since(start))
			return nil
		},
	}
	cmd.Flags().StringVar(&opts.Source, "source", "", "crawl export to read (.parquet or .jsonl)")
	cmd.Flags().StringVar(&opts.Out, "out", "", "output directory for the shards")
	cmd.Flags().IntVar(&opts.ShardSize, "shard-size", collection.DefaultShardSize, "documents per shard")
	cmd.Flags().IntVar(&opts.Limit, "limit", 0, "cap documents read, zero for all")
	return cmd
}

func printBuildResult(cmd *cobra.Command, res collection.Result, took time.Duration) {
	mb := float64(res.Bytes) / (1 << 20)
	_, _ = fmt.Fprintf(cmd.OutOrStdout(),
		"built %d docs from %d hosts into %d shards (%.1f MB) in %s\n",
		res.Docs, res.Hosts, res.Shards, mb, took.Round(time.Millisecond))
}
