package cli

import (
	"fmt"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
	"github.com/tamnd/tsumugi/collection"
)

func newCollectionCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "collection",
		Short: "Manage a multi-shard collection",
		Long:  "collection lists, extends, and compacts a directory of .tsumugi shards.",
	}
	cmd.AddCommand(newCollectionListCmd())
	cmd.AddCommand(newCollectionAddCmd())
	cmd.AddCommand(newCollectionCompactCmd())
	cmd.AddCommand(newCollectionIndexCmd())
	return cmd
}

func newCollectionIndexCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "index <dir>",
		Short: "Build the collection artifact: manifest, statistics, and routing index",
		Long: "index scans a collection's shards once and writes the index.tsm artifact that\n" +
			"serve loads instead of rescanning every shard at startup. Build, add, and compact\n" +
			"refresh it automatically; run this to create it for a collection built before the\n" +
			"artifact existed, or to rebuild it after editing the shard directory by hand.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			start := time.Now()
			if err := collection.WriteIndex(args[0], uint64(time.Now().Unix())); err != nil {
				return err
			}
			ix, err := collection.LoadIndex(args[0])
			if err != nil {
				return err
			}
			_, _ = fmt.Fprintf(cmd.OutOrStdout(),
				"indexed %d shards, %d docs in %s\n",
				ix.NumShards(), ix.Stats.DocCount, time.Since(start).Round(time.Millisecond))
			return nil
		},
	}
}

func newCollectionListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list <dir>",
		Short: "List the shards in a collection",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			infos, err := collection.List(args[0])
			if err != nil {
				return err
			}
			tw := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 2, 2, ' ', 0)
			_, _ = fmt.Fprintln(tw, "SHARD\tBASE\tDOCS\tSIZE")
			var docs uint32
			var bytes int64
			for _, s := range infos {
				_, _ = fmt.Fprintf(tw, "%s\t%d\t%d\t%.1f MB\n",
					shardBase(s.Path), s.NodeBase, s.DocCount, float64(s.Bytes)/(1<<20))
				docs += s.DocCount
				bytes += s.Bytes
			}
			_, _ = fmt.Fprintf(tw, "total\t\t%d\t%.1f MB\n", docs, float64(bytes)/(1<<20))
			return tw.Flush()
		},
	}
}

func newCollectionAddCmd() *cobra.Command {
	var opts collection.Options
	cmd := &cobra.Command{
		Use:   "add <dir>",
		Short: "Add a crawl export to an existing collection",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if opts.Source == "" {
				return fmt.Errorf("a crawl export is required: pass --source")
			}
			opts.Out = args[0]
			start := time.Now()
			res, err := collection.Add(opts)
			if err != nil {
				return err
			}
			printBuildResult(cmd, res, time.Since(start))
			return nil
		},
	}
	cmd.Flags().StringVar(&opts.Source, "source", "", "crawl export to add (.parquet or .jsonl)")
	cmd.Flags().IntVar(&opts.ShardSize, "shard-size", collection.DefaultShardSize, "documents per shard")
	cmd.Flags().IntVar(&opts.Limit, "limit", 0, "cap documents read, zero for all")
	return cmd
}

func newCollectionCompactCmd() *cobra.Command {
	var shardSize int
	cmd := &cobra.Command{
		Use:   "compact <dir>",
		Short: "Merge a collection's shards into fewer, larger ones",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			start := time.Now()
			res, err := collection.Compact(args[0], shardSize)
			if err != nil {
				return err
			}
			_, _ = fmt.Fprintf(cmd.OutOrStdout(),
				"compacted %d docs into %d shards (%.1f MB) in %s\n",
				res.Docs, res.Shards, float64(res.Bytes)/(1<<20), time.Since(start).Round(time.Millisecond))
			return nil
		},
	}
	cmd.Flags().IntVar(&shardSize, "shard-size", collection.DefaultShardSize, "documents per merged shard")
	return cmd
}

// shardBase is the shard file name without its directory, the short label the listing
// shows.
func shardBase(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			return path[i+1:]
		}
	}
	return path
}
