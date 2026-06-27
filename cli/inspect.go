package cli

import (
	"fmt"
	"sort"
	"text/tabwriter"

	"github.com/spf13/cobra"
	"github.com/tamnd/tsumugi"
)

func newInspectCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "inspect <file.tsumugi>",
		Short: "Print a shard's header, regions, and statistics",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			r, err := tsumugi.Open(args[0])
			if err != nil {
				return err
			}
			defer r.Close()
			return printInspect(cmd, r, args[0])
		},
	}
}

func printInspect(cmd *cobra.Command, r *tsumugi.Reader, path string) error {
	h := r.Header
	out := cmd.OutOrStdout()
	fmt.Fprintf(out, "file:     %s\n", path)
	fmt.Fprintf(out, "version:  %d.%d\n", h.VersionMajor, h.VersionMinor)
	fmt.Fprintf(out, "docs:     %d\n", h.DocCount)
	fmt.Fprintf(out, "flags:    %s\n", flagString(h.Flags))
	if h.NodeBase != 0 {
		fmt.Fprintf(out, "node_base:%d\n", h.NodeBase)
	}

	fmt.Fprintln(out, "regions:")
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  kind\tcodec\ton-disk\traw\tratio")
	for _, d := range r.Footer.Regions {
		ratio := 1.0
		if d.RawLength > 0 {
			ratio = float64(d.Length) / float64(d.RawLength)
		}
		fmt.Fprintf(tw, "  %s\t%s\t%d\t%d\t%.2fx\n", d.Kind, d.Codec, d.Length, d.RawLength, ratio)
	}
	tw.Flush()

	if len(r.Footer.Stats) > 0 {
		fmt.Fprintln(out, "stats:")
		keys := make([]string, 0, len(r.Footer.Stats))
		for k := range r.Footer.Stats {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(out, "  %-14s %g\n", k, r.Footer.Stats[k])
		}
	}
	return nil
}

func flagString(flags uint64) string {
	var s []string
	add := func(bit uint64, name string) {
		if flags&bit != 0 {
			s = append(s, name)
		}
	}
	add(tsumugi.FlagHasLexical, "lexical")
	add(tsumugi.FlagHasForward, "forward")
	add(tsumugi.FlagHasFeature, "feature")
	add(tsumugi.FlagHasGraph, "graph")
	add(tsumugi.FlagHasVector, "vector")
	add(tsumugi.FlagHasDictionary, "dictionary")
	add(tsumugi.FlagSearchOnly, "search-only")
	add(tsumugi.FlagImpactPostings, "impact-postings")
	if len(s) == 0 {
		return "(none)"
	}
	out := s[0]
	for _, x := range s[1:] {
		out += "," + x
	}
	return out
}
