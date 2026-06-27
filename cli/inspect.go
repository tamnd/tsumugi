package cli

import (
	"bytes"
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
			defer func() { _ = r.Close() }()
			out := formatInspect(r, args[0])
			_, err = cmd.OutOrStdout().Write(out)
			return err
		},
	}
}

// formatInspect renders a shard report into a buffer so the command does one
// checked write to stdout.
func formatInspect(r *tsumugi.Reader, path string) []byte {
	h := r.Header
	var b bytes.Buffer
	fmt.Fprintf(&b, "file:     %s\n", path)
	fmt.Fprintf(&b, "version:  %d.%d\n", h.VersionMajor, h.VersionMinor)
	fmt.Fprintf(&b, "docs:     %d\n", h.DocCount)
	fmt.Fprintf(&b, "flags:    %s\n", flagString(h.Flags))
	if h.NodeBase != 0 {
		fmt.Fprintf(&b, "node_base:%d\n", h.NodeBase)
	}

	fmt.Fprintln(&b, "regions:")
	// tabwriter writes into the bytes.Buffer and so never fails; ignore the
	// io.Writer errors it surfaces.
	tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(tw, "  kind\tcodec\ton-disk\traw\tratio")
	for _, d := range r.Footer.Regions {
		ratio := 1.0
		if d.RawLength > 0 {
			ratio = float64(d.Length) / float64(d.RawLength)
		}
		_, _ = fmt.Fprintf(tw, "  %s\t%s\t%d\t%d\t%.2fx\n", d.Kind, d.Codec, d.Length, d.RawLength, ratio)
	}
	_ = tw.Flush()

	if len(r.Footer.Stats) > 0 {
		fmt.Fprintln(&b, "stats:")
		keys := make([]string, 0, len(r.Footer.Stats))
		for k := range r.Footer.Stats {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(&b, "  %-14s %g\n", k, r.Footer.Stats[k])
		}
	}
	return b.Bytes()
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
