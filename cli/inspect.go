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
	var sizes bool
	cmd := &cobra.Command{
		Use:   "inspect <file.tsumugi>",
		Short: "Print a shard's header, regions, and statistics",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			r, err := tsumugi.Open(args[0])
			if err != nil {
				return err
			}
			defer func() { _ = r.Close() }()
			var out []byte
			if sizes {
				out = formatSizes(r, args[0])
			} else {
				out = formatInspect(r, args[0])
			}
			_, err = cmd.OutOrStdout().Write(out)
			return err
		},
	}
	cmd.Flags().BoolVar(&sizes, "sizes", false,
		"report per-region bytes-per-document against the doc 13 compression budget")
	return cmd
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
	if h.BuildEpoch != 0 {
		fmt.Fprintf(&b, "epoch:    %d\n", h.BuildEpoch)
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
			// The analyzer and build-config hashes are 64-bit values carried as a
			// reinterpreted float64, so render them as hex rather than the meaningless
			// floating-point number their bit pattern decodes to.
			switch k {
			case tsumugi.StatAnalyzerHash, tsumugi.StatBuildConfigHash:
				fmt.Fprintf(&b, "  %-18s %#016x\n", k, tsumugi.AnalyzerHashFromStat(r.Footer.Stats[k]))
			default:
				fmt.Fprintf(&b, "  %-18s %g\n", k, r.Footer.Stats[k])
			}
		}
	}
	return b.Bytes()
}

// formatSizes renders the doc 13 compactness report: each region's on-disk and raw
// bytes, its compression ratio, its per-document on-disk cost, and the doc 13 budget it
// is checked against, with an ok-or-over verdict. It is the human view of the same
// RegionStats the compactness benchmark gates on, so a build can be eyeballed against the
// budget contract without running the test suite. The total line sums the on-disk bytes
// and the per-document cost across every region, the single number a capacity plan reads.
func formatSizes(r *tsumugi.Reader, path string) []byte {
	var b bytes.Buffer
	fmt.Fprintf(&b, "file:  %s\n", path)
	fmt.Fprintf(&b, "docs:  %d\n", r.Header.DocCount)
	fmt.Fprintln(&b, "sizes:")
	tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(tw, "  region\ton-disk\traw\tratio\tbytes/doc\tbudget")
	var totalOnDisk uint64
	var totalPerDoc float64
	for _, s := range r.RegionStats() {
		_, _ = fmt.Fprintf(tw, "  %s\t%s\t%s\t%.2fx\t%.0f\t%s\n",
			s.Kind, humanBytes(s.OnDisk), humanBytes(s.Raw), s.Ratio, s.BytesPerDoc, s.BudgetVerdict())
		totalOnDisk += s.OnDisk
		totalPerDoc += s.BytesPerDoc
	}
	_, _ = fmt.Fprintf(tw, "  %s\t%s\t\t\t%.0f\t\n", "total", humanBytes(totalOnDisk), totalPerDoc)
	_ = tw.Flush()
	return b.Bytes()
}

// humanBytes renders a byte count in the units the doc 13 sample report uses, so a
// region's size reads as "912 MB" rather than a nine-digit integer.
func humanBytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := uint64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
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
