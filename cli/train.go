package cli

import (
	"fmt"
	"math"
	"os"

	"github.com/spf13/cobra"
	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/rank"
)

func newTrainCmd() *cobra.Command {
	var (
		out       string
		groupSize int
		rounds    int
	)
	cmd := &cobra.Command{
		Use:   "train <dir>",
		Short: "Bootstrap a ranking model from a collection",
		Long: "train fits a LambdaMART model over a collection's feature matrix using the\n" +
			"static-rank prior as a bootstrap label, the cold-start model the serve command\n" +
			"ranks with until real relevance judgments replace the prior. It reads every\n" +
			"shard's features, groups documents into synthetic queries, fits the model, and\n" +
			"writes it to a file.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if out == "" {
				return fmt.Errorf("a model output path is required: pass --out")
			}
			d, err := bootstrapDataset(args[0], groupSize)
			if err != nil {
				return err
			}
			if len(d.Features) == 0 {
				return fmt.Errorf("collection %s has no feature data to train on", args[0])
			}
			p := rank.DefaultParams()
			p.Rounds = rounds
			ens := rank.Train(d, p)
			f, err := os.Create(out)
			if err != nil {
				return err
			}
			if err := ens.Save(f); err != nil {
				_ = f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
			_, _ = fmt.Fprintf(cmd.OutOrStdout(),
				"trained %d trees over %d documents in %d queries, wrote %s\n",
				ens.NumTrees(), len(d.Features), len(d.Groups), out)
			return nil
		},
	}
	cmd.Flags().StringVar(&out, "out", "", "model output file")
	cmd.Flags().IntVar(&groupSize, "group-size", 16, "documents per synthetic query group")
	cmd.Flags().IntVar(&rounds, "rounds", 200, "boosting rounds")
	return cmd
}

// bootstrapDataset reads a collection's feature matrix into a training set, using the
// static-rank feature as the graded bootstrap label. It groups documents into
// fixed-size synthetic queries in collection order, so a query holds a slice of the
// host-clustered id space and the model learns the static-rank prior from the other
// features that explain it. This is the cold-start label the spec's LTR bootstrap
// stands up before real judgments exist.
func bootstrapDataset(dir string, groupSize int) (*rank.Dataset, error) {
	if groupSize < 2 {
		groupSize = 16
	}
	infos, err := collection.List(dir)
	if err != nil {
		return nil, err
	}
	cols := feature.DefaultSchema()
	d := &rank.Dataset{NumFeatures: len(cols)}
	staticIdx := schemaIndex(cols, feature.FeatStaticRank)

	var inGroup int
	for _, in := range infos {
		r, err := tsumugi.Open(in.Path)
		if err != nil {
			return nil, err
		}
		if !r.HasRegion(tsumugi.RegionFeature) {
			_ = r.Close()
			continue
		}
		b, err := r.Region(tsumugi.RegionFeature)
		if err != nil {
			_ = r.Close()
			return nil, err
		}
		fr, err := feature.Open(b)
		if err != nil {
			_ = r.Close()
			return nil, err
		}
		n := fr.DocCount()
		for id := uint32(0); id < n; id++ {
			row := make([]float64, len(cols))
			for i, c := range cols {
				if v, ok := fr.Value(id, c.ID); ok {
					row[i] = v
				}
			}
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, gradeLabel(row[staticIdx]))
			inGroup++
			if inGroup == groupSize {
				d.Groups = append(d.Groups, groupSize)
				inGroup = 0
			}
		}
		_ = r.Close()
	}
	if inGroup > 0 {
		d.Groups = append(d.Groups, inGroup)
	}
	return d, nil
}

// gradeLabel buckets a static-rank value into a 0..4 graded relevance label, the
// shape LambdaMART's NDCG objective grades against.
func gradeLabel(static float64) float64 {
	g := math.Floor(static / 20)
	if g > 4 {
		g = 4
	}
	if g < 0 {
		g = 0
	}
	return g
}

func schemaIndex(cols []feature.Column, id feature.FeatureID) int {
	for i, c := range cols {
		if c.ID == id {
			return i
		}
	}
	return 0
}
