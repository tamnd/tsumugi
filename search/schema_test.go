package search

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
)

// buildFeatureShard writes a one-document shard whose feature region is built from the
// given column layout and schema version, so a test can produce a shard that does or
// does not match the schema this build scores against.
func buildFeatureShard(t testing.TB, path string, cols []feature.Column, version uint16) {
	t.Helper()
	lb := lexical.NewBuilder(lexical.DefaultParams())
	lb.AddDoc(0, map[lexical.Field]string{lexical.FieldBody: "common term"})
	fb := feature.NewBuilder(cols, version)
	fb.Set(0, cols[0].ID, 0.5)

	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	w.SetDocCount(1)
	w.SetNodeBase(0)
	if err := w.AddRegion(tsumugi.RegionLexical, tsumugi.CodecZstd, 0, 0, lb.Build()); err != nil {
		t.Fatalf("add lexical: %v", err)
	}
	if err := w.AddRegion(tsumugi.RegionFeature, tsumugi.CodecZstd, 0, 0, fb.Build()); err != nil {
		t.Fatalf("add feature: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// TestShardAcceptsCanonicalSchema checks a shard built against the canonical schema and
// version opens without complaint, the baseline the refusals are measured against.
func TestShardAcceptsCanonicalSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ok.tsumugi")
	buildFeatureShard(t, path, feature.DefaultSchema(), feature.SchemaVersion)
	s, err := OpenShard(path, nil)
	if err != nil {
		t.Fatalf("open canonical shard: %v", err)
	}
	_ = s.Close()
}

// TestShardRefusesSchemaVersionMismatch checks a shard whose feature region records a
// different schema version is refused at open, before any query can misread it.
func TestShardRefusesSchemaVersionMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "badver.tsumugi")
	buildFeatureShard(t, path, feature.DefaultSchema(), feature.SchemaVersion+1)
	_, err := OpenShard(path, nil)
	if !errors.Is(err, ErrSchemaMismatch) {
		t.Fatalf("open got %v, want ErrSchemaMismatch", err)
	}
}

// TestShardRefusesSchemaHashMismatch checks a shard whose columns are reordered, so it
// carries the right version but a different layout, is refused. This is the case the
// version number alone would miss and the fingerprint catches.
func TestShardRefusesSchemaHashMismatch(t *testing.T) {
	cols := feature.DefaultSchema()
	cols[0], cols[1] = cols[1], cols[0]
	path := filepath.Join(t.TempDir(), "badhash.tsumugi")
	buildFeatureShard(t, path, cols, feature.SchemaVersion)
	_, err := OpenShard(path, nil)
	if !errors.Is(err, ErrSchemaMismatch) {
		t.Fatalf("open got %v, want ErrSchemaMismatch", err)
	}
}

// stampedModel trains a tiny model and stamps it with the given schema, so the broker
// check has a real compiled model to inspect.
func stampedModel(t testing.TB, version uint16, hash uint64) *rank.Model {
	t.Helper()
	ens := trainEnsemble(t)
	ens.SetSchema(version, hash)
	return ens.Compile()
}

// trainEnsemble fits the same tiny ensemble trainModel uses but returns it before
// compiling, so a test can stamp it.
func trainEnsemble(t testing.TB) *rank.Ensemble {
	t.Helper()
	cols := feature.DefaultSchema()
	nf := len(cols)
	d := &rank.Dataset{NumFeatures: nf}
	r := lcgSeed(7)
	const queries, per = 20, 8
	for q := 0; q < queries; q++ {
		d.Groups = append(d.Groups, per)
		for i := 0; i < per; i++ {
			row := make([]float64, nf)
			for f := range row {
				row[f] = r()
			}
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, float64(i%3))
		}
	}
	p := rank.DefaultParams()
	p.Rounds = 10
	return rank.Train(d, p)
}

// TestBrokerCheckModel checks the broker accepts a model stamped with the canonical
// schema, refuses one stamped with a different schema, and tolerates an unstamped model
// as the legacy path.
func TestBrokerCheckModel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "shard.tsumugi")
	docs := makeCorpus(40)
	buildShardFile(t, path, docs, 0, 40, 0, false)

	open := func(m *rank.Model) (*Broker, error) {
		s, err := OpenShard(path, newTestCascade(m))
		if err != nil {
			t.Fatalf("open shard: %v", err)
		}
		b := NewBroker([]*Shard{s}, newTestCascade(m))
		return b, b.CheckModel()
	}

	t.Run("matching", func(t *testing.T) {
		b, err := open(stampedModel(t, feature.SchemaVersion, feature.DefaultSchemaHash()))
		if err != nil {
			t.Fatalf("matching model refused: %v", err)
		}
		_ = b.Close()
	})

	t.Run("mismatched", func(t *testing.T) {
		b, err := open(stampedModel(t, feature.SchemaVersion, feature.DefaultSchemaHash()^1))
		if !errors.Is(err, ErrSchemaMismatch) {
			t.Fatalf("mismatched model got %v, want ErrSchemaMismatch", err)
		}
		_ = b.Close()
	})

	t.Run("unstamped", func(t *testing.T) {
		b, err := open(trainModel(t)) // trainModel compiles without a stamp
		if err != nil {
			t.Fatalf("unstamped model refused: %v", err)
		}
		_ = b.Close()
	})
}
