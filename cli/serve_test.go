package cli

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
)

// writeShard builds a tiny lexical-plus-feature shard for the serve test, recording the
// query-side analyzer hash so the shard matches the broker it is served by.
func writeShard(t *testing.T, path string, texts []string, nodeBase uint32) {
	t.Helper()
	writeShardHash(t, path, texts, nodeBase, queryAnalyzerHash())
}

// writeShardHash is writeShard with an explicit recorded analyzer hash, so a test can
// build a shard the broker will accept (matching hash) or refuse (mismatched hash). A
// zero hash records nothing, the shard built before the hash existed.
func writeShardHash(t *testing.T, path string, texts []string, nodeBase uint32, hash uint64) {
	t.Helper()
	lb := lexical.NewBuilder(lexical.DefaultParams())
	fb := feature.NewBuilder(feature.DefaultSchema(), 1)
	for i, txt := range texts {
		lb.AddDoc(uint32(i), map[lexical.Field]string{lexical.FieldBody: txt})
		fb.Set(uint32(i), feature.FeatInDegree, float64((i+1)*100))
	}
	w, err := tsumugi.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	w.SetDocCount(uint32(len(texts)))
	w.SetNodeBase(uint64(nodeBase))
	w.SetStat(tsumugi.StatTokenCount, float64(len(texts)*3))
	if hash != 0 {
		w.SetAnalyzerHash(hash)
	}
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

// writeModel trains a tiny ensemble and saves it to disk for the serve loader.
func writeModel(t *testing.T, path string) {
	writeModelStamped(t, path, feature.SchemaVersion, feature.DefaultSchemaHash())
}

// writeModelStamped trains the same tiny model writeModel does but stamps it with the
// given feature schema before saving, so a test can write a model that does or does not
// match the schema the serve path scores against.
func writeModelStamped(t *testing.T, path string, version uint16, hash uint64) {
	t.Helper()
	nf := len(feature.DefaultSchema())
	d := &rank.Dataset{NumFeatures: nf}
	var s uint64 = 7
	rnd := func() float64 {
		s = s*6364136223846793005 + 1442695040888963407
		return float64(s>>11) / float64(1<<53)
	}
	for q := 0; q < 20; q++ {
		d.Groups = append(d.Groups, 8)
		for i := 0; i < 8; i++ {
			row := make([]float64, nf)
			for f := range row {
				row[f] = rnd()
			}
			d.Features = append(d.Features, row)
			d.Labels = append(d.Labels, float64(int(row[0]*4)))
		}
	}
	p := rank.DefaultParams()
	p.Rounds = 20
	ens := rank.Train(d, p)
	ens.SetSchema(version, hash)
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create model: %v", err)
	}
	if err := ens.Save(f); err != nil {
		t.Fatalf("save model: %v", err)
	}
	_ = f.Close()
}

// TestServeSearch builds a small two-shard collection, loads it through the serve
// path, and checks the HTTP handler returns a ranked JSON top-k with global ids
// spanning both shards.
func TestServeSearch(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox", "lazy brown dog"}, 0)
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs", "swift brown hare"}, 1000)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()
	if broker.NumShards() != 2 {
		t.Fatalf("shards = %d, want 2", broker.NumShards())
	}

	srv := &httpServer{broker: broker, pipeline: pl, timeout: 0}
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "?q=brown&k=4")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var got searchResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Hits) == 0 {
		t.Fatalf("no hits returned")
	}
	if got.Shards != 2 {
		t.Fatalf("response shards = %d, want 2", got.Shards)
	}
	for i := 1; i < len(got.Hits); i++ {
		if got.Hits[i].Score > got.Hits[i-1].Score {
			t.Fatalf("hits not score-sorted: %+v", got.Hits)
		}
	}
	// The query term is in every document, so the global top-k should reach into the
	// second shard's id space, proving the fan-out merged both shards.
	var sawHigh bool
	for _, h := range got.Hits {
		if h.DocID >= 1000 {
			sawHigh = true
		}
	}
	if !sawHigh {
		t.Fatalf("no hit from the second shard: %+v", got.Hits)
	}
}

// TestServeRefusesMismatchedShard checks the glob fallback path refuses to serve a shard
// recorded under an analyzer the broker does not query with. Serving it would match the
// query against tokens the broker never produces, the silent wrong-results case the hash
// exists to prevent.
func TestServeRefusesMismatchedShard(t *testing.T) {
	dir := t.TempDir()
	writeShardHash(t, filepath.Join(dir, "a.tsumugi"), []string{"brown fox"}, 0, queryAnalyzerHash()^0xFF)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	_, _, err := openCollection(dir, modelPath)
	if !errors.Is(err, collection.ErrAnalyzerMismatch) {
		t.Fatalf("openCollection error = %v, want ErrAnalyzerMismatch", err)
	}
}

// TestServeAcceptsMatchedShard checks the glob fallback path serves a shard whose recorded
// analyzer matches the broker's, the healthy case the refusal must not block.
func TestServeAcceptsMatchedShard(t *testing.T) {
	dir := t.TempDir()
	writeShardHash(t, filepath.Join(dir, "a.tsumugi"), []string{"brown fox"}, 0, queryAnalyzerHash())
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, _, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	_ = broker.Close()
}

// TestServeRefusesMismatchedManifest checks the manifest path refuses a collection whose
// recorded analyzer does not match the broker's. The manifest carries one hash for the
// whole collection, so the broker rejects in a single check without opening every shard.
func TestServeRefusesMismatchedManifest(t *testing.T) {
	dir := t.TempDir()
	writeShardHash(t, filepath.Join(dir, "shard-00000.tsumugi"), []string{"brown fox"}, 0, queryAnalyzerHash()^0xAA)
	writeShardHash(t, filepath.Join(dir, "shard-00001.tsumugi"), []string{"brown bear"}, 1000, queryAnalyzerHash()^0xAA)
	if err := collection.WriteIndex(dir, collection.NoEpoch); err != nil {
		t.Fatalf("WriteIndex: %v", err)
	}
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	_, _, err := openCollection(dir, modelPath)
	if !errors.Is(err, collection.ErrAnalyzerMismatch) {
		t.Fatalf("openCollection error = %v, want ErrAnalyzerMismatch", err)
	}
}

// TestWriteIndexRefusesMixedAnalyzers checks WriteIndex refuses a collection whose shards
// disagree on their analyzer hash: a mixed-analyzer collection cannot be queried
// consistently, so the failure surfaces at build time rather than at serve time.
func TestWriteIndexRefusesMixedAnalyzers(t *testing.T) {
	dir := t.TempDir()
	writeShardHash(t, filepath.Join(dir, "shard-00000.tsumugi"), []string{"brown fox"}, 0, 0x1111)
	writeShardHash(t, filepath.Join(dir, "shard-00001.tsumugi"), []string{"brown bear"}, 1000, 0x2222)

	err := collection.WriteIndex(dir, collection.NoEpoch)
	if !errors.Is(err, collection.ErrAnalyzerMismatch) {
		t.Fatalf("WriteIndex error = %v, want ErrAnalyzerMismatch", err)
	}
}
