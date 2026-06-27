package cli

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
)

// writeShard builds a tiny lexical-plus-feature shard for the serve test.
func writeShard(t *testing.T, path string, texts []string, nodeBase uint32) {
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

	broker, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()
	if broker.NumShards() != 2 {
		t.Fatalf("shards = %d, want 2", broker.NumShards())
	}

	srv := &httpServer{broker: broker, timeout: 0}
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
