package collection_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/vector"
)

// denseFindByURL scans a shard's forward url column for the given URL and returns its
// dense docID, so a test can name a document by its stable URL rather than guess the id
// the graph reorder assigned. The vector region numbers its nodes by the same docID, so
// this id indexes both the forward store and the vector region.
func denseFindByURL(t *testing.T, fwd *forward.Region, url string) (uint32, bool) {
	t.Helper()
	n := fwd.DocCount()
	for id := uint32(0); id < n; id++ {
		if v, ok := fwd.Column("url", id); ok && string(v) == url {
			return id, true
		}
	}
	return 0, false
}

// TestBuildEmitsVectorRegion is the unit gate for the dense-enabled build: a build with a
// kept dimension writes a per-shard vector region and records the dimension in the footer,
// and a build without one writes neither, the opt-in default. It then reads the region back
// through the production int8 cosine path and checks a document's own lead line, encoded by
// the same default encoder the query pipeline uses, cosines higher against that document
// than against an unrelated one, the load-bearing property that a document and a query the
// build and the pipeline embed live in one comparable space.
func TestBuildEmitsVectorRegion(t *testing.T) {
	tmp := t.TempDir()
	lines := []string{
		`{"url":"https://a.example/kernel","host":"a.example","markdown":"# Kernel scheduling\nthe kernel scheduler assigns cpu time slices to runnable threads and processes"}`,
		`{"url":"https://b.example/pastry","host":"b.example","markdown":"# Croissant recipe\nfold the butter into the laminated dough and proof the pastry before baking"}`,
		`{"url":"https://c.example/finance","host":"c.example","markdown":"# Bond yields\nthe treasury bond yield curve inverted as investors priced in interest rate cuts"}`,
		`{"url":"https://d.example/garden","host":"d.example","markdown":"# Tomato plants\nprune the tomato seedlings and water the raised garden beds each morning"}`,
	}
	src := filepath.Join(tmp, "src.jsonl")
	if err := os.WriteFile(src, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A build with no kept dimension writes no vector region and no vector_dim stat, the
	// opt-in default that leaves every existing collection unchanged.
	plain := filepath.Join(tmp, "plain")
	if _, err := collection.Build(collection.Options{Source: src, Out: plain, ShardSize: 1000}); err != nil {
		t.Fatalf("plain build: %v", err)
	}
	infos, err := collection.List(plain)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range infos {
		r, err := tsumugi.Open(in.Path)
		if err != nil {
			t.Fatal(err)
		}
		if r.HasRegion(tsumugi.RegionVector) {
			t.Errorf("plain build emitted a vector region")
		}
		if _, ok := r.Stat(tsumugi.StatVectorDim); ok {
			t.Errorf("plain build recorded a vector_dim stat")
		}
		_ = r.Close()
	}

	// A build with a kept dimension writes the region and records the dimension.
	const dim = 64
	out := filepath.Join(tmp, "dense")
	if _, err := collection.Build(collection.Options{Source: src, Out: out, ShardSize: 1000, DenseDim: dim}); err != nil {
		t.Fatalf("dense build: %v", err)
	}
	infos, err = collection.List(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 1 {
		t.Fatalf("got %d shards, want 1", len(infos))
	}
	r, err := tsumugi.Open(infos[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()

	if !r.HasRegion(tsumugi.RegionVector) {
		t.Fatal("dense build emitted no vector region")
	}
	if v, ok := r.Stat(tsumugi.StatVectorDim); !ok || int(v) != dim {
		t.Fatalf("vector_dim stat = %v (ok=%v), want %d", v, ok, dim)
	}

	vb, err := r.Region(tsumugi.RegionVector)
	if err != nil {
		t.Fatal(err)
	}
	reg, err := vector.Open(vb)
	if err != nil {
		t.Fatalf("open vector region: %v", err)
	}
	if reg.Dim() != dim {
		t.Fatalf("region Dim() = %d, want %d", reg.Dim(), dim)
	}

	fb, err := r.Region(tsumugi.RegionForward)
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := forward.Open(fb)
	if err != nil {
		t.Fatal(err)
	}
	defer fwd.Close()

	// The query side uses the same default encoder the build embedded documents with, so a
	// document's own lead line, encoded as a query, must cosine higher against that document
	// than against an unrelated one through the region's int8 rerank path.
	enc := dense.NewDefault(dim)
	self := "https://a.example/kernel"
	selfID, ok := denseFindByURL(t, fwd, self)
	if !ok {
		t.Fatalf("url %q not found", self)
	}
	other := "https://b.example/pastry"
	otherID, ok := denseFindByURL(t, fwd, other)
	if !ok {
		t.Fatalf("url %q not found", other)
	}
	body, _ := fwd.Column("body", selfID)
	q := enc.Encode(lexical.Analyze(firstLine(string(body))))

	selfCos, ok := reg.Cosine(q, selfID)
	if !ok {
		t.Fatal("no cosine for self doc")
	}
	otherCos, _ := reg.Cosine(q, otherID)
	if selfCos <= otherCos {
		t.Fatalf("self cosine %.4f not above unrelated %.4f: the built vectors do not place a document near its own text", selfCos, otherCos)
	}
}

// TestDenseBuildCCrawl proves the dense-enabled build end to end on the real crawl: it
// builds a multi-shard collection with a kept dimension through the ordinary Build path,
// then reads each shard's vector region back through the production int8 cosine and checks a
// page's own lead line, encoded by the query pipeline's default encoder, ranks that page
// above an unrelated one on the large majority of pages. Unlike TestDenseEncodeCCrawl,
// which builds its vectors in the test, this routes the vectors the build itself wrote,
// through the container region and the real rotation-and-quantize reader, so it is the gate
// that the build embeds documents into a usable space, not just that the encoder can.
func TestDenseBuildCCrawl(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 8000, DenseDim: denseDim})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards, got %d", res.Shards)
	}

	infos, err := collection.List(out)
	if err != nil {
		t.Fatal(err)
	}
	enc := dense.NewDefault(denseDim)

	wins, trials := 0, 0
	for _, in := range infos {
		r, err := tsumugi.Open(in.Path)
		if err != nil {
			t.Fatal(err)
		}
		// Every shard the dense build wrote records the dimension and carries the region.
		if v, ok := r.Stat(tsumugi.StatVectorDim); !ok || int(v) != denseDim {
			t.Fatalf("%s: vector_dim stat = %v (ok=%v), want %d", filepath.Base(in.Path), v, ok, denseDim)
		}
		vb, err := r.Region(tsumugi.RegionVector)
		if err != nil {
			t.Fatal(err)
		}
		reg, err := vector.Open(vb)
		if err != nil {
			t.Fatalf("open vector region: %v", err)
		}
		if reg.Dim() != denseDim {
			t.Fatalf("region Dim() = %d, want %d", reg.Dim(), denseDim)
		}
		fb, err := r.Region(tsumugi.RegionForward)
		if err != nil {
			t.Fatal(err)
		}
		fwd, err := forward.Open(fb)
		if err != nil {
			t.Fatal(err)
		}
		n := fwd.DocCount()
		// For a sample of documents in the shard, encode the page's own lead line as a query
		// and pit its cosine against its own vector versus an unrelated document's, both read
		// from the built region. A correct build wins the large majority.
		for id := uint32(0); id < n; id += 7 {
			body, ok := fwd.Column("body", id)
			if !ok {
				continue
			}
			leadTerms := lexical.Analyze(firstLine(string(body)))
			if len(leadTerms) < 3 {
				continue
			}
			q := enc.Encode(leadTerms)
			zero := true
			for _, x := range q {
				if x != 0 {
					zero = false
					break
				}
			}
			if zero {
				continue
			}
			other := (id + n/2) % n
			if other == id {
				continue
			}
			selfCos, ok := reg.Cosine(q, id)
			if !ok {
				continue
			}
			otherCos, _ := reg.Cosine(q, other)
			trials++
			if selfCos > otherCos {
				wins++
			}
		}
		fwd.Close()
		_ = r.Close()
	}

	if trials == 0 {
		t.Fatal("no page carried an encodable lead line over the real corpus")
	}
	// Random indexing plus the one-bit-plus-int8 quantization loses some resolution, so the
	// bar is a large majority, not perfection, the same shape TestDenseEncodeCCrawl holds.
	rate := float64(wins) / float64(trials)
	if rate < 0.90 {
		t.Fatalf("self-outranks-unrelated only %.1f%% (%d/%d); the build's document vectors do not place pages near their own text", rate*100, wins, trials)
	}
	t.Logf("dense build: %d/%d (%.1f%%) pages outrank an unrelated page on their own lead line", wins, trials, rate*100)
}
