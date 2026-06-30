package cli

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

// TestPipelineCorrectsTypo builds a shard whose vocabulary holds a word often enough to be
// a correction target, then serves a query that misspells it by one edit. The pipeline's
// fleet-built corrector should auto-substitute the correct term, retrieve the document
// that carries it, and report the correction in the response, the end-to-end proof that
// query understanding runs in the serve path rather than just in the query package.
func TestPipelineCorrectsTypo(t *testing.T) {
	dir := t.TempDir()
	// The corrector's MinFreq is two, so the target term must appear in at least two
	// documents to enter the correction dictionary.
	docs := []string{
		"the brown bear forages",
		"a brown bear sleeps",
		"brown bear tracks in snow",
	}
	writeShard(t, filepath.Join(dir, "a.tsumugi"), docs, 0)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	srv := &httpServer{backend: broker, pipeline: pl, timeout: 0}
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	defer ts.Close()

	// "bera" is one transposition from "bear", a word the dictionary holds at df three.
	got := getSearch(t, ts.URL+"?q=bera&k=5")
	if !got.Corrected {
		t.Fatalf("query was not corrected: %+v", got)
	}
	if len(got.Hits) == 0 {
		t.Fatalf("corrected query returned no hits: %+v", got)
	}
}

// TestPipelineEmptyQuery checks the serve path short-circuits a query with nothing to
// retrieve on to an empty hit list rather than fanning it out, the spec's empty-query
// behavior, and still answers with the shard count so a caller can tell the broker is up.
func TestPipelineEmptyQuery(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"brown bear"}, 0)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	srv := &httpServer{backend: broker, pipeline: pl, timeout: 0}
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	defer ts.Close()

	got := getSearch(t, ts.URL+"?q=")
	if len(got.Hits) != 0 {
		t.Fatalf("empty query returned hits: %+v", got)
	}
	if got.Shards != 1 {
		t.Fatalf("response shards = %d, want 1", got.Shards)
	}
}

// TestPipelineExpandsRecall checks the pipeline folds a term's curated expansion
// alternatives into the retrieval set, so a query for an acronym recalls a document that
// carries only the expansion. The default table maps "js" to "javascript", so a document
// holding "javascript" answers a query for "js" even though the two share no token.
func TestPipelineExpandsRecall(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"javascript powers the web", "static html page"}, 0)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	// The query term "js" carries "javascript" as an expansion alternative, so the
	// retrieval set the broker ships must include the expansion.
	pq := pl.parse("js")
	terms := pq.RetrievalTerms()
	if !contains(terms, "javascript") {
		t.Fatalf("retrieval terms %v missing the expansion of js", terms)
	}

	srv := &httpServer{backend: broker, pipeline: pl, timeout: 0}
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	defer ts.Close()
	got := getSearch(t, ts.URL+"?q=js&k=5")
	if len(got.Hits) == 0 {
		t.Fatalf("expanded query returned no hits: %+v", got)
	}
}

// TestPipelineNoVectorRegionLeavesDenseOff checks the dense encoder is wired only when the
// shards carry a vector region. The collection build emits no vector region, so the
// pipeline's encoder must be nil and ApplyDense a no-op, which leaves DenseVec empty.
func TestPipelineNoVectorRegionLeavesDenseOff(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"brown bear forages"}, 0)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	defer func() { _ = broker.Close() }()

	if pl.encoder != nil {
		t.Fatalf("dense encoder wired with no vector region")
	}
	pq := pl.parse("brown bear")
	if pq.DenseVec != nil {
		t.Fatalf("dense vector produced with no vector region: %d bytes", len(pq.DenseVec))
	}
}

// getSearch issues a search request and decodes the JSON response.
func getSearch(t *testing.T, url string) searchResponse {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	var got searchResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return got
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}
