package cli

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/search"
)

// loadReloadable opens a directory through the serve path and returns the broker, the
// reloader, and a cleanup, the shared setup for the reload tests.
func loadReloadable(t *testing.T, dir string) (*reloader, func()) {
	t.Helper()
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)
	broker, _, rl, err := loadServe(dir, modelPath)
	if err != nil {
		t.Fatalf("loadServe: %v", err)
	}
	return rl, func() { _ = broker.Close() }
}

// docsAcross queries the broker through the reloader and returns whether any hit falls in
// the given global id range, the test of whether a shard's documents are being served.
func servedHasRange(t *testing.T, rl *reloader, lo, hi uint32) bool {
	t.Helper()
	hits := rl.broker.Search(context.Background(), search.Query{Text: "brown", K: 200})
	for _, h := range hits {
		if h.DocID >= lo && h.DocID < hi {
			return true
		}
	}
	return false
}

// TestReloadPublishAddsShard checks publishing a shard built into the directory after
// startup serves its documents and grows the served counts.
func TestReloadPublishAddsShard(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox", "lazy brown dog"}, 0)
	rl, done := loadReloadable(t, dir)
	defer done()

	if rl.broker.NumShards() != 1 {
		t.Fatalf("startup shards = %d, want 1", rl.broker.NumShards())
	}
	if servedHasRange(t, rl, 1000, 2000) {
		t.Fatalf("second shard's docs served before it was published")
	}

	// Build a new shard into the directory and publish it.
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs", "swift brown hare"}, 1000)
	if err := rl.publish("b.tsumugi"); err != nil {
		t.Fatalf("publish: %v", err)
	}
	if rl.broker.NumShards() != 2 {
		t.Fatalf("after publish shards = %d, want 2", rl.broker.NumShards())
	}
	if !servedHasRange(t, rl, 1000, 2000) {
		t.Fatalf("published shard's docs not served")
	}
	// A re-publish of the same name is a no-op, not a double-serve.
	if err := rl.publish("b.tsumugi"); err != nil {
		t.Fatalf("re-publish: %v", err)
	}
	if rl.broker.NumShards() != 2 {
		t.Fatalf("re-publish changed shard count to %d", rl.broker.NumShards())
	}
}

// TestReloadRetireRemovesShard checks retiring a served shard stops serving its documents
// and shrinks the count, and that retiring an unknown name is a harmless no-op.
func TestReloadRetireRemovesShard(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox"}, 0)
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs"}, 1000)
	rl, done := loadReloadable(t, dir)
	defer done()

	if !servedHasRange(t, rl, 1000, 2000) {
		t.Fatalf("second shard's docs not served at startup")
	}
	if !rl.retire("b.tsumugi") {
		t.Fatalf("retire reported the shard was not served")
	}
	if rl.broker.NumShards() != 1 {
		t.Fatalf("after retire shards = %d, want 1", rl.broker.NumShards())
	}
	if servedHasRange(t, rl, 1000, 2000) {
		t.Fatalf("retired shard's docs still served")
	}
	if rl.retire("b.tsumugi") {
		t.Fatalf("retiring an already-retired shard reported a removal")
	}
	if rl.retire("nope.tsumugi") {
		t.Fatalf("retiring an unknown shard reported a removal")
	}
}

// TestReloadSync checks the directory sync publishes new files and retires gone ones in one
// sweep, the operation the poll and the /admin/reload endpoint run.
func TestReloadSync(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox"}, 0)
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs"}, 1000)
	rl, done := loadReloadable(t, dir)
	defer done()

	// A no-change sync publishes and retires nothing.
	if pub, ret, err := rl.sync(); err != nil || pub != 0 || ret != 0 {
		t.Fatalf("idle sync = (%d, %d, %v), want (0, 0, nil)", pub, ret, err)
	}

	// Add one shard, remove another, then sync: one published, one retired.
	writeShard(t, filepath.Join(dir, "c.tsumugi"), []string{"swift brown hare"}, 2000)
	if err := os.Remove(filepath.Join(dir, "a.tsumugi")); err != nil {
		t.Fatalf("remove: %v", err)
	}
	pub, ret, err := rl.sync()
	if err != nil {
		t.Fatalf("sync: %v", err)
	}
	if pub != 1 || ret != 1 {
		t.Fatalf("sync = (published %d, retired %d), want (1, 1)", pub, ret)
	}
	if rl.numServed() != 2 {
		t.Fatalf("after sync served = %d, want 2 (b and c)", rl.numServed())
	}
	if servedHasRange(t, rl, 0, 1000) {
		t.Fatalf("removed shard a still served after sync")
	}
	if !servedHasRange(t, rl, 2000, 3000) {
		t.Fatalf("added shard c not served after sync")
	}
}

// TestReloadRefusesMismatchedShard checks a publish of a shard recorded under a different
// analyzer is refused and the shard is not served, the same silent-wrong-results guard the
// startup loader applies, applied on the live path.
func TestReloadRefusesMismatchedShard(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"brown fox"}, 0)
	rl, done := loadReloadable(t, dir)
	defer done()

	writeShardHash(t, filepath.Join(dir, "bad.tsumugi"), []string{"brown bear"}, 1000, queryAnalyzerHash()^0xFF)
	err := rl.publish("bad.tsumugi")
	if !errors.Is(err, collection.ErrAnalyzerMismatch) {
		t.Fatalf("publish error = %v, want ErrAnalyzerMismatch", err)
	}
	if rl.broker.NumShards() != 1 {
		t.Fatalf("refused shard was served: shards = %d", rl.broker.NumShards())
	}
	// A sync over a directory holding the bad shard publishes the good ones and reports the
	// bad shard's error rather than aborting.
	writeShard(t, filepath.Join(dir, "good.tsumugi"), []string{"brown hare"}, 2000)
	pub, _, serr := rl.sync()
	if !errors.Is(serr, collection.ErrAnalyzerMismatch) {
		t.Fatalf("sync error = %v, want ErrAnalyzerMismatch", serr)
	}
	if pub != 1 {
		t.Fatalf("sync published %d, want 1 (good.tsumugi despite bad.tsumugi)", pub)
	}
}

// TestAdminReloadEndpoint exercises the HTTP admin surface: POST /admin/publish adds a
// shard, POST /admin/reload syncs the directory, POST /admin/retire removes one, a GET is
// rejected, and a server with no reloader returns 404.
func TestAdminReloadEndpoint(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox"}, 0)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)
	broker, pl, rl, err := loadServe(dir, modelPath)
	if err != nil {
		t.Fatalf("loadServe: %v", err)
	}
	defer func() { _ = broker.Close() }()

	srv := &httpServer{broker: broker, pipeline: pl, admission: nil, reloader: rl}
	mux := http.NewServeMux()
	mux.HandleFunc("/admin/reload", srv.reload)
	mux.HandleFunc("/admin/publish", srv.publish)
	mux.HandleFunc("/admin/retire", srv.retire)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// A GET is rejected: admin endpoints mutate state.
	getResp, err := http.Get(ts.URL + "/admin/reload")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	_ = getResp.Body.Close()
	if getResp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /admin/reload = %d, want 405", getResp.StatusCode)
	}

	// Publish a freshly built shard by name.
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs"}, 1000)
	pubResp := postReload(t, ts.URL+"/admin/publish?shard=b.tsumugi")
	if pubResp.Published != 1 || pubResp.Shards != 2 {
		t.Fatalf("publish response = %+v, want published 1 shards 2", pubResp)
	}

	// Build a third shard, drop the first, and reload the whole directory.
	writeShard(t, filepath.Join(dir, "c.tsumugi"), []string{"swift brown hare"}, 2000)
	if err := os.Remove(filepath.Join(dir, "a.tsumugi")); err != nil {
		t.Fatalf("remove: %v", err)
	}
	relResp := postReload(t, ts.URL+"/admin/reload")
	if relResp.Published != 1 || relResp.Retired != 1 || relResp.Shards != 2 {
		t.Fatalf("reload response = %+v, want published 1 retired 1 shards 2", relResp)
	}

	// Retire by name.
	retResp := postReload(t, ts.URL+"/admin/retire?shard=c.tsumugi")
	if retResp.Retired != 1 || retResp.Shards != 1 {
		t.Fatalf("retire response = %+v, want retired 1 shards 1", retResp)
	}

	// A server with no reloader wired returns 404 on the admin path.
	bare := &httpServer{broker: broker, pipeline: pl}
	bmux := http.NewServeMux()
	bmux.HandleFunc("/admin/reload", bare.reload)
	bts := httptest.NewServer(bmux)
	defer bts.Close()
	noResp, err := http.Post(bts.URL+"/admin/reload", "", nil)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_ = noResp.Body.Close()
	if noResp.StatusCode != http.StatusNotFound {
		t.Fatalf("admin without reloader = %d, want 404", noResp.StatusCode)
	}
}

// postReload POSTs to an admin URL and decodes the reload response.
func postReload(t *testing.T, url string) reloadResponse {
	t.Helper()
	resp, err := http.Post(url, "", nil)
	if err != nil {
		t.Fatalf("post %s: %v", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("post %s = %d, want 200", url, resp.StatusCode)
	}
	var out reloadResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return out
}
