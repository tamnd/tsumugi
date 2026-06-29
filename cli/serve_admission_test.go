package cli

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/tamnd/tsumugi/search"
)

// newAdmissionServer builds a tiny two-shard serve handler with an explicit admission
// gate, returning the running test server and the gate so a test can occupy slots out of
// band and observe the handler's reject-fast behavior.
func newAdmissionServer(t *testing.T, capacity int) (*httptest.Server, *search.Admission) {
	t.Helper()
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "a.tsumugi"), []string{"the quick brown fox", "lazy brown dog"}, 0)
	writeShard(t, filepath.Join(dir, "b.tsumugi"), []string{"brown bear runs", "swift brown hare"}, 1000)
	modelPath := filepath.Join(dir, "model.bin")
	writeModel(t, modelPath)

	broker, pl, err := openCollection(dir, modelPath)
	if err != nil {
		t.Fatalf("openCollection: %v", err)
	}
	t.Cleanup(func() { _ = broker.Close() })

	adm := search.NewAdmission(capacity)
	srv := &httpServer{broker: broker, pipeline: pl, timeout: 0, admission: adm}
	ts := httptest.NewServer(http.HandlerFunc(srv.search))
	t.Cleanup(ts.Close)
	return ts, adm
}

// TestServeRejectsWhenAtCapacity occupies the only slot out of band, so the broker is at
// capacity, and checks the handler answers 503 fast rather than blocking, then admits
// again once the slot is freed. This proves the handler acquires before doing work and
// rejects rather than queues, the doc-11 admission discipline.
func TestServeRejectsWhenAtCapacity(t *testing.T) {
	ts, adm := newAdmissionServer(t, 1)

	held := adm.Acquire() // occupy the only slot, as a long-running search would
	if held == nil {
		t.Fatal("could not acquire the only slot")
	}

	resp, err := http.Get(ts.URL + "?q=brown&k=4")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		_ = resp.Body.Close()
		t.Fatalf("status = %d, want 503 when at capacity", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra == "" {
		t.Error("503 response should carry a Retry-After header")
	}
	_ = resp.Body.Close()

	// Free the slot; the next request must be admitted and served.
	held.Release()
	resp2, err := http.Get(ts.URL + "?q=brown&k=4")
	if err != nil {
		t.Fatalf("get after release: %v", err)
	}
	defer func() { _ = resp2.Body.Close() }()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("status after release = %d, want 200", resp2.StatusCode)
	}
	var got searchResponse
	if err := json.NewDecoder(resp2.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Hits) == 0 {
		t.Fatal("served request returned no hits")
	}
}

// TestServeReleasesSlotAfterWrite checks the handler holds the slot for the whole search
// and releases it once the response is written: after a served request the in-flight count
// returns to zero, so the slot count is a true bound and not leaked. Running a sequence of
// requests through a capacity-1 gate proves each one fully released before the next, which
// could not happen if the handler leaked or double-counted slots.
func TestServeReleasesSlotAfterWrite(t *testing.T) {
	ts, adm := newAdmissionServer(t, 1)
	for i := 0; i < 5; i++ {
		resp, err := http.Get(ts.URL + "?q=brown&k=4")
		if err != nil {
			t.Fatalf("get %d: %v", i, err)
		}
		if resp.StatusCode != http.StatusOK {
			_ = resp.Body.Close()
			t.Fatalf("request %d status = %d, want 200; a leaked slot would 503 here", i, resp.StatusCode)
		}
		_ = resp.Body.Close()
	}
	if got := adm.InFlight(); got != 0 {
		t.Fatalf("in-flight after serving = %d, want 0 (slot leaked)", got)
	}
}

// TestServeDisabledAdmissionAlwaysAdmits checks a zero-capacity gate never rejects: the
// handler serves every request, the unbounded default.
func TestServeDisabledAdmissionAlwaysAdmits(t *testing.T) {
	ts, _ := newAdmissionServer(t, 0)
	for i := 0; i < 20; i++ {
		resp, err := http.Get(ts.URL + "?q=brown&k=4")
		if err != nil {
			t.Fatalf("get %d: %v", i, err)
		}
		ok := resp.StatusCode == http.StatusOK
		_ = resp.Body.Close()
		if !ok {
			t.Fatalf("disabled gate returned %d on request %d", resp.StatusCode, i)
		}
	}
}

// TestServeHTTPDrainsOnShutdown checks serveHTTP returns when its context is cancelled and
// the admission gate is drained, the clean-shutdown path a deploy relies on.
func TestServeHTTPDrainsOnShutdown(t *testing.T) {
	adm := search.NewAdmission(2)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(200) })

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serveHTTP(ctx, "127.0.0.1:0", mux, adm, 10*time.Millisecond) }()

	// Give the listener a moment to come up, then signal shutdown.
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case <-done:
		// serveHTTP returned after draining; the gate is closed so it rejects new work.
		if adm.Acquire() != nil {
			t.Fatal("admission gate should be closed after a drained shutdown")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("serveHTTP did not return after context cancel")
	}
}
