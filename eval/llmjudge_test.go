package eval

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestLLMJudgeWireFormat exercises the LLM judge against a fake chat-completions server,
// the path a real model would take without a live model: it checks the judge posts the
// UMBRELA prompt as a user message at temperature zero, carries the bearer token, and
// parses the model's reply into a grade. The fake server stands in for the model so the
// wire contract is tested deterministically and the live-LLM path stays out of CI.
func TestLLMJudgeWireFormat(t *testing.T) {
	var gotAuth string
	var gotReq chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotReq); err != nil {
			t.Errorf("decode request: %v", err)
		}
		resp := chatResponse{}
		resp.Choices = append(resp.Choices, struct {
			Message chatMessage `json:"message"`
		}{Message: chatMessage{Role: "assistant", Content: "M: 3 T: 2\nFinal score: 3"}})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	j := NewLLMJudge(srv.URL, "secret-key", "judge-model", WithHTTPClient(srv.Client()))
	g, err := j.Grade(context.Background(), "go memory model", Passage{Title: "Go memory model", Body: "happens-before"})
	if err != nil {
		t.Fatalf("Grade: %v", err)
	}
	if g != GradePerfect {
		t.Fatalf("grade = %d, want %d", g, GradePerfect)
	}
	if gotAuth != "Bearer secret-key" {
		t.Errorf("auth header = %q, want bearer", gotAuth)
	}
	if gotReq.Model != "judge-model" {
		t.Errorf("model = %q, want judge-model", gotReq.Model)
	}
	if gotReq.Temperature != 0 {
		t.Errorf("temperature = %v, want 0 for reproducible labels", gotReq.Temperature)
	}
	if len(gotReq.Messages) != 1 || !strings.Contains(gotReq.Messages[0].Content, "go memory model") {
		t.Errorf("request did not carry the UMBRELA prompt with the query: %+v", gotReq.Messages)
	}
}

func TestLLMJudgeNon200IsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "model overloaded", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	j := NewLLMJudge(srv.URL, "", "m", WithHTTPClient(srv.Client()))
	if _, err := j.Grade(context.Background(), "q", Passage{Body: "b"}); err == nil {
		t.Fatal("expected an error on a non-200 response, a failed judgment must not be a silent zero")
	}
}

func TestNewLLMJudgeFromEnv(t *testing.T) {
	t.Setenv(EnvJudgeURL, "")
	if _, ok := NewLLMJudgeFromEnv(); ok {
		t.Fatal("with no URL set, the env judge must report false so the caller falls back")
	}
	t.Setenv(EnvJudgeURL, "http://localhost:1234/v1")
	t.Setenv(EnvJudgeModel, "qwen")
	j, ok := NewLLMJudgeFromEnv()
	if !ok {
		t.Fatal("with a URL set, the env judge must be built")
	}
	if j.model != "qwen" || j.baseURL != "http://localhost:1234/v1" {
		t.Fatalf("env judge misconfigured: model=%q url=%q", j.model, j.baseURL)
	}
}
