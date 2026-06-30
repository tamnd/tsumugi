package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Environment variables the LLM judge reads, so an operator points the harness at a
// judge model without a code change and CI leaves them unset, which is what keeps the
// live-LLM path out of the deterministic test run.
const (
	EnvJudgeURL   = "TSUMUGI_JUDGE_URL"
	EnvJudgeKey   = "TSUMUGI_JUDGE_KEY"
	EnvJudgeModel = "TSUMUGI_JUDGE_MODEL"
)

// LLMJudge grades a query-passage pair by calling an OpenAI-compatible chat-completions
// endpoint with the UMBRELA prompt, the LLM judge doc 14 builds the bulk of the labels
// with. It speaks the chat-completions wire format every local and hosted server in the
// stack already exposes, sends the prompt at temperature zero so the same pair grades the
// same way across runs, and parses the model's reply into a grade. The harness keeps it
// behind the gold-set agreement check, so its labels are trusted only while they reproduce
// human judgment, never on the strength of being a model's output.
type LLMJudge struct {
	client      *http.Client
	baseURL     string
	apiKey      string
	model       string
	temperature float64
}

// LLMOption configures an LLMJudge at construction, the seam a test swaps the HTTP
// client through to point the judge at an httptest server instead of a live model.
type LLMOption func(*LLMJudge)

// WithHTTPClient sets the HTTP client the judge calls through, the hook a test uses to
// route requests at a fake server and a deployment uses to set a timeout or a transport.
func WithHTTPClient(c *http.Client) LLMOption {
	return func(j *LLMJudge) { j.client = c }
}

// WithTemperature overrides the sampling temperature, which defaults to zero. Zero is
// the reproducible setting the committed labels need; a caller raises it only to study
// the judge's variance, never for a benchmark.
func WithTemperature(t float64) LLMOption {
	return func(j *LLMJudge) { j.temperature = t }
}

// NewLLMJudge builds a judge against the chat-completions endpoint at baseURL using the
// named model, authenticating with apiKey when it is non-empty. baseURL is the API root,
// the part before /chat/completions, so the same constructor reaches a local server and a
// hosted one by their base alone.
func NewLLMJudge(baseURL, apiKey, model string, opts ...LLMOption) *LLMJudge {
	j := &LLMJudge{
		client:  &http.Client{Timeout: 60 * time.Second},
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
	}
	for _, o := range opts {
		o(j)
	}
	return j
}

// NewLLMJudgeFromEnv builds a judge from the TSUMUGI_JUDGE_* environment, returning false
// when the URL is unset, which is the signal the caller has no live judge configured and
// should fall back to the deterministic grader. CI leaves the variables unset and gets
// false, so the live-LLM path never runs in the deterministic suite, while an operator who
// sets them gets a judge wired to their model.
func NewLLMJudgeFromEnv(opts ...LLMOption) (*LLMJudge, bool) {
	url := os.Getenv(EnvJudgeURL)
	if url == "" {
		return nil, false
	}
	model := os.Getenv(EnvJudgeModel)
	if model == "" {
		model = "default"
	}
	return NewLLMJudge(url, os.Getenv(EnvJudgeKey), model, opts...), true
}

// chatRequest and chatResponse are the slice of the OpenAI chat-completions schema the
// judge uses: a model, a single user message holding the UMBRELA prompt, and a
// temperature on the way out, and the first choice's message content on the way back.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// Grade sends the UMBRELA prompt for the pair to the model and parses the reply into a
// grade. It posts the chat-completions request, reads the first choice, and runs the
// reply through ParseGrade, so a model that reasons before it answers still yields a
// grade and a model that returns no integer is an error rather than a silent zero. A
// transport error, a non-200 status, or an empty choice list is returned to the caller,
// since a failed judgment must not be mistaken for an irrelevant label.
func (j *LLMJudge) Grade(ctx context.Context, query string, p Passage) (Grade, error) {
	reqBody, err := json.Marshal(chatRequest{
		Model:       j.model,
		Messages:    []chatMessage{{Role: "user", Content: UmbrelaPrompt(query, p)}},
		Temperature: j.temperature,
	})
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, j.baseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if j.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+j.apiKey)
	}
	resp, err := j.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("eval: judge returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var cr chatResponse
	if err := json.Unmarshal(body, &cr); err != nil {
		return 0, fmt.Errorf("eval: decode judge response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return 0, fmt.Errorf("eval: judge returned no choices")
	}
	return ParseGrade(cr.Choices[0].Message.Content)
}
