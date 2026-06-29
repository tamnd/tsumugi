package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/spf13/cobra"
	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/query"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// queryAnalyzerHash is the hash of the analyzer the broker analyzes queries with, the
// package-level lexical.Analyze. A shard or a collection built with any other analyzer
// cannot be queried consistently, so startup compares this against the recorded hash and
// refuses a mismatch rather than returning silently wrong results.
func queryAnalyzerHash() uint64 { return lexical.DefaultAnalyzer.Hash() }

func newServeCmd() *cobra.Command {
	var (
		dir         string
		addr        string
		modelP      string
		timeout     time.Duration
		cacheSize   int
		maxInFlight int
	)
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Serve ranked search over a directory of shards",
		Long: "serve opens a directory of .tsumugi shards and answers ranked queries over\n" +
			"HTTP. It loads the routing index and the fleet-wide statistics from the\n" +
			"collection's index.tsm artifact, falling back to scanning the shards when no\n" +
			"artifact is present. Each request fans out to the shards that can contribute,\n" +
			"gathers their candidates, and runs one global rerank, so the merged top-k is the\n" +
			"result a single index over every shard would give.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if dir == "" {
				return fmt.Errorf("a shard directory is required: pass --dir")
			}
			if modelP == "" {
				return fmt.Errorf("a ranking model is required: pass --model")
			}
			broker, pl, err := openCollection(dir, modelP)
			if err != nil {
				return err
			}
			defer func() { _ = broker.Close() }()

			// Wire a result cache when one is sized, so the head of the heavy-tailed query
			// distribution serves from cache without re-running the cascade. A size of zero
			// leaves the broker cacheless, running every query through the cascade.
			if cacheSize > 0 {
				broker.SetResultCache(search.NewResultCache(cacheSize))
			}

			// Bound the in-flight searches so the broker degrades by rejecting rather than
			// collapsing under unbounded concurrency. A request that cannot get a slot is
			// answered 503 fast rather than queued into latency it would miss its deadline in.
			// A zero capacity disables the gate, leaving the broker unbounded.
			adm := search.NewAdmission(maxInFlight)
			srv := &httpServer{broker: broker, pipeline: pl, timeout: timeout, admission: adm}
			mux := http.NewServeMux()
			mux.HandleFunc("/search", srv.search)
			mux.HandleFunc("/healthz", srv.health)

			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "serving %d shards (%d docs) on %s\n",
				broker.NumShards(), broker.Stats().DocCount, addr)
			return serveHTTP(cmd.Context(), addr, mux, adm, timeout)
		},
	}
	cmd.Flags().StringVar(&dir, "dir", "", "directory of .tsumugi shards to serve")
	cmd.Flags().StringVar(&addr, "addr", ":8080", "address to listen on")
	cmd.Flags().StringVar(&modelP, "model", "", "trained ranking model file")
	cmd.Flags().DurationVar(&timeout, "timeout", 10*time.Millisecond, "per-request deadline")
	cmd.Flags().IntVar(&cacheSize, "cache", 0, "result cache capacity in entries (0 disables the cache)")
	cmd.Flags().IntVar(&maxInFlight, "max-inflight", 0, "maximum concurrent in-flight searches (0 disables admission control)")
	return cmd
}

// serveHTTP runs the broker's HTTP server until the context is cancelled, then drains the
// in-flight searches before returning, so a deploy or a restart does not drop the queries
// that were in flight when the shutdown signal arrived. On cancellation it stops admitting
// new searches, stops the listener from accepting new connections, and waits for the
// admission gate to drain, bounded by one request deadline so a stuck search cannot hang
// the shutdown. A nil context runs the server with no drain, the bare-listen behavior.
func serveHTTP(ctx context.Context, addr string, h http.Handler, adm *search.Admission, timeout time.Duration) error {
	srv := &http.Server{Addr: addr, Handler: h}
	if ctx == nil {
		return srv.ListenAndServe()
	}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	select {
	case err := <-errc:
		return err // the listener failed on its own (bad addr, port in use)
	case <-ctx.Done():
	}
	// Stop admitting, then drain. The drain bound is the request deadline, doubled so the
	// last admitted search has a full deadline to finish plus slack for the socket write,
	// with a floor for a zero-deadline server so the drain still terminates.
	bound := 2 * timeout
	if bound <= 0 {
		bound = 5 * time.Second
	}
	dctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()
	_ = adm.Drain(dctx)
	return srv.Shutdown(dctx)
}

// openCollection opens every shard in a directory, loads the model, wires a broker
// over them, and builds the query-understanding pipeline the broker runs each query
// through. Each shard and the broker share the same compiled model so a document scores
// identically wherever it is reranked, and the pipeline is built from the same open
// shards so the corrector's dictionary and the dense plane's dimension match the fleet
// the broker serves.
func openCollection(dir, modelPath string) (*search.Broker, *pipeline, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open model: %w", err)
	}
	ens, err := rank.LoadEnsemble(f)
	_ = f.Close()
	if err != nil {
		return nil, nil, fmt.Errorf("load model: %w", err)
	}
	model := ens.Compile()

	shards, broker, err := openShards(dir, model)
	if err != nil {
		return nil, nil, err
	}
	return broker, buildPipeline(shards), nil
}

// openShards opens the directory's shards and wires a broker over them, returning the
// shard slice alongside the broker so the caller can build the query pipeline from the
// same open shards. It prefers the persisted collection artifact, which carries the
// manifest, the fleet-wide statistics, and the routing index, so the broker starts
// without rescanning every shard's vocabulary, which is what lets serve start in time at
// fleet scale; a collection built before the artifact existed has none, so it falls back
// to the glob scan.
func openShards(dir string, model *rank.Model) ([]*search.Shard, *search.Broker, error) {
	if ix, err := collection.LoadIndex(dir); err == nil {
		// The manifest records the collection-wide analyzer hash in one place, so the
		// broker verifies it once here rather than opening every shard's footer.
		if h := ix.AnalyzerHash; h != 0 && h != queryAnalyzerHash() {
			return nil, nil, fmt.Errorf("%w: collection is %#016x, broker analyzer is %#016x",
				collection.ErrAnalyzerMismatch, h, queryAnalyzerHash())
		}
		return shardsFromIndex(dir, model, ix)
	} else if !os.IsNotExist(err) {
		return nil, nil, fmt.Errorf("load index: %w", err)
	}

	paths, err := filepath.Glob(filepath.Join(dir, "*.tsumugi"))
	if err != nil {
		return nil, nil, err
	}
	if len(paths) == 0 {
		return nil, nil, fmt.Errorf("no .tsumugi shards in %s", dir)
	}
	shards := make([]*search.Shard, 0, len(paths))
	closeAll := func() {
		for _, opened := range shards {
			_ = opened.Close()
		}
	}
	for _, p := range paths {
		s, err := search.OpenShard(p, newCascade(model))
		if err != nil {
			closeAll()
			return nil, nil, fmt.Errorf("open shard %s: %w", p, err)
		}
		// No manifest in this directory, so verify each shard's own recorded analyzer
		// against the broker's. A shard built before the hash was recorded reports
		// nothing and is left to the operator, the same unknown the reader returns.
		if h, ok := s.AnalyzerHash(); ok && h != queryAnalyzerHash() {
			_ = s.Close()
			closeAll()
			return nil, nil, fmt.Errorf("%w: shard %s is %#016x, broker analyzer is %#016x",
				collection.ErrAnalyzerMismatch, p, h, queryAnalyzerHash())
		}
		shards = append(shards, s)
	}
	broker := search.NewBroker(shards, newCascade(model))
	if err := broker.CheckModel(); err != nil {
		closeAll()
		return nil, nil, err
	}
	return shards, broker, nil
}

// shardsFromIndex opens the shards the artifact names, in the artifact's order, and
// wires a broker over them with the persisted routing index and statistics. Opening in
// the manifest's order is what keeps the routing index's shard ids aligned with the
// shard slice, so a routed shard id always points at the shard the artifact recorded it
// for.
func shardsFromIndex(dir string, model *rank.Model, ix *collection.Index) ([]*search.Shard, *search.Broker, error) {
	shards := make([]*search.Shard, 0, len(ix.Shards))
	for _, info := range ix.Shards {
		p := filepath.Join(dir, filepath.Base(info.Path))
		s, err := search.OpenShard(p, newCascade(model))
		if err != nil {
			for _, opened := range shards {
				_ = opened.Close()
			}
			return nil, nil, fmt.Errorf("open shard %s: %w", p, err)
		}
		shards = append(shards, s)
	}
	routing := search.NewRoutingIndex(ix.RoutingMap(), ix.AlwaysRouted(), len(shards))
	stats := search.GlobalStats{
		DocCount:    ix.Stats.DocCount,
		TokenCount:  ix.Stats.TokenCount,
		AvgDocLen:   ix.Stats.AvgDocLen,
		AvgFieldLen: ix.Stats.AvgFieldLen,
	}
	broker := search.NewBrokerWith(shards, newCascade(model), routing, stats)
	if err := broker.CheckModel(); err != nil {
		for _, opened := range shards {
			_ = opened.Close()
		}
		return nil, nil, err
	}
	return shards, broker, nil
}

func newCascade(model *rank.Model) *rank.Cascade {
	return rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
}

// httpServer answers search requests over a broker with a per-request deadline. It runs
// each raw query through the query-understanding pipeline once, at the broker, before
// fanning the parsed query out to the shards, the analyze-once rule the pipeline owns.
type httpServer struct {
	broker   *search.Broker
	pipeline *pipeline
	timeout  time.Duration

	// admission bounds the in-flight searches. The slot is acquired at the top of the
	// handler and released with a deferred Release that covers the response encode and the
	// socket write, the hold-for-the-whole-search discipline doc 11 pins. A disabled gate
	// (capacity zero) admits everything, so the field is never nil but may be a no-op.
	admission *search.Admission
}

type searchResponse struct {
	Hits   []hitJSON `json:"hits"`
	Shards int       `json:"shards"`
	TookMs float64   `json:"took_ms"`

	// Lang is the language the detector routed analysis on, empty for the default
	// chain. Corrected is true when spell correction auto-substituted a term, and
	// Suggestion carries the did-you-mean rendering when one was offered rather than
	// applied, so a caller can show "showing results for" or "did you mean".
	// Completeness tells the client whether the top-k is over every contributing shard
	// or a subset, so a partial answer (a shard dropped at the deadline) is reported as
	// a degraded result rather than passed off as complete.
	Completeness completenessJSON `json:"completeness"`

	Lang       string `json:"lang,omitempty"`
	Corrected  bool   `json:"corrected,omitempty"`
	Suggestion string `json:"suggestion,omitempty"`

	// Cached is true when the result was served from the broker's result cache rather than
	// computed by the cascade, so a client or an operator can see the cache hit rate in the
	// responses themselves.
	Cached bool `json:"cached,omitempty"`
}

// completenessJSON is the response's completeness indicator: complete is true when
// every contributing shard responded by the deadline, and the two counts let a client
// see the fraction reached (doc 11, "Failure modes and partial results").
type completenessJSON struct {
	Complete    bool `json:"complete"`
	ShardsTotal int  `json:"shards_total"`
	ShardsOK    int  `json:"shards_ok"`
	// Degraded names the rung of the degradation ladder the broker served the query at,
	// "none" for a full-quality result, so a client and an operator can see that a result
	// was, say, lexical-only or missing the lowest-static-rank shards by design. It is the
	// quality-reduction half of the metadata, independent of the deadline-drop Complete
	// reports (doc 11, "The degradation order").
	Degraded string `json:"degraded"`
}

type hitJSON struct {
	DocID uint32  `json:"doc_id"`
	Score float64 `json:"score"`
}

func (s *httpServer) search(w http.ResponseWriter, r *http.Request) {
	// Acquire an admission slot first, before any work, and hold it with a deferred Release
	// at the top so it covers the whole search including the response encode and the socket
	// write below, the hold-for-the-whole-search discipline that keeps the slot count a true
	// bound on in-flight searches (doc 11). A request that cannot get a slot is rejected fast
	// with 503 rather than queued into latency it would miss its deadline in.
	slot := s.admission.Acquire()
	if slot == nil {
		w.Header().Set("Retry-After", "1")
		http.Error(w, "broker at capacity", http.StatusServiceUnavailable)
		return
	}
	defer slot.Release()

	k := 10
	if v := r.URL.Query().Get("k"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			k = n
		}
	}
	// Understand the query once here, at the broker: detect its language, analyze it
	// with that language's chain, correct and expand and dense-encode it, then ship the
	// single parsed result to the shards, so a shard never re-runs the analysis chain.
	pq := s.pipeline.parse(r.URL.Query().Get("q"))
	resp := searchResponse{
		Shards:     s.broker.NumShards(),
		Lang:       pq.Lang,
		Corrected:  pq.Corrected,
		Suggestion: pq.Suggestion,
	}
	if pq.Empty() {
		resp.Hits = []hitJSON{}
		// A query with nothing to route is trivially complete: no shard was dropped.
		resp.Completeness.Complete = true
		resp.Completeness.Degraded = search.DegradeNone.String()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	q := toQuery(pq, k)
	ctx := r.Context()
	if s.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.timeout)
		defer cancel()
	}
	start := time.Now()
	// Serve from the result cache when one is wired, falling back to the budgeted search
	// on a miss. A miss serves at the degradation level the remaining deadline budget calls
	// for, so a request entering with little budget left answers within budget at a known
	// lower quality rather than overrunning the per-request deadline (doc 11, degradation
	// order); a hit serves the cached full-quality result regardless of the current budget.
	res, hit := s.broker.SearchCached(ctx, q)
	resp.Cached = hit
	resp.TookMs = float64(time.Since(start).Microseconds()) / 1000
	resp.Completeness = completenessJSON{
		Complete:    res.Complete(),
		ShardsTotal: res.ShardsTotal,
		ShardsOK:    res.ShardsOK,
		Degraded:    res.Degraded.String(),
	}
	resp.Hits = make([]hitJSON, len(res.Hits))
	for i, h := range res.Hits {
		resp.Hits[i] = hitJSON{DocID: h.DocID, Score: h.Score}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// toQuery translates an understood query into the broker's retrieval query: the lexical
// plane retrieves on the expansion-folded retrieval terms, and the dense plane takes the
// query's dense vector decoded from its wire form, nil when the dense plane is off so the
// shards skip it. The analysis already ran at the broker, so the shards take the
// pre-analyzed terms and never re-run the chain.
func toQuery(pq *query.ParsedQuery, k int) search.Query {
	return search.Query{
		Terms:  pq.RetrievalTerms(),
		Vector: query.DecodeDenseVec(pq.DenseVec),
		K:      k,
	}
}

func (s *httpServer) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status": "ok",
		"shards": s.broker.NumShards(),
		"docs":   s.broker.Stats().DocCount,
		// in_flight against capacity is the load metric an operator watches to see when the
		// broker is near capacity and shedding load (doc 11, "Metrics"). capacity is zero
		// when admission control is disabled.
		"in_flight": s.admission.InFlight(),
		"capacity":  s.admission.Cap(),
	})
}
