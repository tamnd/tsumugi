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
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

func newServeCmd() *cobra.Command {
	var (
		dir     string
		addr    string
		modelP  string
		timeout time.Duration
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
			broker, err := openCollection(dir, modelP)
			if err != nil {
				return err
			}
			defer func() { _ = broker.Close() }()

			srv := &httpServer{broker: broker, timeout: timeout}
			mux := http.NewServeMux()
			mux.HandleFunc("/search", srv.search)
			mux.HandleFunc("/healthz", srv.health)

			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "serving %d shards (%d docs) on %s\n",
				broker.NumShards(), broker.Stats().DocCount, addr)
			return http.ListenAndServe(addr, mux)
		},
	}
	cmd.Flags().StringVar(&dir, "dir", "", "directory of .tsumugi shards to serve")
	cmd.Flags().StringVar(&addr, "addr", ":8080", "address to listen on")
	cmd.Flags().StringVar(&modelP, "model", "", "trained ranking model file")
	cmd.Flags().DurationVar(&timeout, "timeout", 10*time.Millisecond, "per-request deadline")
	return cmd
}

// openCollection opens every shard in a directory, loads the model, and wires a
// broker over them. Each shard and the broker share the same compiled model so a
// document scores identically wherever it is reranked.
func openCollection(dir, modelPath string) (*search.Broker, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("open model: %w", err)
	}
	ens, err := rank.LoadEnsemble(f)
	_ = f.Close()
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	model := ens.Compile()

	// Prefer the persisted collection artifact: it carries the manifest, the fleet-wide
	// statistics, and the routing index, so the broker starts without rescanning every
	// shard's vocabulary, which is what lets serve start in time at fleet scale. A
	// collection built before the artifact existed has none, so fall back to the scan.
	if ix, err := collection.LoadIndex(dir); err == nil {
		return brokerFromIndex(dir, model, ix)
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("load index: %w", err)
	}

	paths, err := filepath.Glob(filepath.Join(dir, "*.tsumugi"))
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no .tsumugi shards in %s", dir)
	}
	shards := make([]*search.Shard, 0, len(paths))
	for _, p := range paths {
		s, err := search.OpenShard(p, newCascade(model))
		if err != nil {
			for _, opened := range shards {
				_ = opened.Close()
			}
			return nil, fmt.Errorf("open shard %s: %w", p, err)
		}
		shards = append(shards, s)
	}
	return search.NewBroker(shards, newCascade(model)), nil
}

// brokerFromIndex opens the shards the artifact names, in the artifact's order, and
// wires a broker over them with the persisted routing index and statistics. Opening in
// the manifest's order is what keeps the routing index's shard ids aligned with the
// shard slice, so a routed shard id always points at the shard the artifact recorded it
// for.
func brokerFromIndex(dir string, model *rank.Model, ix *collection.Index) (*search.Broker, error) {
	shards := make([]*search.Shard, 0, len(ix.Shards))
	for _, info := range ix.Shards {
		p := filepath.Join(dir, filepath.Base(info.Path))
		s, err := search.OpenShard(p, newCascade(model))
		if err != nil {
			for _, opened := range shards {
				_ = opened.Close()
			}
			return nil, fmt.Errorf("open shard %s: %w", p, err)
		}
		shards = append(shards, s)
	}
	routing := search.NewRoutingIndex(ix.RoutingMap(), ix.AlwaysRouted(), len(shards))
	stats := search.GlobalStats{
		DocCount:   ix.Stats.DocCount,
		TokenCount: ix.Stats.TokenCount,
		AvgDocLen:  ix.Stats.AvgDocLen,
	}
	return search.NewBrokerWith(shards, newCascade(model), routing, stats), nil
}

func newCascade(model *rank.Model) *rank.Cascade {
	return rank.NewCascade(&rank.Linear{RetrievalWeight: 1}, model)
}

// httpServer answers search requests over a broker with a per-request deadline.
type httpServer struct {
	broker  *search.Broker
	timeout time.Duration
}

type searchResponse struct {
	Hits   []hitJSON `json:"hits"`
	Shards int       `json:"shards"`
	TookMs float64   `json:"took_ms"`
}

type hitJSON struct {
	DocID uint32  `json:"doc_id"`
	Score float64 `json:"score"`
}

func (s *httpServer) search(w http.ResponseWriter, r *http.Request) {
	q := search.Query{
		Text: r.URL.Query().Get("q"),
		K:    10,
	}
	if k := r.URL.Query().Get("k"); k != "" {
		if v, err := strconv.Atoi(k); err == nil && v > 0 {
			q.K = v
		}
	}
	ctx := r.Context()
	if s.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.timeout)
		defer cancel()
	}
	start := time.Now()
	hits := s.broker.Search(ctx, q)
	resp := searchResponse{
		Hits:   make([]hitJSON, len(hits)),
		Shards: s.broker.NumShards(),
		TookMs: float64(time.Since(start).Microseconds()) / 1000,
	}
	for i, h := range hits {
		resp.Hits[i] = hitJSON{DocID: h.DocID, Score: h.Score}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *httpServer) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status": "ok",
		"shards": s.broker.NumShards(),
		"docs":   s.broker.Stats().DocCount,
	})
}
