package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// The serving tree the aggregator builds is in-process: NewAggregator takes a slice of
// Searcher values and fans a query across them by calling their methods directly. That is
// the whole tree on one machine, which caps the corpus at what one machine's shards can
// hold. This file is the seam that lets the same tree span machines: a Searcher on one
// machine is reached from another over HTTP, so an aggregator on a head node fans across
// brokers on many leaf nodes and the corpus grows with the fleet rather than with one box.
//
// The seam is the Searcher interface itself. A RemoteSearcher is a client that satisfies
// Searcher by forwarding each method to a peer over HTTP, and NewSearcherHandler is the
// server that answers those calls against a local Searcher. Because both sides speak the
// same interface, an aggregator cannot tell a remote child from a local one: a tree mixes
// in-process brokers and RemoteSearchers freely, and a RemoteSearcher can point at another
// machine's aggregator, so the tree nests across machines to any depth (doc 11, "The
// serving topology", spanning hosts).
//
// The wire form is JSON over HTTP, the same transport the serve command already speaks, so
// a deployment needs no extra dependency or daemon. The Searcher types marshal cleanly as
// JSON, so the encoding is the default field-name mapping with no hand-written codec. A
// deployment that needs a denser or faster wire swaps the codec behind this same seam
// without touching the aggregator or the broker; the interface is the contract, the JSON is
// an implementation detail of this one file.

// metaResponse is the per-subtree metadata a RemoteSearcher caches so the three static
// Searcher methods answer locally without a round trip per query. NumShards and NumDocs and
// the fleet statistics change only when the peer publishes or retires shards, far less often
// than queries arrive, so the client fetches them once at construction and on an explicit
// Refresh rather than on every call. Fetching them once is also what keeps the cross-machine
// merge exact at a fixed corpus: the aggregator divides one idf by the NumDocs it summed from
// these snapshots and pushes down the field averages it folded from these Stats, so as long
// as the snapshots describe the corpus the query runs against, the remote tree reproduces the
// monolith the same way the in-process tree does.
type metaResponse struct {
	NumShards int         `json:"num_shards"`
	NumDocs   uint64      `json:"num_docs"`
	Stats     GlobalStats `json:"stats"`

	// VectorDim and HasVector report the dense input dimension the subtree's shards agree on, so a
	// head node building the dense query encoder produces a vector of the width every leaf can read.
	// They ride the metadata snapshot because the dimension changes only when the fleet's vector
	// region changes, as rarely as the shard counts do, so a head reads it once at construction
	// alongside the rest. HasVector is false for a subtree with no dense plane, which leaves the
	// head's encoder off rather than encoding into a width no leaf carries.
	VectorDim int  `json:"vector_dim"`
	HasVector bool `json:"has_vector"`
}

// vocabEntry is one term of a subtree's vocabulary on the /vocab stream: the term and its
// document frequency summed across the subtree. The stream is newline-delimited JSON, one entry
// per line, so a head node reads a broker's whole vocabulary without either end buffering it all
// in memory, which matters because a broker's dictionary is large and the head merges many of
// them into one fleet-wide corrector.
type vocabEntry struct {
	Term string `json:"t"`
	DF   uint32 `json:"d"`
}

// vocabIterator is the optional capability a Searcher exposes to serve its vocabulary over
// /vocab: a node that can enumerate its terms (a broker over shards) implements it, and a node
// that cannot (a bare aggregator, whose vocabulary lives on its remote children) does not, so
// the handler answers /vocab only where it can.
type vocabIterator interface {
	ForEachTerm(fn func(term string, df uint32))
}

// vectorDimer is the optional capability a Searcher exposes to report its dense dimension on
// /meta, satisfied by a broker over shards that carry a vector region.
type vectorDimer interface {
	VectorDim() (int, bool)
}

// NewSearcherHandler exposes a local Searcher over HTTP as one node of a serving tree that
// spans machines. It answers the calls an aggregator makes on a child: POST /search runs
// SearchComplete over a JSON query and returns the JSON Results, POST /docfreqs sums the
// subtree's per-term document frequencies for a JSON term list, and GET /meta returns the
// subtree's shard and document counts, its fleet statistics, and its dense dimension so a
// client answers NumShards, NumDocs, Stats, and VectorDim without a round trip. It also
// answers GET /vocab, a newline-delimited stream of the subtree's terms and document
// frequencies that a head node folds into a fleet-wide corrector; a node that holds no
// vocabulary of its own answers 404 there. The handler is the
// server half of the RemoteSearcher seam: wrap any Searcher, a broker over shards or an
// aggregator over further nodes, and a parent aggregator on another machine fans across it
// as if it were local. Each handler runs against the request's context, so a client that
// cancels or times out a call cancels the work on this node too, the deadline propagating
// down the tree the same way it does in process.
func NewSearcherHandler(s Searcher) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		if !postOnly(w, r) {
			return
		}
		// The query and the result are the per-query hot-path messages, so they ride whichever
		// wire codec the client chose: the request's Content-Type picks the codec the handler
		// decodes the query with and answers the result in, so a binary client gets a binary
		// answer and a JSON client gets JSON, both off one handler. An unknown or missing type
		// falls back to JSON, the default wire (wire.go).
		codec := codecForContentType(r.Header.Get("Content-Type"))
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read query: "+err.Error(), http.StatusBadRequest)
			return
		}
		q, err := codec.decodeQuery(body)
		if err != nil {
			http.Error(w, "decode query: "+err.Error(), http.StatusBadRequest)
			return
		}
		out, err := codec.encodeResults(s.SearchComplete(r.Context(), q))
		if err != nil {
			http.Error(w, "encode results: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", codec.contentType())
		_, _ = w.Write(out)
	})
	mux.HandleFunc("/docfreqs", func(w http.ResponseWriter, r *http.Request) {
		if !postOnly(w, r) {
			return
		}
		var terms []string
		if err := json.NewDecoder(r.Body).Decode(&terms); err != nil {
			http.Error(w, "decode terms: "+err.Error(), http.StatusBadRequest)
			return
		}
		df := s.DocFreqs(r.Context(), terms)
		if df == nil {
			// A nil map marshals as JSON null, which the client would decode back to a nil map
			// either way, but an empty object is the honest "no frequencies" answer and keeps the
			// wire shape an object, so callers and logs see {} rather than null.
			df = map[string]uint32{}
		}
		writeRemoteJSON(w, df)
	})
	mux.HandleFunc("/meta", func(w http.ResponseWriter, r *http.Request) {
		m := metaResponse{NumShards: s.NumShards(), NumDocs: s.NumDocs(), Stats: s.Stats()}
		if vd, ok := s.(vectorDimer); ok {
			m.VectorDim, m.HasVector = vd.VectorDim()
		}
		writeRemoteJSON(w, m)
	})
	mux.HandleFunc("/vocab", func(w http.ResponseWriter, r *http.Request) {
		// /vocab streams the subtree's vocabulary as newline-delimited vocabEntry JSON so a head node
		// builds its corrector from the fleet without either side holding the whole dictionary at once.
		// A node that cannot enumerate its terms, a bare aggregator whose vocabulary lives on its remote
		// children, answers 404 rather than an empty stream, so a head dialing it knows to gather vocab
		// from that node's own children instead of mistaking absent for empty.
		vi, ok := s.(vocabIterator)
		if !ok {
			http.Error(w, "this node does not serve a vocabulary", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		vi.ForEachTerm(func(term string, df uint32) {
			_ = enc.Encode(vocabEntry{Term: term, DF: df})
		})
	})
	return mux
}

// RemoteSearcher is a Searcher that forwards to a peer over HTTP, the client half of the
// NewSearcherHandler seam. It lets an aggregator hold a child that lives on another machine:
// SearchComplete and DocFreqs each make one HTTP call to the peer, and NumShards, NumDocs,
// and Stats answer from a metadata snapshot fetched once at construction and refreshed on
// demand, so the per-query path is one round trip and the static methods are free. Because it
// satisfies Searcher, an aggregator fans across a RemoteSearcher exactly as it fans across an
// in-process broker, and a RemoteSearcher may point at another aggregator, so a tree spans
// any number of machines at any depth.
type RemoteSearcher struct {
	client  *http.Client
	baseURL string

	// codec is the wire codec the per-query /search call encodes the query with and decodes the
	// result with. It defaults to the JSON codec, so a RemoteSearcher built with no codec option
	// speaks the wire remote.go always spoke; WithBinaryWire swaps it for the dense binary codec.
	// The metadata, document-frequency, and vocabulary calls stay JSON regardless, since they run
	// off the per-query path where the density does not pay (wire.go).
	codec wireCodec

	// meta is the snapshot the static methods answer from, set at construction and replaced by
	// Refresh. It is read without a lock because it is fetched before the searcher is handed to
	// an aggregator and refreshed only by the owner between query waves, not concurrently with
	// the fan-out that reads it; a deployment that refreshes under live traffic guards it itself.
	meta metaResponse
}

// RemoteOption configures a RemoteSearcher at construction.
type RemoteOption func(*RemoteSearcher)

// WithHTTPClient sets the HTTP client the RemoteSearcher makes its calls with, so a
// deployment can pin connection pooling, keep-alives, and transport timeouts rather than
// taking the default client. A nil client is ignored, leaving the default.
func WithHTTPClient(c *http.Client) RemoteOption {
	return func(rs *RemoteSearcher) {
		if c != nil {
			rs.client = c
		}
	}
}

// WithBinaryWire switches the RemoteSearcher's per-query /search call to the dense binary wire
// codec instead of the default JSON: the query goes out and the result comes back in the compact
// binary form (wire.go), which writes a dense vector and the top-k scores as their raw bytes
// rather than as decimal text. The handler reads the request's content type and answers in the
// same codec, so a binary RemoteSearcher and a JSON one can both dial one handler. A deployment
// turns this on at both ends of a hop it controls; the off-hot-path metadata and vocabulary
// calls stay JSON either way.
func WithBinaryWire() RemoteOption {
	return func(rs *RemoteSearcher) { rs.codec = binaryCodec{} }
}

// NewRemoteSearcher dials a peer exposed by NewSearcherHandler and returns a Searcher that
// forwards to it. It fetches the peer's metadata once here so NumShards, NumDocs, and Stats
// answer locally, and it fails construction if the peer is unreachable so a tree is not built
// over a dead child that would silently drop every query; a child that dies after construction
// is handled at query time by SearchComplete reporting its whole subtree missed. The context
// bounds only the metadata fetch, not the searcher's lifetime.
func NewRemoteSearcher(ctx context.Context, baseURL string, opts ...RemoteOption) (*RemoteSearcher, error) {
	rs := &RemoteSearcher{client: http.DefaultClient, baseURL: strings.TrimRight(baseURL, "/"), codec: jsonCodec{}}
	for _, o := range opts {
		o(rs)
	}
	if err := rs.Refresh(ctx); err != nil {
		return nil, fmt.Errorf("remote searcher %s: %w", rs.baseURL, err)
	}
	return rs, nil
}

// Refresh re-fetches the peer's metadata snapshot, so a parent picks up a child that has
// published or retired shards since construction. A deployment calls it between query waves,
// off the serving path, because the static methods read the snapshot without a lock.
func (rs *RemoteSearcher) Refresh(ctx context.Context) error {
	var m metaResponse
	if err := rs.getJSON(ctx, "/meta", &m); err != nil {
		return err
	}
	rs.meta = m
	return nil
}

// NumShards reports the leaf shard count beneath the peer from the cached snapshot, the count
// an aggregator charges against this child when it is dropped at the deadline.
func (rs *RemoteSearcher) NumShards() int { return rs.meta.NumShards }

// NumDocs reports the document count beneath the peer from the cached snapshot, the N a parent
// sums into the fleet total it divides one shared idf by.
func (rs *RemoteSearcher) NumDocs() uint64 { return rs.meta.NumDocs }

// Stats reports the peer's fleet statistics from the cached snapshot, which a parent folds into
// the deployment-wide field averages it pushes back down so a partitioned remote tree's L2
// scores stay on one scale.
func (rs *RemoteSearcher) Stats() GlobalStats { return rs.meta.Stats }

// VectorDim reports the dense input dimension the peer's subtree agrees on and whether the dense
// plane is on, from the cached snapshot, so a head node building the dense query encoder produces
// a vector of the width every leaf beneath this peer can read. It is the remote counterpart to
// Broker.VectorDim, read once with the rest of the metadata rather than per query.
func (rs *RemoteSearcher) VectorDim() (int, bool) { return rs.meta.VectorDim, rs.meta.HasVector }

// Vocab streams the peer's vocabulary, calling fn for each term with its document frequency summed
// across the peer's subtree, so a head node feeds a fleet-wide corrector from this peer without
// either end buffering the whole dictionary. A peer that does not serve a vocabulary (a bare
// aggregator, answering 404) returns an error the head treats as "ask this peer's own children",
// not as an empty vocabulary. The context bounds the stream the same as any other call.
func (rs *RemoteSearcher) Vocab(ctx context.Context, fn func(term string, df uint32)) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rs.baseURL+"/vocab", nil)
	if err != nil {
		return err
	}
	resp, err := rs.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("/vocab: %s: %s", resp.Status, bytes.TrimSpace(body))
	}
	dec := json.NewDecoder(resp.Body)
	for {
		var e vocabEntry
		if err := dec.Decode(&e); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("/vocab: decode: %w", err)
		}
		fn(e.Term, e.DF)
	}
}

// DocFreqs asks the peer for each term's document frequency across its subtree. A failed call
// returns nil, which only loosens the idf the parent computes rather than corrupting a score,
// the same way the in-process aggregator treats a child it could not reach by the deadline.
func (rs *RemoteSearcher) DocFreqs(ctx context.Context, terms []string) map[string]uint32 {
	if len(terms) == 0 {
		return nil
	}
	var df map[string]uint32
	if err := rs.postJSON(ctx, "/docfreqs", terms, &df); err != nil {
		return nil
	}
	return df
}

// SearchComplete runs the query on the peer and returns its Results. An unreachable or
// timed-out peer is a dropped subtree, not an error to surface: it returns a Results that
// charges the peer's whole shard count to the total and nothing to the count reached, so the
// parent aggregator rolls the answer up as partial rather than passing a hole off as complete.
// This is the cross-machine form of the broker's own deadline drop, the honest upper bound on
// what was missed (doc 11, "Failure modes and partial results", composed across hosts).
func (rs *RemoteSearcher) SearchComplete(ctx context.Context, q Query) Results {
	body, err := rs.codec.encodeQuery(q)
	if err != nil {
		return Results{ShardsTotal: rs.meta.NumShards}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, rs.baseURL+"/search", bytes.NewReader(body))
	if err != nil {
		return Results{ShardsTotal: rs.meta.NumShards}
	}
	req.Header.Set("Content-Type", rs.codec.contentType())
	respBody, err := rs.doRaw(req)
	if err != nil {
		return Results{ShardsTotal: rs.meta.NumShards}
	}
	res, err := rs.codec.decodeResults(respBody)
	if err != nil {
		return Results{ShardsTotal: rs.meta.NumShards}
	}
	return res
}

func (rs *RemoteSearcher) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rs.baseURL+path, nil)
	if err != nil {
		return err
	}
	return rs.do(req, out)
}

func (rs *RemoteSearcher) postJSON(ctx context.Context, path string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, rs.baseURL+path, bytes.NewReader(buf))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return rs.do(req, out)
}

// doRaw runs a request and returns the whole response body, the codec-agnostic path the
// per-query /search call takes: the body may be JSON or the dense binary form, so the caller
// hands it to the wire codec rather than to a JSON decoder. A non-200 is an error carrying the
// status and a bounded snippet of the body, the same shape do reports.
func (rs *RemoteSearcher) doRaw(req *http.Request) ([]byte, error) {
	resp, err := rs.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("%s: %s: %s", req.URL.Path, resp.Status, bytes.TrimSpace(body))
	}
	return io.ReadAll(resp.Body)
}

func (rs *RemoteSearcher) do(req *http.Request, out any) error {
	resp, err := rs.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s: %s: %s", req.URL.Path, resp.Status, bytes.TrimSpace(body))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// postOnly rejects a non-POST request on a route that mutates nothing but reads a request
// body, with 405 and an Allow header, so a stray GET gets a clear answer rather than a decode
// error. It returns whether the request may proceed.
func postOnly(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "this endpoint requires POST", http.StatusMethodNotAllowed)
		return false
	}
	return true
}

func writeRemoteJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
