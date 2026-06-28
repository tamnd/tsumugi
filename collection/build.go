package collection

import (
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/lexical"
)

// DefaultShardSize is the document count a shard holds by default, the granularity
// the collection shards a crawl at.
const DefaultShardSize = 50000

// forwardColumn is a forward-store column declaration, kept local so the collection
// package names its document schema without leaking the storage type outward.
type forwardColumn struct {
	Name string
	Blob bool
}

// Options controls a build or an add.
type Options struct {
	Source    string // crawl export to read
	Out       string // collection directory to write into
	ShardSize int    // documents per shard
	Limit     int    // cap on documents read, zero for all

	// TrustSeeds and SpamSeeds are the curated TrustRank and Anti-TrustRank seed
	// URLs, the spine the spec's doc 07 extends with inverse-PageRank candidates.
	// They are recorded with the build so a rank is reproducible; an empty list
	// seeds trust purely from inverse PageRank and leaves anti-trust uniform.
	TrustSeeds []string
	SpamSeeds  []string
}

// Result reports what a build or add produced.
type Result struct {
	Docs   int
	Shards int
	Hosts  int
	Bytes  int64
}

// Build turns a crawl export into a fresh collection of shards. It reads the source,
// orders the documents by host then url so a host's pages land in the same shard and
// near each other, which is the locality the compression and the cache rely on, then
// cuts the ordered stream into shards and writes each one. The build assigns the dense
// global document ids in that same order, the id space every later stage keys off.
func Build(opts Options) (Result, error) {
	return build(opts, 0, 0)
}

// Add brings a later crawl into an existing collection without touching its shards.
// It continues the global id space past the highest existing id and names its shard
// files after the existing ones, so the new shards extend the collection rather than
// rewrite it, which is the freshness path immutability makes safe.
func Add(opts Options) (Result, error) {
	base, err := nextBase(opts.Out)
	if err != nil {
		return Result{}, err
	}
	idx, err := nextIndex(opts.Out)
	if err != nil {
		return Result{}, err
	}
	return build(opts, base, idx)
}

func build(opts Options, baseStart uint32, indexStart int) (Result, error) {
	if opts.ShardSize <= 0 {
		opts.ShardSize = DefaultShardSize
	}
	if err := os.MkdirAll(opts.Out, 0o755); err != nil {
		return Result{}, err
	}
	docs, hosts, err := readSource(opts.Source, opts.Limit)
	if err != nil {
		return Result{}, err
	}
	if len(docs) == 0 {
		return Result{}, fmt.Errorf("collection: source %s yielded no documents", opts.Source)
	}
	// Order by host then url: a host's pages share a shard and sit adjacent, the
	// locality the delta and dictionary compression exploit. This is the host-
	// grouping first pass of the doc 06 node ordering.
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Host != docs[j].Host {
			return docs[i].Host < docs[j].Host
		}
		return docs[i].URL < docs[j].URL
	})

	// Second pass: Recursive Graph Bisection refines the host-grouped order over
	// the resolved link graph, so the dense docID a document gets is its position
	// in the order that compresses the graph best. Permute the documents by it
	// before cutting shards; the signals computed next are indexed by the permuted
	// position, so they line up with the ids the shards are written under.
	order := collectionOrder(docs)
	reordered := make([]convert.Document, len(docs))
	for newID, oldID := range order {
		reordered[newID] = docs[oldID]
	}
	docs = reordered

	// Compute the link signals over the whole collection before cutting shards.
	// The web graph is almost entirely cross-shard, so this is the only place a
	// real signal exists; each shard then receives its slice of every signal vector
	// to bake into its feature matrix. The signals are indexed by the same host+url
	// order the shards are cut from, so sig.slice(lo, hi) lines up with docs[lo:hi].
	sig := globalSignals(docs, opts.TrustSeeds, opts.SpamSeeds)

	res := Result{Docs: len(docs), Hosts: hosts}
	base := baseStart
	index := indexStart
	for lo := 0; lo < len(docs); lo += opts.ShardSize {
		hi := lo + opts.ShardSize
		if hi > len(docs) {
			hi = len(docs)
		}
		path := shardPath(opts.Out, index)
		n, err := writeShard(path, docs[lo:hi], sig.slice(lo, hi), base)
		if err != nil {
			return Result{}, err
		}
		res.Bytes += n
		base += uint32(hi - lo)
		index++
		res.Shards++
	}
	// Refresh the collection artifact so serve reads the manifest, the fleet-wide
	// statistics, and the routing index from one file instead of rescanning every
	// shard. The index covers the whole directory, so an add reindexes the union of
	// the old and new shards, not just the slice this call wrote.
	if err := WriteIndex(opts.Out, uint64(time.Now().Unix())); err != nil {
		return Result{}, fmt.Errorf("write index: %w", err)
	}
	return res, nil
}

// readSource reads every document from a crawl export, skipping records with no body
// since they carry no text to index, and counts the distinct hosts. It buffers the
// whole crawl so the build can order it; a crawl too large to buffer is the streaming
// case left for later.
func readSource(path string, limit int) ([]convert.Document, int, error) {
	src, err := convert.OpenSource(path)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = src.Close() }()

	var docs []convert.Document
	hosts := map[string]struct{}{}
	for {
		d, ok, err := src.Next()
		if err != nil {
			return nil, 0, err
		}
		if !ok {
			break
		}
		if d.Body == "" {
			continue
		}
		docs = append(docs, d)
		hosts[d.Host] = struct{}{}
		if limit > 0 && len(docs) >= limit {
			break
		}
	}
	return docs, len(hosts), nil
}

// writeShard builds the lexical, feature, forward, and graph regions for one slice
// of documents and writes them into a single shard file at the given global base. It
// returns the file size. The lexical index gets the title, body, and url fields; the
// feature matrix gets the derived content and url signals plus the collection-wide
// link signals in sig (one entry per document, aligned to docs); the forward store
// keeps the url, title, and body so the shard holds the text it was built from; the
// graph region carries the link graph recovered from the page bodies.
func writeShard(path string, docs []convert.Document, sig graphSignals, base uint32) (int64, error) {
	lb := lexical.NewBuilder(lexical.DefaultParams())
	fb := feature.NewBuilder(feature.DefaultSchema(), feature.SchemaVersion)
	cols := docColumns()
	fwdCols := make([]forward.Column, len(cols))
	for i, c := range cols {
		fwdCols[i] = forward.Column{Name: c.Name, Type: forward.ColString}
		if c.Blob {
			fwdCols[i].Flags = forward.FlagBlob
		}
	}
	fwd := forward.NewBuilder(fwdCols)
	gb := graph.NewBuilder(len(docs))

	// Map each document's canonical URL to its local node id so an extracted link
	// whose target lives in this shard resolves to an intra-shard edge. A target
	// that resolves outside the shard is dropped here; the cross-shard edges are the
	// next milestone's concern, where a collection-wide node id carries them.
	urlToID := make(map[string]int, len(docs))
	for i, d := range docs {
		if cu, ok := analyze.CanonicalURL(d.URL); ok {
			urlToID[cu] = i
		}
	}

	var tokens, titleTokens, bodyTokens, urlTokens float64
	for i, d := range docs {
		a := analyze.Document(d)
		id := uint32(i)
		lb.AddDoc(id, map[lexical.Field]string{
			lexical.FieldTitle: a.Title,
			lexical.FieldBody:  d.Body,
			lexical.FieldURL:   d.URL,
		})
		// Per-field token counts feed the fleet average field lengths the broker BM25F
		// normalizes each field by. token_count stays title+body so avg_doc_len is
		// unchanged; the per-field sums are recorded alongside it.
		bt := len(lexical.Analyze(d.Body))
		tt := len(lexical.Analyze(a.Title))
		ut := len(lexical.Analyze(d.URL))
		bodyTokens += float64(bt)
		titleTokens += float64(tt)
		urlTokens += float64(ut)
		tokens += float64(bt + tt)
		for fid, v := range a.Features {
			fb.Set(id, fid, v)
		}
		// The collection-wide link signals, the columns that need the whole graph;
		// the per-document analyze stage leaves them at zero.
		fb.Set(id, feature.FeatPageRank, sig.pageRank[i])
		fb.Set(id, feature.FeatHostRank, sig.hostRank[i])
		fb.Set(id, feature.FeatDomainRank, sig.domainRank[i])
		fb.Set(id, feature.FeatTrust, sig.trust[i])
		fb.Set(id, feature.FeatSpamMass, sig.spamMass[i])
		fb.Set(id, feature.FeatInDegree, float64(sig.inDegree[i]))
		fb.Set(id, feature.FeatLinkingDomains, float64(sig.linkingDomains[i]))
		fb.Set(id, feature.FeatLinkingHosts, float64(sig.linkingHosts[i]))
		fb.Set(id, feature.FeatReciprocity, sig.reciprocity[i])
		fb.Set(id, feature.FeatHostLinkDiv, sig.hostLinkDiv[i])
		fb.Set(id, feature.FeatNearDup, sig.nearDup[i])
		fb.Set(id, feature.FeatOutboundSpam, sig.outboundSpam[i])
		// The composite static rank supersedes the per-document prior the analyze
		// stage wrote into FeatStaticRank above; it is the blend over the whole
		// collection's signals that orders the postings.
		fb.Set(id, feature.FeatStaticRank, sig.staticRank[i])
		fwd.Set(id, "url", []byte(d.URL))
		fwd.Set(id, "title", []byte(a.Title))
		fwd.Set(id, "body", []byte(d.Body))
		for _, tgt := range analyze.Links(d) {
			if j, ok := urlToID[tgt]; ok {
				gb.AddEdge(i, j)
			}
		}
	}

	gregion := gb.Build()
	g, err := graph.Open(gregion)
	if err != nil {
		return 0, err
	}

	w, err := tsumugi.Create(path)
	if err != nil {
		return 0, err
	}
	w.SetDocCount(uint32(len(docs)))
	w.SetNodeBase(uint64(base))
	w.SetStat(tsumugi.StatTokenCount, tokens)
	w.SetStat(tsumugi.StatTitleTokenCount, titleTokens)
	w.SetStat(tsumugi.StatBodyTokenCount, bodyTokens)
	w.SetStat(tsumugi.StatURLTokenCount, urlTokens)
	w.SetStat(tsumugi.StatEdgeCount, float64(g.EdgeCount()))
	// Record the analyzer the build tokenized with so a broker can verify in one
	// comparison that it is about to query the shard with the same analyzer. The build
	// runs the package-level lexical.Analyze, so the recorded hash is DefaultAnalyzer's.
	w.SetAnalyzerHash(lexical.DefaultAnalyzer.Hash())
	if err := w.AddRegion(tsumugi.RegionLexical, tsumugi.CodecZstd, 0, 0, lb.Build()); err != nil {
		return 0, err
	}
	if err := w.AddRegion(tsumugi.RegionFeature, tsumugi.CodecZstd, 0, 0, fb.Build()); err != nil {
		return 0, err
	}
	if err := w.AddRegion(tsumugi.RegionForward, tsumugi.CodecZstd, 0, 0, fwd.Build()); err != nil {
		return 0, err
	}
	if err := w.AddRegion(tsumugi.RegionGraph, tsumugi.CodecZstd, 0, 0, gregion); err != nil {
		return 0, err
	}
	if err := w.Close(); err != nil {
		return 0, err
	}
	st, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return st.Size(), nil
}
