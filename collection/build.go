package collection

import (
	"errors"
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
	"github.com/tamnd/tsumugi/mph"
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
	return build(opts, 0, 0, false)
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
	return build(opts, base, idx, true)
}

func build(opts Options, baseStart uint32, indexStart int, recrawl bool) (Result, error) {
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
	// Cross-crawl dedup: an add brings a later crawl into an existing collection, so a
	// page the collection already holds and this crawl re-fetched must not build a
	// second copy. The intra-crawl dedupByIdentity in readSource folds duplicates
	// within one crawl; this folds them across crawls, keying on the same canonical URL
	// identity through the persisted membership directory. A collection built before the
	// directory existed has none, so the dedup is skipped (the documents all build, the
	// pre-directory behavior). The existing copy stays untouched because the shards are
	// immutable; the re-fetch is the one dropped.
	if recrawl {
		rd, err := LoadRecrawlDir(opts.Out)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return Result{}, fmt.Errorf("load recrawl directory: %w", err)
		}
		if rd != nil {
			docs, _ = dropRecrawled(docs, rd)
			hosts = countHosts(docs)
			// Every document the crawl carried was a re-fetch of a page the collection
			// already holds, so there is nothing new to add. Leave the collection as it
			// is rather than write empty shards or rebuild the index over no change.
			if len(docs) == 0 {
				return Result{Docs: 0}, nil
			}
		}
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
	sig, graphRegion, dir := globalSignals(docs, opts.TrustSeeds, opts.SpamSeeds)

	// Assign every document its corpus-stable global node id by the host and domain
	// partition, the id the graph region keys a far out-edge by. It is a separate id
	// from the dense docID a shard numbers its documents 0..N with: the dense id is
	// the within-shard position the postings and forward store use, the global id is
	// the host-clustered name a cross-shard edge points at, so a page's far targets on
	// one host fall in a contiguous id range the cross-shard list gap-encodes cheaply.
	// The shards carry it as their graph id table; serving keeps using the contiguous
	// nodeBase+dense docID handle, which is untouched.
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())

	res := Result{Docs: len(docs), Hosts: hosts}
	base := baseStart
	index := indexStart
	for lo := 0; lo < len(docs); lo += opts.ShardSize {
		hi := lo + opts.ShardSize
		if hi > len(docs) {
			hi = len(docs)
		}
		path := shardPath(opts.Out, index)
		n, err := writeShard(path, docs[lo:hi], sig.slice(lo, hi), base, lo, gids, dir)
		if err != nil {
			return Result{}, err
		}
		res.Bytes += n
		base += uint32(hi - lo)
		index++
		res.Shards++
	}
	// Persist the collection-wide link graph as its own artifact, the cross-shard
	// graph the out-of-core StreamPageRank streams from at scale without buffering
	// the adjacency. A fresh build's node space is the whole collection from global
	// id zero (baseStart and indexStart are zero), so the region's dense [0, N) ids
	// are the collection's global ids directly; an add extends an existing collection
	// and its union graph is a later milestone's concern, so it leaves the artifact
	// the original build wrote in place.
	if baseStart == 0 && indexStart == 0 && len(graphRegion) > 0 {
		if err := writeCollectionGraph(opts.Out, graphRegion, len(docs)); err != nil {
			return Result{}, fmt.Errorf("write collection graph: %w", err)
		}
	}
	// Refresh the collection artifact so serve reads the manifest, the fleet-wide
	// statistics, and the routing index from one file instead of rescanning every
	// shard. The index covers the whole directory, so an add reindexes the union of
	// the old and new shards, not just the slice this call wrote.
	if err := WriteIndex(opts.Out, uint64(time.Now().Unix())); err != nil {
		return Result{}, fmt.Errorf("write index: %w", err)
	}
	// Refresh the re-crawl membership directory so the next crawl can tell a page the
	// collection already holds from a new one in a hash probe rather than a full shard
	// scan. It is rebuilt over every shard in the directory, so an add's directory
	// covers the union of the old and new shards, the same whole-directory coverage the
	// index gets.
	if err := WriteRecrawlDir(opts.Out); err != nil {
		return Result{}, fmt.Errorf("write recrawl directory: %w", err)
	}
	return res, nil
}

// readSource reads every document from a crawl export, skipping records with no body
// since they carry no text to index, collapsing records that share a canonical URL to
// the most recent fetch, and counts the distinct hosts of what survives. It buffers
// the whole crawl so the build can order it; a crawl too large to buffer is the
// streaming case left for later.
func readSource(path string, limit int) ([]convert.Document, int, error) {
	src, err := convert.OpenSource(path)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = src.Close() }()

	var raw []convert.Document
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
		raw = append(raw, d)
		if limit > 0 && len(raw) >= limit {
			break
		}
	}
	docs, _ := dedupByIdentity(raw)
	hosts := map[string]struct{}{}
	for _, d := range docs {
		hosts[d.Host] = struct{}{}
	}
	return docs, len(hosts), nil
}

// dedupByIdentity collapses documents that name the same page to one, keying on doc
// 02's canonical URL identity: two URL spellings of a page (a trailing slash, a
// tracking parameter, a default port) fold to one canonical URL and so to one
// document, which is what keeps a page from being indexed, ranked, and counted twice.
// When a canonical URL appears more than once, the most recent fetch wins (the crawl
// dates are YYYY-MM-DD, so a lexical comparison is chronological), and a tie keeps the
// one already held so the result is deterministic. The surviving documents stay in
// first-seen order, so a build over the same crawl is reproducible. A record whose URL
// has no canonical form has no identity to collide on and is kept as-is. The second
// return is the number of duplicates dropped, for the caller to report or test.
func dedupByIdentity(docs []convert.Document) ([]convert.Document, int) {
	at := make(map[string]int, len(docs)) // canonical URL -> index into out
	out := make([]convert.Document, 0, len(docs))
	dropped := 0
	for _, d := range docs {
		cu, ok := analyze.CanonicalURL(d.URL)
		if !ok {
			out = append(out, d)
			continue
		}
		i, seen := at[cu]
		if !seen {
			at[cu] = len(out)
			out = append(out, d)
			continue
		}
		dropped++
		if d.CrawlDate > out[i].CrawlDate {
			out[i] = d
		}
	}
	return out, dropped
}

// writeShard builds the lexical, feature, forward, and graph regions for one slice
// of documents and writes them into a single shard file at the given global base. It
// returns the file size. The lexical index gets the title, body, and url fields; the
// feature matrix gets the derived content and url signals plus the collection-wide
// link signals in sig (one entry per document, aligned to docs); the forward store
// keeps the url, title, and body so the shard holds the text it was built from; the
// graph region carries the link graph recovered from the page bodies.
func writeShard(path string, docs []convert.Document, sig graphSignals, base uint32, lo int, gids []uint64, dir *mph.Dir) (int64, error) {
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
	// The shard's graph region carries the partition global node ids for its nodes as
	// its id table (dense docID d in this shard is collection index lo+d, whose global
	// id is gids[lo+d]), the names a cross-shard edge resolves against.
	gb := graph.NewBuilder(len(docs)).WithNodeIDs(gids[lo : lo+len(docs)])

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
		// The detected-language id supersedes the latin-script ratio stand-in the analyze
		// stage wrote into FeatLanguage; it is the real language the identifier read over
		// the body, computed once over the whole collection alongside the consistency signal.
		fb.Set(id, feature.FeatLanguage, float64(sig.langID[i]))
		// The composite static rank supersedes the per-document prior the analyze
		// stage wrote into FeatStaticRank above; it is the blend over the whole
		// collection's signals that orders the postings.
		fb.Set(id, feature.FeatStaticRank, sig.staticRank[i])
		// doc_id is the portable cross-crawl identity, the sha256 of the canonical URL.
		// A row with no usable canonical URL stores the zero id rather than the hash of
		// an empty string, the same absence the graph build reads from a non-resolving
		// link; every ingested document has a host, so this is the rare malformed case.
		if did, ok := analyze.DocID(d.URL); ok {
			fwd.Set(id, "doc_id", did[:])
		}
		fwd.Set(id, "url", []byte(d.URL))
		fwd.Set(id, "title", []byte(a.Title))
		fwd.Set(id, "body", []byte(d.Body))
		// Resolve each outbound link against the collection-wide directory. A target
		// the collection holds is either in this shard (an intra-shard edge, keyed by
		// the local dense docID) or in another shard (a cross-shard edge, keyed by the
		// target's global node id, framed into the region's cross-shard list to route to
		// the owning shard later). A target the crawl never captured does not resolve
		// and is dropped, which on a breadth-first sample is almost all of them.
		for _, tgt := range analyze.Links(d) {
			j, ok := dir.Lookup([]byte(tgt))
			if !ok || int(j) == lo+i {
				continue
			}
			if int(j) >= lo && int(j) < lo+len(docs) {
				gb.AddEdge(i, int(j)-lo)
			} else {
				gb.AddCrossEdge(i, gids[j])
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
