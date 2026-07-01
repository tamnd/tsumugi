package collection

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"os"
	"sort"

	"github.com/tamnd/tsumugi"
	"github.com/tamnd/tsumugi/analyze"
	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/dense"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/forward"
	"github.com/tamnd/tsumugi/graph"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/vector"
)

// shardMeta carries the build-level values every shard in a collection stamps
// identically: the build epoch and the configuration hash. They are computed once
// for the whole build and threaded into each shard so the shards a single build
// writes agree on when, and under what configuration, they were produced.
type shardMeta struct {
	epoch      uint64
	configHash uint64
	impact     bool // build the lexical region impact-ordered from the static rank
	denseDim   int  // kept dimension of the dense vector region, zero to emit none
}

// buildConfigHash digests the build configuration into the 64-bit value every shard
// records as its build_config_hash. It folds in each input that decides how a shard's
// bytes are produced: the container format version, the feature schema version, the
// routing index version, the analyzer hash, the shard size, and the curated trust and
// spam seeds in sorted order so the digest does not depend on the order the seeds were
// passed. A change to any of these changes the digest, so two builds that share it
// produced configuration-identical shards, the property a reproducibility check rests
// on. It deliberately leaves out the build epoch and the corpus: the epoch is recorded
// separately in the header, and the digest is meant to identify the configuration, not
// the particular crawl, so the same configuration over a different corpus keeps it.
func buildConfigHash(shardSize int, trustSeeds, spamSeeds []string, impact bool, denseDim int) uint64 {
	h := fnv.New64a()
	var u [8]byte
	put := func(v uint64) {
		binary.LittleEndian.PutUint64(u[:], v)
		_, _ = h.Write(u[:])
	}
	put(uint64(tsumugi.VersionMajor))
	put(uint64(tsumugi.VersionMinor))
	put(uint64(feature.SchemaVersion))
	put(indexVersion)
	put(lexical.DefaultAnalyzer.Hash())
	put(uint64(shardSize))
	// The posting ordering is a configuration input: an impact-ordered build produces
	// different posting bodies than a docID-ordered one, so the digest must separate them
	// or a reproducibility check would call two differently-ordered collections identical.
	if impact {
		put(1)
	} else {
		put(0)
	}
	// The dense dimension is a configuration input: a shard built with a vector region
	// carries different bytes than one built without, and two collections at different
	// dimensions embed into different spaces, so the digest must separate them or a
	// reproducibility check would call two differently-embedded collections identical.
	put(uint64(denseDim))
	putSeeds := func(seeds []string) {
		s := append([]string(nil), seeds...)
		sort.Strings(s)
		put(uint64(len(s)))
		for _, x := range s {
			put(uint64(len(x)))
			_, _ = h.Write([]byte(x))
		}
	}
	putSeeds(trustSeeds)
	putSeeds(spamSeeds)
	return h.Sum64()
}

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

	// BuildEpoch is the build timestamp in seconds stamped into every shard header
	// and the collection index, the one input a build reads from a clock. The library
	// never reads a clock itself: a caller that wants a reproducible build passes a
	// fixed epoch and gets byte-identical shards and index, while the CLI defaults it
	// to the current time so an operational build still records when it ran. Zero is a
	// valid pinned epoch, the deterministic default for a library caller or a test.
	BuildEpoch uint64

	// Impact builds the lexical region impact-ordered rather than docID-ordered: each
	// shard's posting lists are sorted by the composite static rank quantized to a byte,
	// and served by the early-termination traversal that scores query-term coverage
	// weighted by that rank (spec doc 04's second ordering). It is the inference-free
	// retrieval mode: no learned per-term weights, the static rank alone orders and scores.
	// The dictionary and every other region are unchanged, so routing and the feature and
	// forward planes are identical; only the posting bodies and their order differ.
	Impact bool

	// DenseDim is the kept dimension of the per-shard dense vector region the build emits,
	// the retrieval plane doc 08 pins alongside the lexical one. Zero leaves it off: no
	// vector region is written and the serving cascade's dense plane stays inert, the
	// behavior before this was wired. A positive value embeds every document's analyzed
	// body with the package default static encoder (dense.NewDefault), the same encoder the
	// query pipeline builds its query vector with, so a document and a query at this
	// dimension live in one comparable space, and writes the quantized ANN region the shard
	// serves dense recall from. The dimension is recorded in each shard's footer as
	// vector_dim and read back by the broker, which turns the dense plane on the moment a
	// shard carries a region and every shard agrees on the dimension.
	DenseDim int
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

	// Resolve the collection-wide canonical-URL directory and assign every document its
	// corpus-stable global node id before any signal is computed. The directory resolves
	// each outbound link to a collection index, and the global id is the host-clustered
	// name a cross-shard edge points at; both feed the per-shard graph regions built next.
	// The global id is separate from the dense docID a shard numbers its documents 0..N
	// with: the dense id is the within-shard position the postings and forward store use,
	// the global id is the contiguous per-host range a cross-shard list gap-encodes
	// cheaply. The shards carry it as their graph id table; serving keeps using the
	// contiguous nodeBase+dense docID handle, which is untouched.
	dir := buildDir(docs)
	gids := AssignGlobalIDs(docs, DefaultPartitionParams())

	// Invert the outbound anchor text into per-document inbound anchor fields now that
	// the directory can resolve every link to a node id. A document's anchor field is
	// the phrases other pages used to link it, weighted by distinct source domain and by
	// off-domain endorsement, the off-page describes-me signal the shards index as
	// FieldAnchor. It is gathered over the whole collection here, in the same node-id
	// order the shards slice their documents by, so each writeShard reads its own slice.
	anchors := anchorFields(docs, dir)

	// Build every shard's graph region first, in shard order, then compute the link
	// signals off those persisted per-shard graphs. The M15 reorder: the web graph is
	// almost entirely cross-shard, so a real link signal only exists across the whole
	// collection, but rather than rank over a second merged in-core graph the build now
	// joins the per-shard regions with the cross-shard rank loops, the shardable form
	// that holds no expanded adjacency resident. Each shard then receives its slice of
	// every signal vector to bake into its feature matrix. The signals are indexed by the
	// same host+url order the shards are cut from, so sig.slice(lo, hi) lines up with
	// docs[lo:hi] and with the shard whose graph carries those documents.
	layouts, regions, err := buildShardGraphs(docs, gids, dir, opts.ShardSize)
	if err != nil {
		return Result{}, fmt.Errorf("build shard graphs: %w", err)
	}
	sig := shardedSignals(regions, docs, gids, opts.TrustSeeds, opts.SpamSeeds, dir, DefaultPartitionParams())

	// The epoch and the configuration digest are build-level: every shard stamps the
	// same pair, so the shards a single build writes agree on when and under what
	// configuration they were produced. Computing them once here keeps writeShard from
	// re-deriving the digest per shard and guarantees the shards cannot disagree.
	meta := shardMeta{
		epoch:      opts.BuildEpoch,
		configHash: buildConfigHash(opts.ShardSize, opts.TrustSeeds, opts.SpamSeeds, opts.Impact, opts.DenseDim),
		impact:     opts.Impact,
		denseDim:   opts.DenseDim,
	}

	res := Result{Docs: len(docs), Hosts: hosts}
	base := baseStart
	index := indexStart
	for _, sl := range layouts {
		path := shardPath(opts.Out, index)
		n, err := writeShard(path, docs[sl.lo:sl.hi], anchors[sl.lo:sl.hi], sig.slice(sl.lo, sl.hi), base, sl.gregion, meta)
		if err != nil {
			return Result{}, err
		}
		res.Bytes += n
		base += uint32(sl.hi - sl.lo)
		index++
		res.Shards++
	}
	// Persist the collection-wide link graph as its own artifact, the cross-shard
	// graph the out-of-core StreamPageRank streams from at scale without buffering
	// the adjacency. It is assembled out of core from the documents, the same union
	// the per-shard regions carry sharded, encoded over the dense [0, N) global id
	// space the artifact spans. A fresh build's node space is the whole collection from
	// global id zero (baseStart and indexStart are zero), so the region's dense [0, N)
	// ids are the collection's global ids directly; an add extends an existing collection
	// and its union graph is a later milestone's concern, so it leaves the artifact the
	// original build wrote in place.
	if baseStart == 0 && indexStart == 0 {
		if graphRegion := buildGraphRegionBytes(docs, dir); len(graphRegion) > 0 {
			if err := writeCollectionGraph(opts.Out, graphRegion, len(docs)); err != nil {
				return Result{}, fmt.Errorf("write collection graph: %w", err)
			}
		}
	}
	// Refresh the collection artifact so serve reads the manifest, the fleet-wide
	// statistics, and the routing index from one file instead of rescanning every
	// shard. The index covers the whole directory, so an add reindexes the union of
	// the old and new shards, not just the slice this call wrote.
	if err := WriteIndex(opts.Out, opts.BuildEpoch); err != nil {
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

// writeShard builds the lexical, feature, and forward regions for one slice of
// documents and writes them, with the prebuilt graph region, into a single shard file
// at the given global base. It returns the file size. The lexical index gets the title,
// body, url, and inbound-anchor fields (anchors[i] is the anchor field the inversion
// gathered for the shard-local document i); the feature matrix gets the derived content
// and url signals plus
// the collection-wide link signals in sig (one entry per document, aligned to docs); the
// forward store keeps the url, title, and body so the shard holds the text it was built
// from. The graph region is passed in: the M15 reorder builds every shard's graph region
// first so the signals can be computed off the persisted shard graphs, and writeShard
// embeds the very bytes those signals were read from, so the stored graph and the graph
// the signals saw are one.
func writeShard(path string, docs []convert.Document, anchors []string, sig graphSignals, base uint32, gregion []byte, meta shardMeta) (int64, error) {
	lb := lexical.NewBuilder(lexical.DefaultParams())
	fb := feature.NewBuilder(feature.DefaultSchema(), feature.SchemaVersion)
	cols := docColumns()
	fwdCols := make([]forward.Column, len(cols))
	for i, c := range cols {
		fwdCols[i] = forward.Column{Name: c.Name, Type: forward.ColString, Codec: forward.CodecZstdDict}
		if c.Blob {
			// A blob column, the body, is read by a leading window in the L2 scan, so it
			// is stored block-structured: a windowed read decodes only the blocks the scan
			// reaches instead of the whole body, which bounds the L2 body-decompression
			// tail that sets the serving p99 on a large-body corpus.
			fwdCols[i].Flags = forward.FlagBlob
			fwdCols[i].Codec = forward.CodecZstdDictBlocked
		}
		// doc_id is a 32-byte sha256, effectively random and incompressible, and the
		// recrawl and compact paths scan it as raw bytes; storing it uncompressed
		// keeps that scan zero-copy and wastes no frame overhead on noise. The text
		// columns share a derived dictionary so many small similar values compress
		// against one context, and the region is stored uncompressed in the container
		// so opening a shard inflates no bodies, only the values a query reaches do.
		if c.Name == "doc_id" {
			fwdCols[i].Codec = forward.CodecNone
		}
	}
	fwd := forward.NewBuilder(fwdCols)

	// The dense retrieval plane: when the build is configured with a kept dimension, embed
	// every document with the package default static encoder and collect the vectors into a
	// vector-region builder. The encoder is the same dense.NewDefault the query pipeline
	// builds its query vector with, so a document and a query at this dimension land in one
	// comparable space. Add is called once per document below, in dense docID order, so the
	// vector region's node id for a document is the shard-local docID the lexical and
	// forward planes use, and the cascade can fuse a dense hit with a lexical one by id.
	var enc *dense.StaticEncoder
	var vb *vector.Builder
	if meta.denseDim > 0 {
		enc = dense.NewDefault(meta.denseDim)
		vb = vector.NewBuilder(meta.denseDim)
	}

	var tokens, titleTokens, bodyTokens, urlTokens, anchorTokens float64
	for i, d := range docs {
		a := analyze.Document(d)
		id := uint32(i)
		lb.AddDoc(id, map[lexical.Field]string{
			lexical.FieldTitle:  a.Title,
			lexical.FieldBody:   d.Body,
			lexical.FieldURL:    d.URL,
			lexical.FieldAnchor: anchors[i],
		})
		// Per-field token counts feed the fleet average field lengths the broker BM25F
		// normalizes each field by. token_count stays title+body so avg_doc_len is
		// unchanged; the per-field sums are recorded alongside it.
		bodyTerms := lexical.Analyze(d.Body)
		bt := len(bodyTerms)
		tt := len(lexical.Analyze(a.Title))
		ut := len(lexical.Analyze(d.URL))
		at := len(lexical.Analyze(anchors[i]))
		bodyTokens += float64(bt)
		titleTokens += float64(tt)
		urlTokens += float64(ut)
		anchorTokens += float64(at)
		tokens += float64(bt + tt)
		for fid, v := range a.Features {
			fb.Set(id, fid, v)
		}
		// The anchor field length is a collection-wide quantity: it is known only after the
		// inbound link text is inverted by target, so the per-document analyze stage cannot
		// set it the way it sets the title, body, and url field lengths. Set it here, once the
		// inversion has assembled each document's anchor field, so the query-independent matrix
		// carries the anchor-field-length column the ranking model reads alongside the other
		// three field lengths.
		fb.Set(id, feature.FeatAnchorFieldLen, float64(at))
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
		fwd.Set(id, "anchor", []byte(anchors[i]))
		// Embed the document into the dense plane, when it is on, from the same analyzed
		// body tokens the lexical body field indexes, pooled and L2-normalized by the static
		// encoder. Every document is added, in docID order, so the vector region's node ids
		// line up one-for-one with the shard's docIDs. A body whose terms are all unknown to
		// the table pools to the zero vector, which the region reads as no dense signal (a
		// cosine of zero against any query), so the document stays lexically reachable while
		// contributing no spurious dense neighbor. Encode returns exactly denseDim floats, so
		// the Add width is uniform across the shard.
		if vb != nil {
			vb.Add(enc.Encode(bodyTerms))
		}
	}

	// Open the prebuilt graph region to read its edge count for the footer; the bytes are
	// stored as-is below. graph.Open holds no adjacency resident, so this is a header parse.
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
	// Stamp the build-level metadata: the epoch into the header (passed in, never read
	// from a clock, so a build with a pinned epoch is byte-identical) and the
	// configuration digest into the footer so a reader can check two shards were built
	// the same way without reopening every region.
	w.SetBuildEpoch(meta.epoch)
	w.SetStat(tsumugi.StatBuildConfigHash, tsumugi.AnalyzerHashStat(meta.configHash))
	w.SetStat(tsumugi.StatTokenCount, tokens)
	w.SetStat(tsumugi.StatTitleTokenCount, titleTokens)
	w.SetStat(tsumugi.StatBodyTokenCount, bodyTokens)
	w.SetStat(tsumugi.StatURLTokenCount, urlTokens)
	w.SetStat(tsumugi.StatAnchorTokenCount, anchorTokens)
	w.SetStat(tsumugi.StatEdgeCount, float64(g.EdgeCount()))
	// The remaining shard-level numbers the footer promises but the build left empty.
	// node_min and node_max bracket the shard's global id range so the graph tooling can
	// validate a cross-shard edge target without opening a region; avg_doc_len is the
	// whole-document mean (title+body) the plain BM25 normalizer reads, distinct from the
	// per-field averages above; term_count is the dictionary size for capacity planning.
	if len(docs) > 0 {
		w.SetStat(tsumugi.StatNodeMin, float64(base))
		w.SetStat(tsumugi.StatNodeMax, float64(base)+float64(len(docs))-1)
		w.SetStat(tsumugi.StatAvgDocLen, tokens/float64(len(docs)))
	}
	// Build the region bytes now so the feature dequant constants and the lexical term
	// count can be read back into the footer before the regions are written. An impact
	// build orders and scores the postings by the composite static rank quantized to a
	// byte; the docID-ordered build is the classic BM25F region. Both reuse the same
	// accumulated term set, so the dictionary, the bloom filter, and every other region
	// are identical and only the posting bodies differ.
	var lexBytes []byte
	if meta.impact {
		impacts := quantizeImpact(sig.staticRank)
		lexBytes = lb.BuildImpact(func(id uint32) uint8 { return impacts[id] })
	} else {
		lexBytes = lb.Build()
	}
	featBytes := fb.Build()
	if lr, err := lexical.Open(lexBytes); err == nil {
		w.SetStat(tsumugi.StatTermCount, float64(lr.TermCount()))
	}
	// Write the per-column feature dequant constants into the footer statistics, the
	// container-level dequant block doc 03 names; the feature region still carries its
	// own self-describing copy, and the two agree because both come from this one build.
	feature.WriteDequantStats(w.SetStat, fb.Dequant())
	// Record the analyzer the build tokenized with so a broker can verify in one
	// comparison that it is about to query the shard with the same analyzer. The build
	// runs the package-level lexical.Analyze, so the recorded hash is DefaultAnalyzer's.
	w.SetAnalyzerHash(lexical.DefaultAnalyzer.Hash())
	if err := w.AddRegion(tsumugi.RegionLexical, tsumugi.CodecZstd, 0, 0, lexBytes); err != nil {
		return 0, err
	}
	if err := w.AddRegion(tsumugi.RegionFeature, tsumugi.CodecZstd, 0, 0, featBytes); err != nil {
		return 0, err
	}
	// The forward region compresses per value internally now, so the container
	// stores it uncompressed: a shard open mmaps the region and inflates only the
	// values a query touches, instead of decompressing every body up front.
	if err := w.AddRegion(tsumugi.RegionForward, tsumugi.CodecNone, 0, 0, fwd.Build()); err != nil {
		return 0, err
	}
	if err := w.AddRegion(tsumugi.RegionGraph, tsumugi.CodecZstd, 0, 0, gregion); err != nil {
		return 0, err
	}
	// The dense vector region, when the build embedded documents. Its bytes are already a
	// packed quantized format (rotated one-bit codes plus the int8 rerank payload and the
	// HNSW graph), so like the forward region it is stored with no container codec: a shard
	// open mmaps it and the reader inflates nothing up front. The kept dimension goes into
	// the footer as vector_dim so the broker can read the dense-plane width without opening
	// the region, and refuse to serve a fleet whose shards disagree on it.
	if vb != nil {
		vecBytes, err := vb.Build()
		if err != nil {
			return 0, fmt.Errorf("build vector region: %w", err)
		}
		w.SetStat(tsumugi.StatVectorDim, float64(meta.denseDim))
		if err := w.AddRegion(tsumugi.RegionVector, tsumugi.CodecNone, 0, 0, vecBytes); err != nil {
			return 0, err
		}
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
