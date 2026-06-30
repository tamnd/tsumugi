package search

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
	"github.com/tamnd/tsumugi/rank"
)

// Broker fans a query across many shards, gathers their candidates, and runs one
// global rerank to produce a fleet-wide top-k. It owns the served shard set, the
// routing index that prunes the fan-out, the fleet-wide statistics, and the cascade
// that does the global rerank. A broker serves queries concurrently, and a publish or
// retire swaps the served shard set atomically while queries keep running (doc 11).
//
// The global rerank is where the broker's exactness lives. Each shard does only
// retrieval; the broker collects every candidate with its per-document feature row,
// merges the per-shard ranked lists into global ranked lists, and runs the cascade
// once over the union. Because a document's final score is the global L2 model over
// its own feature row, the score does not depend on which shard the document came
// from, so given a recall-complete candidate set the broker's top-k equals the top-k
// a single index over every shard would produce.
type Broker struct {
	// state is the served shard set and the routing index and statistics computed over
	// it, held behind one atomic pointer so a query reads a single consistent snapshot
	// and a publish or retire swaps the whole set in one store. A query loads it once at
	// entry and uses that snapshot to the end, so a concurrent swap never splits a query
	// across two shard sets.
	state atomic.Pointer[brokerState]

	cascade        *rank.Cascade
	maxConcurrency int

	// swapMu serializes shard-set swaps (publish, retire, close) and excludes the brief
	// load-and-reference a query takes at entry. A swap holds the write lock; a query holds
	// the read lock only long enough to load the current snapshot and take a reference on
	// it, never for the query itself. This is what makes the reclaim below safe without a
	// lock on the query path: a query that has referenced a snapshot can never be racing the
	// reclaim that frees one of that snapshot's shards, because the reference was taken under
	// the read lock and the reclaim runs under the write lock.
	swapMu sync.RWMutex

	// mapped counts, per open shard, the number of live snapshots that contain it. A swap
	// increments the count for the snapshot it installs, and a snapshot that has drained of
	// in-flight queries decrements the count for its shards; a shard whose count reaches zero
	// sits in no live snapshot, so no in-flight query can still reach it, and its file mapping
	// is unmapped then. This is how a retired shard's address space is reclaimed while the
	// broker runs rather than held until Close, the property a fleet churning shards toward the
	// 100k-shard target needs so retired mappings do not accumulate for the life of the process.
	// It is only ever touched under swapMu's write lock.
	mapped map[*Shard]int

	// cache is the optional result cache (doc 11). It is nil unless a deployment wires
	// one, so a broker without a cache runs every query through the cascade unchanged. A
	// publish or retire clears it, because a changed shard set can change a query's top-k.
	cache *ResultCache
}

// brokerState is the swappable serving snapshot: the shards a broker serves, the
// routing index over them, and the fleet-wide statistics. It is immutable once stored,
// so a query that loads it reads a consistent set even as a publish swaps in a new one.
type brokerState struct {
	shards  []*Shard
	routing *RoutingIndex
	stats   GlobalStats

	// queryRefs counts the references on this snapshot: one owner reference held while it is
	// the current served set, plus one per in-flight query that has loaded it. The owner
	// reference is dropped when a swap supersedes the snapshot; the count reaches zero only
	// after the snapshot is no longer current and every query that loaded it has finished, at
	// which point its retired shards are safe to unmap. It is set to one when the snapshot is
	// installed; a freshly constructed snapshot starts at zero until then.
	queryRefs atomic.Int64

	// baseN is the number of shards the routing index covers: shards[:baseN] are routed by
	// the base index and shards[baseN:] by the overlay, the shards published since the last
	// full rebuild (see incremental.go). A full build sets baseN to the whole set and leaves
	// the overlay empty; an incremental publish appends to the overlay and leaves baseN
	// where it was.
	baseN int

	// overlay holds the routing contributions of the shards published since the last full
	// rebuild, each a front-coded dictionary over one shard's own terms, so a publish adds a
	// shard for the cost of its own vocabulary rather than a rescan of the fleet.
	overlay []overlayEntry

	// sums are the raw fleet statistic accumulators stats is derived from, kept so a publish
	// can fold one shard in and re-divide rather than rescanning every shard for the
	// averages.
	sums statSums

	// shardStatic is the per-shard static-rank summary the shard-dropping degradation
	// rung orders shards by, computed once on first use under staticOnce so a state whose
	// queries never drop shards never pays the scan. It belongs to the snapshot because
	// it summarizes this snapshot's shards.
	staticOnce  sync.Once
	shardStatic []float64
}

// newState builds a serving snapshot over a shard set, scanning the shards for the
// routing index and the fleet-wide statistics. It is the scan path a publish and a
// retire rebuild through, fine because they are infrequent relative to queries.
func newState(shards []*Shard) *brokerState {
	sums := computeStatSums(shards)
	return &brokerState{
		shards:  shards,
		baseN:   len(shards),
		routing: BuildRoutingIndex(shards),
		sums:    sums,
		stats:   sums.global(),
	}
}

// NewBroker builds a broker over already-opened shards with a global cascade. It
// constructs the routing index and the fleet-wide statistics from the shards. The
// cascade carries the global L1 cut and L2 model the broker reranks the merged
// candidate set with. This is the scan path: it walks every shard's vocabulary to
// build the routing index, which is fine for a handful of shards but is what the
// persisted artifact replaces at fleet scale, through NewBrokerWith.
func NewBroker(shards []*Shard, cascade *rank.Cascade) *Broker {
	return NewBrokerWith(shards, cascade, BuildRoutingIndex(shards), computeGlobalStats(shards))
}

// NewBrokerWith builds a broker over already-opened shards with a routing index and
// fleet-wide statistics supplied from outside, the values a collection artifact loads.
// It is the scale path: serve reads the manifest, the statistics, and the routing index
// from one file rather than rescanning every shard, so startup is proportional to the
// query vocabulary, not the corpus. The routing index's shard ids must line up with the
// order of the shards slice.
func NewBrokerWith(shards []*Shard, cascade *rank.Cascade, routing *RoutingIndex, stats GlobalStats) *Broker {
	b := &Broker{
		cascade:        cascade,
		maxConcurrency: runtime.GOMAXPROCS(0),
		mapped:         make(map[*Shard]int, len(shards)),
	}
	st := &brokerState{
		shards:  shards,
		baseN:   len(shards),
		routing: routing,
		sums:    sumsFromStats(stats),
		stats:   stats,
	}
	st.queryRefs.Store(1)
	for _, s := range shards {
		b.mapped[s]++
	}
	b.state.Store(st)
	return b
}

// loadState loads the current serving snapshot without taking a reference, for the scalar
// reads (shard count, statistics) that return a value and never touch a shard's mapping, so
// a concurrent reclaim of a superseded snapshot cannot fault them. A path that reads shard
// data must go through acquire instead so the snapshot's shards stay mapped while it runs.
func (b *Broker) loadState() *brokerState { return b.state.Load() }

// acquire loads the current snapshot and takes a reference on it, so the snapshot's shards
// stay mapped for the whole query even if a concurrent retire supersedes the snapshot and
// the query outlives the swap. The read lock is held only for the load-and-reference, not
// for the query: it serializes against a swap's write lock just long enough that the
// reference is taken before the snapshot can be superseded, which is what keeps the matching
// release from racing the reclaim. Every acquire must be paired with a release.
func (b *Broker) acquire() *brokerState {
	b.swapMu.RLock()
	st := b.state.Load()
	st.queryRefs.Add(1)
	b.swapMu.RUnlock()
	return st
}

// release drops a query's reference on a snapshot. When the count reaches zero the snapshot
// is no longer current and no query still holds it, so its shards can be reclaimed; the
// common case, a query against the still-current snapshot, never reaches zero because the
// owner reference keeps the count above it, so release takes no lock at all. Only the last
// query on a superseded snapshot pays the write lock to run the reclaim.
func (b *Broker) release(st *brokerState) {
	if st.queryRefs.Add(-1) == 0 {
		b.swapMu.Lock()
		b.reclaim(st)
		b.swapMu.Unlock()
	}
}

// reclaim drops one snapshot's contribution to the per-shard mapping counts and unmaps every
// shard the snapshot was the last live holder of. A survivor shared with the current snapshot
// keeps a positive count and stays mapped; a shard retired before this snapshot drained falls
// to zero and is closed exactly once. The caller holds swapMu's write lock, so the counts and
// the close run without racing a swap or another reclaim.
func (b *Broker) reclaim(st *brokerState) {
	for _, s := range st.shards {
		c, ok := b.mapped[s]
		if !ok {
			// Already released, by an earlier reclaim or by Close clearing the table at
			// shutdown. A reference dropped after Close lands here and closes nothing twice.
			continue
		}
		if c <= 1 {
			delete(b.mapped, s)
			_ = s.Close()
		} else {
			b.mapped[s] = c - 1
		}
	}
}

// installState makes ns the current snapshot: it takes ns's owner reference, counts ns's
// shards into the mapping table, swaps the pointer, and drops the superseded snapshot's owner
// reference, reclaiming it immediately when no query still holds it. The caller holds swapMu's
// write lock, so the mapping-count updates and the pointer swap are atomic with respect to a
// query's acquire and to another swap.
func (b *Broker) installState(ns *brokerState) {
	ns.queryRefs.Store(1)
	for _, s := range ns.shards {
		b.mapped[s]++
	}
	old := b.state.Load()
	b.state.Store(ns)
	if old.queryRefs.Add(-1) == 0 {
		b.reclaim(old)
	}
}

// Publish swaps in a serving snapshot that includes shard, then clears the result cache
// (doc 11 publish lifecycle): a new shard can hold a document that belongs in a cached
// query's top-k, so a result computed against the old set can be stale against the new
// one, and the coarse correct thing is to drop the cache and let traffic re-warm it. The
// swap is atomic, so a query in flight finishes against the snapshot it loaded and the
// next query sees the new shard.
func (b *Broker) Publish(shard *Shard) {
	b.swapMu.Lock()
	b.installState(b.state.Load().withShard(shard))
	b.swapMu.Unlock()
	b.invalidateCache()
}

// Retire removes every served shard the predicate selects, swaps in a snapshot over the
// survivors, and clears the result cache (the mirror of Publish): a retired shard can
// remove a document from a cached query's top-k, so the cache is dropped on a retire the
// same way. It returns the number of shards retired. A retired shard's mapping is held
// until the last query that loaded a snapshot containing it finishes, then unmapped by the
// reclaim, so the retire never faults an in-flight reader yet does not leak the mapping for
// the life of the broker.
func (b *Broker) Retire(pred func(*Shard) bool) int {
	b.swapMu.Lock()
	cur := b.state.Load()
	keep := make([]*Shard, 0, len(cur.shards))
	removed := 0
	for _, s := range cur.shards {
		if pred(s) {
			removed++
		} else {
			keep = append(keep, s)
		}
	}
	if removed == 0 {
		b.swapMu.Unlock()
		return 0
	}
	b.installState(newState(keep))
	b.swapMu.Unlock()
	b.invalidateCache()
	return removed
}

// mappedShards is the number of distinct shard mappings the broker currently holds open,
// the current snapshot's shards plus any retired shards a still-draining snapshot keeps
// mapped. A test watches it fall as retired shards are reclaimed.
func (b *Broker) mappedShards() int {
	b.swapMu.RLock()
	defer b.swapMu.RUnlock()
	return len(b.mapped)
}

// invalidateCache clears the result cache if one is wired, the step a publish or retire
// runs so no top-k computed against the old shard set is served against the new one.
func (b *Broker) invalidateCache() {
	if b.cache != nil {
		b.cache.Clear()
	}
}

// CheckModel verifies the cascade's L2 model was trained against the feature schema
// this build scores against, the broker-construction half of the schema guard whose
// shard-open half lives in newShard. A stamped model whose schema does not match is
// refused; an unstamped model, hash zero, is allowed but unverified, the legacy or
// hand-built path that carries no guarantee. Because every shard self-verified its
// own feature region against the same canonical schema at open, a model that matches
// the canonical schema matches the whole fleet, so this one comparison is enough.
func (b *Broker) CheckModel() error {
	if b.cascade == nil || b.cascade.L2 == nil {
		return nil
	}
	m := b.cascade.L2
	if m.SchemaHash() == 0 {
		return nil
	}
	if m.SchemaVersion() != feature.SchemaVersion || m.SchemaHash() != feature.DefaultSchemaHash() {
		return fmt.Errorf("%w: model trained against schema v%d hash %016x, this build expects v%d hash %016x",
			ErrSchemaMismatch, m.SchemaVersion(), m.SchemaHash(), feature.SchemaVersion, feature.DefaultSchemaHash())
	}
	return nil
}

// Stats returns the fleet-wide collection statistics over the served shard set.
func (b *Broker) Stats() GlobalStats { return b.loadState().stats }

// NumShards is the number of shards the broker currently serves.
func (b *Broker) NumShards() int { return len(b.loadState().shards) }

// RoutingBytes is the resident size of the broker's routing structures, the base front-coded
// index plus the not-yet-folded overlay of per-shard dictionaries. It is the footprint a
// scale measurement charges to routing, so a continuous ingest can watch the overlay's cost
// grow between compactions and confirm a fold reclaims it, the property that keeps routing
// memory bounded as the served set grows toward the 100k-shard target.
func (b *Broker) RoutingBytes() int {
	st := b.loadState()
	return st.routing.sizeBytes() + st.overlaySize()
}

// Close releases every shard mapping the broker still holds open: the current snapshot's
// shards and any retired shard a still-draining snapshot kept mapped. During the broker's
// life the reclaim unmaps a retired shard as soon as the last query holding it finishes, so
// at a clean shutdown the mapping table is usually just the current set; Close is the
// backstop that releases whatever remains. The table holds each distinct mapping once, so a
// munmap runs exactly once per mapping. It assumes no query is in flight, the caller's
// contract, so it does not wait on references.
func (b *Broker) Close() error {
	b.swapMu.Lock()
	defer b.swapMu.Unlock()
	var first error
	for s := range b.mapped {
		if err := s.Close(); err != nil && first == nil {
			first = err
		}
	}
	b.mapped = make(map[*Shard]int)
	return first
}

// shardResult is one shard's contribution to a fan-out: its ranked lists already
// shifted into the global id space, the feature rows of its candidates, and the
// index of the shard that produced them so the global rerank can route a survivor's
// online feature extraction back to the shard that holds its text.
type shardResult struct {
	shard int
	lex   []scored
	dense []scored
	feats map[uint32][]float64
}

// Results is a fleet-wide top-k with the completeness the query reached. A query
// returns the best top-k it can with an honest completeness indicator rather than
// failing or overrunning, so a partial answer is a normal degraded result, not an
// error (doc 11, "Failure modes and partial results"). ShardsTotal is the number of
// contributing shards routing selected; ShardsOK is the number that responded inside
// the deadline. The engine never silently returns a partial result as if it were
// complete: Complete is true only when every contributing shard responded.
type Results struct {
	Hits        []Hit
	ShardsTotal int
	ShardsOK    int

	// Degraded is the rung of the degradation ladder the broker served this query at,
	// DegradeNone for a full-quality result. It is the second of the two independent
	// degradation facts a result carries: this one is the quality reduction the broker
	// chose under budget pressure, while Complete reports whether a shard fell off the
	// deadline regardless of the level. An operator reads it to see that a result was,
	// say, lexical-only or missing the lowest-static-rank shards by design.
	Degraded DegradeLevel
}

// Complete reports whether every contributing shard responded by the deadline, so the
// top-k is over the whole contributing set rather than a subset. It is the flag the
// serve path surfaces to the client.
func (r Results) Complete() bool { return r.ShardsOK == r.ShardsTotal }

// Search fans the query out to the routed shards, gathers their candidates, merges
// the per-shard ranked lists into global ranked lists, and runs the global rerank,
// returning the fleet-wide top-k. It is the back-compatible shape that drops the
// completeness indicator; SearchComplete returns the same top-k with the count of
// shards reached so a caller can tell a complete answer from a degraded one.
func (b *Broker) Search(ctx context.Context, q Query) []Hit {
	return b.SearchComplete(ctx, q).Hits
}

// SearchComplete is Search with the completeness indicator doc 11 requires. The
// fan-out waits for the contributing shards up to the deadline and a shard that has
// not responded is dropped, so the result is exact over the shards that responded and
// missing a dropped shard's contribution, and the returned Results says how many of
// the contributing shards that is. A slow shard cannot hold the whole query past its
// budget, because the collection stops at the deadline rather than waiting for every
// dispatched shard.
func (b *Broker) SearchComplete(ctx context.Context, q Query) Results {
	return b.SearchDegraded(ctx, q, DegradeNone)
}

// SetResultCache wires a result cache into the broker, the cache the cached search path
// consults. A broker with no cache wired runs every query through the cascade; wiring a
// cache is how a deployment turns it on. It is set once at construction, before the
// broker serves, so it needs no synchronization of its own.
func (b *Broker) SetResultCache(c *ResultCache) { b.cache = c }

// ResultCache returns the broker's result cache, or nil if none is wired, so the publish
// lifecycle can clear it on a collection change.
func (b *Broker) ResultCache() *ResultCache { return b.cache }

// SearchCached serves a query through the result cache when one is wired: a hit returns
// the cached ranked top-k without re-running the cascade, and a miss runs the budgeted
// search and caches the result. The bool reports whether the query was a cache hit.
//
// Only a complete, full-quality result is cached. A partial result (a shard dropped at
// the deadline) or a degraded one (served at a lower rung under budget pressure) is
// transient and lower quality, so caching it would serve a stale degraded answer to
// later queries that had the budget to do better; the cache holds only results that ran
// the whole cascade over every contributing shard. A hit always serves that good result
// even when the current query's budget is tight, which is the point: the cache takes the
// head of the distribution off the cascade entirely.
//
// With no cache wired this is SearchWithinBudget plus a false hit flag, so a caller can
// use it uniformly whether or not a cache is configured.
func (b *Broker) SearchCached(ctx context.Context, q Query) (Results, bool) {
	if b.cache != nil {
		if hits, total, ok := b.cache.Get(q); ok {
			return Results{Hits: hits, ShardsTotal: total, ShardsOK: total, Degraded: DegradeNone}, true
		}
	}
	res := b.SearchWithinBudget(ctx, q)
	if b.cache != nil && res.Complete() && res.Degraded == DegradeNone {
		b.cache.Put(q, res.Hits, res.ShardsTotal)
	}
	return res, false
}

// SearchWithinBudget runs the query at the degradation level the remaining deadline
// budget calls for, the policy half of the degradation order: a query entering with
// most of its budget left runs at full quality, and one entering with little left
// serves a cheaper, lower-quality result within budget rather than overrunning it. With
// no deadline on the context it runs at full quality. The chosen level is reported in
// the result's Degraded field so the reduction is observable.
func (b *Broker) SearchWithinBudget(ctx context.Context, q Query) Results {
	level := DegradeNone
	if dl, ok := ctx.Deadline(); ok {
		level = DegradeForBudget(time.Until(dl))
	}
	return b.SearchDegraded(ctx, q, level)
}

// SearchDegraded is SearchComplete at a chosen rung of the fixed degradation ladder
// (doc 11, "The degradation order"). The levers fire in the spec's order: shrink the
// per-shard L0, drop the dense plane to serve lexical-only, drop the lowest-static-rank
// shards from the fan-out, and trim the broker L2 candidate count, each rung cumulative
// over the ones below it. Every rung still ranks its smaller candidate set through the
// full L1 and L2 stages, never skipping a stage, so a degraded result is a ranked
// smaller set rather than an unranked one. The drop-a-slow-shard-at-the-deadline step
// is orthogonal and automatic in the fan-out at any level, reported through Complete.
func (b *Broker) SearchDegraded(ctx context.Context, q Query, level DegradeLevel) Results {
	casc, survivors, rows, total, ok := b.candidates(ctx, q, level)
	cands := casc.ScoreRows(survivors, rows, q.K)
	hits := make([]Hit, len(cands))
	for i, c := range cands {
		hits[i] = Hit{DocID: c.DocID, Score: c.Score}
	}
	return Results{Hits: hits, ShardsTotal: total, ShardsOK: ok, Degraded: level}
}

// FeatureHit is a retrieval candidate paired with the exact L2 feature row the cascade
// would score it on: the matrix row gathered from its shard followed by the online
// query-dependent features. It is what SearchFeatures returns so an offline trainer sees
// the same feature vector the serving path builds, the spec's training-and-serving
// identical-features requirement.
type FeatureHit struct {
	DocID uint32
	Row   []float64
}

// SearchFeatures runs the cascade up to but not including the L2 model: it retrieves,
// fuses the planes, runs the L1 linear cut, and extracts each survivor's full L2 feature
// row, returning the survivors paired with those rows instead of the model's order. It
// is the seam the training bootstrap retrieves candidates through, so the rows it labels
// and fits the model over are byte-for-byte the rows ScoreRows consumes at serve time,
// which is what makes a model trained on them see identical features when it ranks. It
// runs at full quality, with no degradation, because the trainer wants the complete
// candidate pool the spec's bootstrap pools and judges.
func (b *Broker) SearchFeatures(ctx context.Context, q Query) []FeatureHit {
	_, survivors, rows, _, _ := b.candidates(ctx, q, DegradeNone)
	out := make([]FeatureHit, len(survivors))
	for i, c := range survivors {
		out[i] = FeatureHit{DocID: c.DocID, Row: rows[i]}
	}
	return out
}

// candidates runs the shared cascade front: it routes the query, gathers the global idf,
// fans out to the shards, merges the per-plane ranked lists, runs the L1 linear cut, and
// extracts the survivors' full L2 feature rows, returning the per-query cascade (degraded
// or not), the L1 survivors, their feature rows aligned to the survivors, and the shard
// counts. It is the half SearchDegraded and SearchFeatures share: the former scores the
// rows with the L2 model and returns ranked hits, the latter returns the rows themselves.
// Pulling it out keeps the two paths byte-identical through retrieval and extraction, so
// a candidate's training row is exactly its serving row.
func (b *Broker) candidates(ctx context.Context, q Query, level DegradeLevel) (*rank.Cascade, []rank.Candidate, [][]float64, int, int) {
	// Load the served snapshot once at entry, take a reference on it, and use it to the end.
	// A concurrent publish or retire swaps in a new snapshot for the next query, but this
	// query routes, gathers idf, fans out, and reranks against one consistent shard set, so a
	// swap never splits a query across two sets. The reference keeps the snapshot's shards
	// mapped until the query finishes even if a retire supersedes the snapshot mid-query, so a
	// retired shard is reclaimed only after this query lets go of it, never under its read.
	st := b.acquire()
	defer b.release(st)
	d := degradationFor(level)
	// Analyze the query once at the broker and ship the term set to every shard, so the
	// analysis chain runs one time per query rather than once per shard the fan-out
	// visits. The shards score q.Terms directly; routing and the idf gather read the
	// same set. A query that already carries Terms (a pre-parsed ParsedQuery) keeps them.
	if q.Terms == nil {
		q.Terms = q.lexTerms()
	}
	// Lever 1: shrink the per-shard L0 candidate width, pushed down to every shard's
	// retrieval. Lever 2: drop the dense plane by clearing the dense query vector, so the
	// shards skip the encode and dense recall and serve lexical-only.
	if d.l0 > 0 {
		q.L0 = d.l0
	}
	if d.dropDense {
		q.Vector = nil
	}
	targets := st.routeTerms(q.Terms)
	// Lever 3: drop the lowest-static-rank shards from the routed set before the gather
	// and the fan-out, so the dropped shards cost nothing downstream. ShardsTotal then
	// reflects the shards actually queried, so the completeness flag stays honest about
	// deadline drops while the Degraded level reports the deliberate reduction.
	if d.dropShardFrac > 0 {
		targets = st.dropLowStatic(targets, d.dropShardFrac)
	}
	// Phase one of distributed exact idf: gather each query term's df across the routed
	// shards and turn it into one collection-wide idf the fan-out scores every shard
	// against. Without this each shard would use its local idf, and the merge would
	// favor whichever shard happens to hold a globally rare term densely. A query that
	// already carries idf overrides, or has no lexical text, skips the gather.
	if len(q.Terms) > 0 && q.TermIDF == nil {
		q.TermIDF = b.globalIDF(ctx, st, q.Terms, targets)
	}
	results, ok := b.fanOut(ctx, st, q, targets)

	var allLex, allDense []scored
	feats := make(map[uint32][]float64)
	owner := make(map[uint32]int) // global id -> index of the shard that holds it
	for _, r := range results {
		allLex = append(allLex, r.lex...)
		allDense = append(allDense, r.dense...)
		for id, fv := range r.feats {
			feats[id] = fv
			owner[id] = r.shard
		}
	}
	// Merge the per-shard ranked lists into one global ranked list per plane by score
	// descending, the order a single index over every shard would have produced.
	sortByScore(allLex)
	sortByScore(allDense)

	// L1 reads the matrix row already gathered from each shard. L2 needs the online
	// query-dependent features, which depend on the candidate's text and so must be
	// extracted at the shard that holds it: the rerank routes each survivor back to
	// its owning shard's per-query extractor, built once and reused across that
	// shard's survivors. Online extraction runs only over the L1 survivors, the
	// bounded set the spec's budget allots the per-candidate text work.
	// The per-field average lengths the online BM25F normalizes by are the broker's own
	// fleet averages, unless an aggregator above pushed down the deployment-wide averages
	// for a partitioned collection, in which case the broker normalizes against those so
	// its L2 scores land on the same scale as its sibling brokers' (doc 11, the
	// partitioned-GlobalStats case). A single broker serving directly gets q.AvgFieldLen
	// nil and uses its own averages, unchanged.
	avgField := st.stats.AvgFieldLen
	if q.AvgFieldLen != nil {
		avgField = *q.AvgFieldLen
	}
	l1feat := func(id uint32) []float64 { return feats[id] }
	// Lever 4: trim the broker L2 candidate count. The cascade is read-only and shared
	// across concurrent queries, so the trim is a per-query shallow copy that shrinks the
	// L1 cut while sharing the immutable L1 and L2 models, never mutating the broker's
	// cascade. The full L1 and L2 stages still run, over a smaller survivor set.
	casc := b.cascade
	if d.l1Keep > 0 {
		v := *b.cascade
		v.L1Keep = d.l1Keep
		casc = &v
	}
	// The cascade is split at the L2 boundary so the broker can extract the survivors'
	// online feature rows in parallel between the L1 cut and the model scoring. The
	// per-stage breakdown found this extraction (a zstd body decode and a field scan per
	// survivor) is the dominant cost of the cascade, and the ~L1Keep survivors decode
	// independently, so fanning the extraction across workers divides its wall time by
	// the core count while the cheap serial scoring pass is unchanged.
	survivors := casc.Survivors(localIDs(allLex), localIDs(allDense), l1feat)
	rows := b.extractRows(survivors, feats, owner, st, q, avgField)
	return casc, survivors, rows, len(targets), ok
}

// extractRows builds the full L2 feature row for each survivor, the matrix row
// gathered from its owning shard followed by the online query-dependent features
// extracted from that shard's text, returning rows aligned with survivors. It is the
// parallel half of the broker rerank: the survivors decode independently, so the
// extraction fans across workers, each writing a disjoint slice of rows so the result
// is identical to a serial pass regardless of scheduling.
//
// The online extractor holds per-query scratch it reuses across the survivors it
// scores and is documented to serve one query on one goroutine, so each worker builds
// its own extractor per shard it touches rather than sharing one across goroutines.
// The feature row an extractor produces depends only on the query, the shard's
// regions, and the document, never on which survivors it scored before, so a
// per-worker extractor yields the same row a single shared one would, which is what
// makes the parallel extraction byte-for-byte equal to the serial l2feat it replaces.
// A survivor whose owner is unknown (no shard claimed it, the matrix-only fallback)
// takes its gathered matrix row unchanged, exactly as the serial path returned base.
func (b *Broker) extractRows(survivors []rank.Candidate, feats map[uint32][]float64, owner map[uint32]int, st *brokerState, q Query, avgField [3]float64) [][]float64 {
	rows := make([][]float64, len(survivors))
	if len(survivors) == 0 {
		return rows
	}
	extractOne := func(exts map[int]*onlineExtractor, id uint32) []float64 {
		base := feats[id]
		si, ok := owner[id]
		if !ok {
			return base
		}
		s := st.shards[si]
		ext := exts[si]
		if ext == nil {
			ext = s.newOnline(q, q.TermIDF, avgField)
			exts[si] = ext
		}
		return s.l2Row(base, ext, id-s.nodeBase)
	}

	workers := b.maxConcurrency
	if workers < 1 {
		workers = 1
	}
	if workers > len(survivors) {
		workers = len(survivors)
	}
	// A single worker is the serial path, which avoids the goroutine and channel cost
	// for the small survivor sets a selective query leaves.
	if workers == 1 {
		exts := make(map[int]*onlineExtractor)
		for i, cd := range survivors {
			rows[i] = extractOne(exts, cd.DocID)
		}
		return rows
	}

	// Partition the survivors into contiguous chunks, one per worker, so each worker
	// writes a disjoint range of rows and no two goroutines touch the same index. The
	// last chunk takes the remainder.
	var wg sync.WaitGroup
	chunk := (len(survivors) + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= len(survivors) {
			break
		}
		hi := lo + chunk
		if hi > len(survivors) {
			hi = len(survivors)
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			exts := make(map[int]*onlineExtractor)
			for i := lo; i < hi; i++ {
				rows[i] = extractOne(exts, survivors[i].DocID)
			}
		}(lo, hi)
	}
	wg.Wait()
	return rows
}

// fanOut runs the routed shards' retrievals concurrently, bounded by the broker's
// concurrency limit, and returns each responding shard's result shifted into the
// global id space along with the count of shards that responded. It is deadline-aware
// on both ends: a cancelled context stops dispatching further shards, and the
// collection stops at the deadline rather than waiting for a dispatched-but-slow
// shard, so a slow shard is dropped instead of holding the query past its budget. A
// dropped shard's slice of the corpus is simply absent from the merge, which the
// returned count reports so the caller can flag the result partial.
func (b *Broker) fanOut(ctx context.Context, st *brokerState, q Query, targets []int) ([]*shardResult, int) {
	// The channel is buffered to the full target count so a shard goroutine that
	// finishes after the collection has stopped at the deadline can still send and
	// exit rather than leaking, blocked on a reader that has gone away.
	out := make(chan *shardResult, len(targets))
	sem := make(chan struct{}, b.maxConcurrency)
	dispatched := 0
dispatch:
	for _, si := range targets {
		select {
		case <-ctx.Done():
			break dispatch
		case sem <- struct{}{}:
		}
		dispatched++
		go func(si int) {
			defer func() { <-sem }()
			s := st.shards[si]
			lex, dense, feats, completed := s.retrieve(ctx, q)
			if !completed {
				// The deadline passed while this shard was retrieving, so it abandoned the
				// rest of its planes. Drop it: not sending leaves it out of the merge and
				// uncounted in ShardsOK, so the collection rolls it up as not responded, the
				// same honest partial a shard the collection stopped waiting for produces.
				return
			}
			base := s.nodeBase
			gl := make([]scored, len(lex))
			for j, c := range lex {
				gl[j] = scored{docID: base + c.docID, score: c.score}
			}
			gd := make([]scored, len(dense))
			for j, c := range dense {
				gd[j] = scored{docID: base + c.docID, score: c.score}
			}
			gf := make(map[uint32][]float64, len(feats))
			for id, fv := range feats {
				gf[base+id] = fv
			}
			out <- &shardResult{shard: si, lex: gl, dense: gd, feats: gf}
		}(si)
	}
	results := make([]*shardResult, 0, dispatched)
	for len(results) < dispatched {
		select {
		case r := <-out:
			results = append(results, r)
		case <-ctx.Done():
			return results, len(results)
		}
	}
	return results, len(results)
}

// globalIDF gathers each query term's document frequency across the routed shards and
// returns the collection-wide idf per term. The document count is the fleet-wide N from
// the broker's statistics, the same N a single index over every shard would divide by,
// and the df is summed over exactly the shards routing selected, which for a lexical
// term are all the shards that hold it, so the sum is the term's true collection df. An
// empty result lets the fan-out fall back to shard-local idf.
func (b *Broker) globalIDF(ctx context.Context, st *brokerState, terms []string, targets []int) map[string]float64 {
	return idfFromDF(b.gatherDF(ctx, st, terms, targets), st.stats.DocCount)
}

// gatherDF sums each term's document frequency across a set of this broker's shards. The
// gather reads only bloom filters and dictionaries, so it is cheap next to the retrieval
// it precedes; it runs concurrently under the broker's concurrency bound and a cancelled
// context returns whatever has been gathered, which only loosens an idf, never corrupts a
// score. The returned map holds only the terms some shard carried, so a term no shard
// holds is absent rather than zero.
func (b *Broker) gatherDF(ctx context.Context, st *brokerState, terms []string, targets []int) map[string]uint32 {
	df := make(map[string]uint32)
	var mu sync.Mutex
	sem := make(chan struct{}, b.maxConcurrency)
	var wg sync.WaitGroup
	for _, si := range targets {
		select {
		case <-ctx.Done():
			wg.Wait()
			return df
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(si int) {
			defer wg.Done()
			defer func() { <-sem }()
			local := st.shards[si].LexDocFreqs(terms)
			if len(local) == 0 {
				return
			}
			mu.Lock()
			for t, f := range local {
				df[t] += f
			}
			mu.Unlock()
		}(si)
	}
	wg.Wait()
	return df
}

// NumDocs is the fleet-wide document count beneath this broker, the N idf divides by, so
// an aggregator over several brokers sums their NumDocs into the fleet N it computes one
// shared idf against (the Searcher contract).
func (b *Broker) NumDocs() uint64 { return b.loadState().stats.DocCount }

// DocFreqs sums each term's document frequency across the broker's shards that hold it, so
// an aggregator can add the per-broker counts into the fleet-wide df no single broker can
// see (the Searcher contract). It routes the terms to the shards that carry them and
// gathers only those, the same cheap bloom-and-dictionary read globalIDF uses, and honors
// the deadline by returning whatever it has gathered when the context is cancelled.
func (b *Broker) DocFreqs(ctx context.Context, terms []string) map[string]uint32 {
	if len(terms) == 0 {
		return nil
	}
	st := b.acquire()
	defer b.release(st)
	return b.gatherDF(ctx, st, terms, st.routeTerms(terms))
}

// idfFromDF turns gathered per-term document frequencies into per-term idf against the
// collection-wide document count. It returns nil for an empty gather so the caller
// leaves the query's idf override unset and the shards score with their local idf.
func idfFromDF(df map[string]uint32, n uint64) map[string]float64 {
	if len(df) == 0 {
		return nil
	}
	out := make(map[string]float64, len(df))
	for t, f := range df {
		out[t] = lexical.IDF(n, uint64(f))
	}
	return out
}

// sortByScore orders a candidate list by score descending, ties broken by ascending
// id so the merged order is deterministic across runs and across shard partitionings.
func sortByScore(ss []scored) {
	sort.Slice(ss, func(a, b int) bool {
		if ss[a].score != ss[b].score {
			return ss[a].score > ss[b].score
		}
		return ss[a].docID < ss[b].docID
	})
}
