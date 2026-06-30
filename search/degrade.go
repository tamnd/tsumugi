package search

import (
	"sort"
	"time"

	"github.com/tamnd/tsumugi/rank"
)

// DegradeLevel is a rung on the fixed degradation ladder of doc 11. Under budget
// pressure the broker climbs the ladder in a fixed order, so a degraded result is a
// known degradation rather than an ad hoc one: shrink L0, then drop the dense plane,
// then drop the lowest-static-rank shards, then trim the broker L2. Each rung is
// cumulative, turning on its own lever and every lever below it, because the order
// trades the least quality for the most budget at each step, so a higher rung is
// reached only when the rungs below it did not free enough budget.
//
// The "drop a slow shard at the deadline" step of the spec's box is not a rung here.
// It is automatic in the fan-out: a shard that misses the deadline is dropped at any
// level, and the drop is reported through the completeness flag (slice 53), which is a
// separate signal from the level reported here. So a result carries two independent
// degradation facts: the level the broker chose under budget pressure, and whether a
// shard fell off the deadline anyway.
type DegradeLevel int

const (
	// DegradeNone is full quality: every plane, every routed shard, the full L0 and L2
	// widths.
	DegradeNone DegradeLevel = iota
	// DegradeL0 shrinks the per-shard L0 candidate count, the cheapest rung: it ranks a
	// smaller candidate set precisely rather than skipping a stage (doc 09 anytime
	// traversal).
	DegradeL0
	// DegradeDense adds dropping the dense plane, serving lexical-only and recovering the
	// encode and dense-recall budget for the smallest quality loss, since the
	// learned-sparse plane already captures most of the semantic win.
	DegradeDense
	// DegradeShards adds dropping the lowest-static-rank shards, lowering the fan-out and
	// the merged set at the cost of recall over the shards least likely to hold a winner.
	DegradeShards
	// DegradeL2 adds trimming the broker L2 candidate count, trading a little precision
	// for cost, the last rung before the floor.
	DegradeL2
)

// String names the rung for the metadata the serve path reports, so an operator can see
// which degradations fired on a request.
func (l DegradeLevel) String() string {
	switch l {
	case DegradeNone:
		return "none"
	case DegradeL0:
		return "shrink-l0"
	case DegradeDense:
		return "drop-dense"
	case DegradeShards:
		return "drop-shards"
	case DegradeL2:
		return "trim-l2"
	default:
		return "unknown"
	}
}

// degradation is the resolved set of levers a DegradeLevel turns on, the cumulative
// ladder made concrete. The shrink amounts are the canon's "least quality for most
// budget" choices: L0 to a quarter, a quarter of the routed shards dropped, the L2
// candidate count to a half. Each rung still ranks a smaller set with the full L1 and
// L2 stages rather than skipping a stage, the spec's never-skip-L1-or-L2 rule.
type degradation struct {
	l0            int     // per-shard L0 override; 0 leaves the shard default
	dropDense     bool    // serve lexical-only, dropping the dense plane
	dropShardFrac float64 // fraction of routed shards to drop, lowest static rank first
	l1Keep        int     // broker L2 candidate count override; 0 leaves the cascade default
}

// degradationFor resolves a level into the concrete levers it turns on, cumulatively:
// every lever of a lower rung stays on at a higher one, because the ladder is
// least-cost-first and a higher rung is reached only after the lower rungs did not free
// enough budget.
func degradationFor(level DegradeLevel) degradation {
	var d degradation
	if level >= DegradeL0 {
		d.l0 = DefaultL0 / 4
	}
	if level >= DegradeDense {
		d.dropDense = true
	}
	if level >= DegradeShards {
		d.dropShardFrac = 0.25
	}
	if level >= DegradeL2 {
		d.l1Keep = rank.DefaultL1Keep / 2
	}
	return d
}

// DegradeForBudget maps the time budget remaining at query entry to a rung on the
// degradation ladder, the policy that turns budget pressure into a fixed, predictable
// degradation rather than an overrun. The thresholds are fractions of the
// ten-millisecond serving budget doc 11 gates on: with most of the budget left the
// query runs at full quality, and as the remaining budget falls the broker climbs the
// ladder so the query still answers within budget. A query that enters with the budget
// already gone serves the cheapest result the ladder offers rather than nothing.
func DegradeForBudget(remaining time.Duration) DegradeLevel {
	switch {
	case remaining >= 8*time.Millisecond:
		return DegradeNone
	case remaining >= 6*time.Millisecond:
		return DegradeL0
	case remaining >= 4*time.Millisecond:
		return DegradeDense
	case remaining >= 2*time.Millisecond:
		return DegradeShards
	default:
		return DegradeL2
	}
}

// dropLowStatic removes the lowest-static-rank shards from the routed target set,
// keeping all but the dropped fraction. A shard's static rank is the highest composite
// static rank among its documents, the best a top-k winner it could hold would score on
// the query-independent prior, so dropping the shards with the lowest such value drops
// the shards least likely to hold a winner, the spec's least-cost-first ordering. The
// fan-out cost falls with the shard count while the recall lost is over the shards that
// were the least likely to win anyway. The kept set is returned in shard-index order so
// the fan-out stays deterministic, and at least one shard is always kept so a query
// never degrades to no shards.
func (st *brokerState) dropLowStatic(targets []int, frac float64) []int {
	if frac <= 0 || len(targets) <= 1 {
		return targets
	}
	drop := int(float64(len(targets)) * frac)
	if drop <= 0 {
		return targets
	}
	if drop >= len(targets) {
		drop = len(targets) - 1
	}
	static := st.shardStatics()
	kept := append([]int(nil), targets...)
	// Order by static rank descending, ties by shard index ascending, so the cut is
	// deterministic regardless of the routing order, then drop the tail.
	sort.SliceStable(kept, func(i, j int) bool {
		si, sj := static[kept[i]], static[kept[j]]
		if si != sj {
			return si > sj
		}
		return kept[i] < kept[j]
	})
	kept = kept[:len(kept)-drop]
	sort.Ints(kept)
	return kept
}

// shardStatics returns the per-shard static-rank summary, the highest static rank in
// each shard, computed once on first use and cached on the serving snapshot. It is
// computed lazily rather than when the snapshot is built so a snapshot whose queries
// never degrade to dropping shards never pays the scan, which keeps the common startup
// proportional to the query vocabulary, not the corpus. The scan reads one feature
// column per document and runs only the first time the shard-dropping rung fires on this
// snapshot; a publish or retire builds a fresh snapshot that recomputes it on demand.
func (st *brokerState) shardStatics() []float64 {
	st.staticOnce.Do(func() {
		st.shardStatic = make([]float64, len(st.shards))
		for i, s := range st.shards {
			st.shardStatic[i] = s.MaxStaticRank()
		}
	})
	return st.shardStatic
}
