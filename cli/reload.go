package cli

import (
	"fmt"
	"path/filepath"
	"sort"
	"sync"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/rank"
	"github.com/tamnd/tsumugi/search"
)

// reloader publishes and retires shards on a running broker so a deployment can change
// the served collection without a restart (doc 11 freshness, publish/retire lifecycle).
// It owns the broker's path-to-shard map, the one place that knows which open shard came
// from which file, so a retire can name the exact shard to remove and a publish can refuse
// a shard already served. The broker does the atomic shard-set swap and the cache
// invalidation; the reloader is the directory-facing layer that decides what to swap.
//
// Every method takes the reloader's mutex, so a sync racing an admin publish serializes
// rather than double-opening a file or double-retiring a shard. The mutex is separate from
// the broker's own publish mutex: the reloader's guards its served map, the broker's
// guards the state swap, and a query never takes either.
type reloader struct {
	mu     sync.Mutex
	dir    string
	model  *rank.Model
	broker *search.Broker

	// served maps a shard's base filename to the open shard the broker serves, so a retire
	// matches the broker's predicate by pointer identity rather than by a shard attribute
	// that two shards could share.
	served map[string]*search.Shard
}

// newReloader builds a reloader over the shards a broker was opened with, seeding the
// served map from the paths aligned with the shard slice. The model is the same compiled
// model the broker scores against, so a shard published later opens with an identical
// cascade.
func newReloader(dir string, model *rank.Model, broker *search.Broker, shards []*search.Shard, paths []string) *reloader {
	served := make(map[string]*search.Shard, len(paths))
	for i, p := range paths {
		served[filepath.Base(p)] = shards[i]
	}
	return &reloader{dir: dir, model: model, broker: broker, served: served}
}

// publish opens the named shard from the served directory, verifies its analyzer matches
// the broker's, publishes it, and records it. A name already served is a no-op, so a
// re-publish is idempotent. A shard whose recorded analyzer does not match is refused and
// closed rather than served, the same silent-wrong-results guard the startup loader applies.
func (rl *reloader) publish(name string) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.publishLocked(name)
}

func (rl *reloader) publishLocked(name string) error {
	if _, ok := rl.served[name]; ok {
		return nil
	}
	p := filepath.Join(rl.dir, name)
	s, err := search.OpenShard(p, newCascade(rl.model))
	if err != nil {
		return fmt.Errorf("open shard %s: %w", name, err)
	}
	if h, ok := s.AnalyzerHash(); ok && h != queryAnalyzerHash() {
		_ = s.Close()
		return fmt.Errorf("%w: shard %s is %#016x, broker analyzer is %#016x",
			collection.ErrAnalyzerMismatch, name, h, queryAnalyzerHash())
	}
	rl.broker.Publish(s)
	rl.served[name] = s
	return nil
}

// retire removes the named shard from the served set, returning whether it was served. The
// broker keeps the retired shard's mapping open until Close, so an in-flight query that
// loaded the old snapshot finishes safely; retire only stops new queries from seeing it.
func (rl *reloader) retire(name string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.retireLocked(name)
}

func (rl *reloader) retireLocked(name string) bool {
	s, ok := rl.served[name]
	if !ok {
		return false
	}
	rl.broker.Retire(func(x *search.Shard) bool { return x == s })
	delete(rl.served, name)
	return true
}

// sync brings the served set in line with the directory's current *.tsumugi files: it
// publishes every file not yet served and retires every served shard whose file has been
// removed, returning the counts. It globs the directory rather than reading the index
// artifact, so a freshly built shard dropped into the directory is picked up even before
// the manifest is rewritten, and a removed file is retired even if the manifest still names
// it. A shard that fails to open or fails the analyzer check is skipped and reported as the
// first error, so one bad file does not block the rest of the sweep.
func (rl *reloader) sync() (published, retired int, err error) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	matches, gerr := filepath.Glob(filepath.Join(rl.dir, "*.tsumugi"))
	if gerr != nil {
		return 0, 0, gerr
	}
	onDisk := make(map[string]struct{}, len(matches))
	for _, p := range matches {
		onDisk[filepath.Base(p)] = struct{}{}
	}

	// Publish in sorted order so a sweep is deterministic, and so the reported first error
	// is stable across runs rather than depending on glob or map order.
	newNames := make([]string, 0, len(onDisk))
	for name := range onDisk {
		if _, ok := rl.served[name]; !ok {
			newNames = append(newNames, name)
		}
	}
	sort.Strings(newNames)
	var firstErr error
	for _, name := range newNames {
		if e := rl.publishLocked(name); e != nil {
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		published++
	}

	// Retire over a snapshot of the served names, since retireLocked deletes from the map.
	gone := make([]string, 0)
	for name := range rl.served {
		if _, ok := onDisk[name]; !ok {
			gone = append(gone, name)
		}
	}
	for _, name := range gone {
		if rl.retireLocked(name) {
			retired++
		}
	}
	return published, retired, firstErr
}

// numServed is the number of shards the reloader currently tracks as served, for tests and
// the admin response.
func (rl *reloader) numServed() int {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return len(rl.served)
}
