package search

import (
	"context"
	"sync"
	"sync/atomic"
)

// Admission bounds the number of in-flight searches so the broker degrades by rejecting
// rather than by collapsing under unbounded concurrency (doc 11, "Admission control").
// A request acquires a slot at entry, and if no slot is available it is rejected fast
// with a busy signal rather than queued into unbounded latency, because a queued request
// still consumes memory and still misses its deadline, so rejecting it fast is kinder
// than admitting it into a queue it will time out in.
//
// The slot is held for the whole search, from acquisition until the response is fully
// written, which is the discipline tatami fixed a real bug by getting right: releasing
// the slot at result-assembly time, before the serialization and the socket write, lets
// a slot be reused while the previous search is still finishing, which over-admits and
// breaks the bound the slot count was supposed to enforce. The caller holds the slot with
// a deferred Release at the top of the handler so it covers the encode and the write.
//
// The counter is lock-free: Acquire and Release move a single atomic in-flight count, so
// the admission counter is the only shared mutable state on the request path besides the
// segment cache, and neither takes a lock. A capacity of zero or less disables admission,
// so every Acquire returns a no-op slot and the broker is unbounded, the default for a
// deployment that has not sized its capacity.
type Admission struct {
	cap      int64
	inFlight int64
	closed   int64

	// drained is closed exactly once, when the admission is closed and the last in-flight
	// search releases its slot, so Drain can wait on it. drainOnce guards the close so a
	// Release that brings the count to zero after Close fires it at most once.
	drained   chan struct{}
	drainOnce sync.Once
}

// NewAdmission returns an admission gate bounding in-flight searches at capacity. A
// capacity of zero or less disables the gate: every Acquire admits, so the broker runs
// unbounded, the behavior of a deployment that has not set a limit.
func NewAdmission(capacity int) *Admission {
	return &Admission{cap: int64(capacity), drained: make(chan struct{})}
}

// Slot is a held admission grant. It is returned by Acquire and must be released exactly
// once, with Release, when the search is fully finished including the response write.
// Release is idempotent: a double release is a no-op, so a deferred Release is safe even
// on a path that released early.
type Slot struct {
	a    *Admission
	done int32
}

// Acquire takes an in-flight slot, returning nil if the broker is at capacity or shutting
// down, so the caller rejects the request fast rather than queuing it. A disabled gate
// (capacity <= 0) always admits with a no-op slot. The increment is a lock-free CAS loop
// on the in-flight count, and the closed flag is re-checked after the increment so a
// shutdown that races the acquire does not admit a search the drain will not wait for.
func (a *Admission) Acquire() *Slot {
	if a == nil || a.cap <= 0 {
		return &Slot{} // disabled: admit with a no-op slot, nothing to release
	}
	if atomic.LoadInt64(&a.closed) != 0 {
		return nil
	}
	for {
		cur := atomic.LoadInt64(&a.inFlight)
		if cur >= a.cap {
			return nil // at capacity: reject fast
		}
		if atomic.CompareAndSwapInt64(&a.inFlight, cur, cur+1) {
			// Re-check after the increment: a Close that landed between the load and the
			// CAS must not leave an admitted search the drain cannot account for, so back
			// the increment out and reject.
			if atomic.LoadInt64(&a.closed) != 0 {
				a.release()
				return nil
			}
			return &Slot{a: a}
		}
	}
}

// Release returns the slot's in-flight count. It is idempotent and safe to call on a
// no-op slot (a disabled gate), so a deferred Release at the top of the handler is always
// correct.
func (s *Slot) Release() {
	if s == nil || s.a == nil {
		return // no-op slot from a disabled gate
	}
	if !atomic.CompareAndSwapInt32(&s.done, 0, 1) {
		return // already released
	}
	s.a.release()
}

// release drops the in-flight count by one and, if the gate is closed and the last search
// has finished, signals the drain. It is the shared tail of Slot.Release and the
// acquire-race back-out.
func (a *Admission) release() {
	n := atomic.AddInt64(&a.inFlight, -1)
	if n == 0 && atomic.LoadInt64(&a.closed) != 0 {
		a.signalDrained()
	}
}

// InFlight reports the number of searches currently holding a slot, the load metric an
// operator watches against the capacity to see when the broker is near capacity and
// shedding load (doc 11, "Metrics").
func (a *Admission) InFlight() int {
	if a == nil {
		return 0
	}
	return int(atomic.LoadInt64(&a.inFlight))
}

// Cap reports the configured in-flight capacity, zero for a disabled gate.
func (a *Admission) Cap() int {
	if a == nil || a.cap <= 0 {
		return 0
	}
	return int(a.cap)
}

// Close stops the gate from admitting new searches: every subsequent Acquire returns nil.
// In-flight searches keep their slots and finish normally. Close is idempotent. A disabled
// gate ignores Close, since it admits everything and has nothing to drain.
func (a *Admission) Close() {
	if a == nil || a.cap <= 0 {
		return
	}
	atomic.StoreInt64(&a.closed, 1)
	// A Close with nothing in flight drains immediately, so a shutdown on an idle broker
	// returns at once rather than waiting for a release that will never come.
	if atomic.LoadInt64(&a.inFlight) == 0 {
		a.signalDrained()
	}
}

// signalDrained closes the drained channel exactly once.
func (a *Admission) signalDrained() {
	a.drainOnce.Do(func() { close(a.drained) })
}

// Drain stops admitting and waits for the in-flight searches to finish, which it knows
// are finished when their slots are released, because a slot is held for the whole search
// including the response write. It is the clean shutdown the serve command needs: a deploy
// or a restart does not drop the queries that were in flight when it started.
//
// The wait is bounded by the context: an in-flight search is itself bounded by its
// per-request deadline, so the drain waits at most one deadline for the last search to
// finish or time out, which means a shutdown completes in bounded time even if a search is
// stuck on a slow shard. Drain returns nil once the in-flight count reaches zero, or the
// context's error if the deadline fires first. A disabled gate drains instantly.
func (a *Admission) Drain(ctx context.Context) error {
	if a == nil || a.cap <= 0 {
		return nil
	}
	a.Close()
	select {
	case <-a.drained:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
