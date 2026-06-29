package search

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAdmissionBoundsInFlight(t *testing.T) {
	a := NewAdmission(2)
	s1 := a.Acquire()
	s2 := a.Acquire()
	if s1 == nil || s2 == nil {
		t.Fatal("first two acquires should admit under capacity 2")
	}
	if got := a.InFlight(); got != 2 {
		t.Fatalf("in-flight = %d, want 2", got)
	}
	// At capacity: the third acquire is rejected fast rather than queued.
	if s3 := a.Acquire(); s3 != nil {
		t.Fatal("third acquire should be rejected at capacity 2")
	}
	// Releasing one frees exactly one slot.
	s1.Release()
	if got := a.InFlight(); got != 1 {
		t.Fatalf("in-flight after one release = %d, want 1", got)
	}
	s4 := a.Acquire()
	if s4 == nil {
		t.Fatal("acquire after a release should admit")
	}
	s2.Release()
	s4.Release()
	if got := a.InFlight(); got != 0 {
		t.Fatalf("in-flight after all releases = %d, want 0", got)
	}
}

func TestAdmissionReleaseIdempotent(t *testing.T) {
	a := NewAdmission(1)
	s := a.Acquire()
	if s == nil {
		t.Fatal("acquire should admit")
	}
	s.Release()
	s.Release() // double release must not drop the count below zero
	if got := a.InFlight(); got != 0 {
		t.Fatalf("in-flight after double release = %d, want 0", got)
	}
	// The double release must not have stolen a slot: capacity is fully available again.
	s2 := a.Acquire()
	if s2 == nil {
		t.Fatal("a slot should be free after the (idempotent) release")
	}
	s2.Release()
}

func TestAdmissionDisabled(t *testing.T) {
	for _, capacity := range []int{0, -1} {
		a := NewAdmission(capacity)
		// A disabled gate admits everything and tracks no in-flight count.
		for i := 0; i < 1000; i++ {
			s := a.Acquire()
			if s == nil {
				t.Fatalf("disabled gate (cap %d) rejected an acquire", capacity)
			}
			s.Release()
		}
		if got := a.Cap(); got != 0 {
			t.Fatalf("disabled gate Cap = %d, want 0", got)
		}
		// Drain on a disabled gate returns immediately.
		if err := a.Drain(context.Background()); err != nil {
			t.Fatalf("disabled gate Drain = %v, want nil", err)
		}
	}
}

func TestAdmissionNilSafe(t *testing.T) {
	var a *Admission
	s := a.Acquire()
	if s == nil {
		t.Fatal("nil gate should admit with a no-op slot")
	}
	s.Release() // must not panic
	if a.InFlight() != 0 || a.Cap() != 0 {
		t.Fatal("nil gate should report zero in-flight and capacity")
	}
	if err := a.Drain(context.Background()); err != nil {
		t.Fatalf("nil gate Drain = %v, want nil", err)
	}
}

func TestAdmissionCloseRejects(t *testing.T) {
	a := NewAdmission(4)
	s := a.Acquire()
	if s == nil {
		t.Fatal("acquire should admit before close")
	}
	a.Close()
	if a.Acquire() != nil {
		t.Fatal("acquire after close must be rejected")
	}
	a.Close() // idempotent
	s.Release()
}

func TestAdmissionDrainWaitsForInFlight(t *testing.T) {
	a := NewAdmission(2)
	s := a.Acquire()
	if s == nil {
		t.Fatal("acquire should admit")
	}
	done := make(chan error, 1)
	go func() { done <- a.Drain(context.Background()) }()

	// The drain must not return while a search still holds a slot.
	select {
	case <-done:
		t.Fatal("Drain returned while a slot was still held")
	case <-time.After(20 * time.Millisecond):
	}
	// Releasing the last slot lets the drain finish.
	s.Release()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Drain = %v, want nil", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Drain did not return after the last slot was released")
	}
}

func TestAdmissionDrainIdleReturnsImmediately(t *testing.T) {
	a := NewAdmission(2)
	if err := a.Drain(context.Background()); err != nil {
		t.Fatalf("Drain on an idle gate = %v, want nil", err)
	}
}

func TestAdmissionDrainBoundedByContext(t *testing.T) {
	a := NewAdmission(1)
	s := a.Acquire() // held for the whole test: the drain must time out on it
	if s == nil {
		t.Fatal("acquire should admit")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	err := a.Drain(ctx)
	if err == nil {
		t.Fatal("Drain should return the context error when a slot is stuck")
	}
	s.Release()
}

// BenchmarkAdmissionAcquireRelease times the lock-free acquire/release pair on the hot
// path, the per-request admission cost the broker pays before every search. It runs
// parallel to measure the contended cost, since the counter is the shared mutable state
// the request path contends on.
func BenchmarkAdmissionAcquireRelease(b *testing.B) {
	a := NewAdmission(1 << 20) // wide enough that the parallel run never rejects
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			s := a.Acquire()
			if s != nil {
				s.Release()
			}
		}
	})
}

// TestAdmissionConcurrentNeverOverAdmits hammers the gate from many goroutines and checks
// the in-flight count never exceeds the capacity, the lock-free bound holding under the
// race detector.
func TestAdmissionConcurrentNeverOverAdmits(t *testing.T) {
	const capacity = 8
	a := NewAdmission(capacity)
	var peak int64
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				s := a.Acquire()
				if s == nil {
					continue // rejected at capacity, the correct fast path
				}
				n := int64(a.InFlight())
				for {
					p := atomic.LoadInt64(&peak)
					if n <= p || atomic.CompareAndSwapInt64(&peak, p, n) {
						break
					}
				}
				s.Release()
			}
		}()
	}
	wg.Wait()
	if peak > capacity {
		t.Fatalf("observed in-flight peak %d exceeds capacity %d", peak, capacity)
	}
	if got := a.InFlight(); got != 0 {
		t.Fatalf("in-flight after all goroutines done = %d, want 0", got)
	}
}
