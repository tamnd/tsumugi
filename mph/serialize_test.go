package mph

import (
	"fmt"
	"testing"
)

// makeDir builds a directory over n synthetic keys, each mapped to its index, the
// shape the recrawl membership directory takes (a canonical URL to a global id).
func makeDir(n int) (*Dir, [][]byte) {
	keys := make([][]byte, n)
	ids := make([]uint32, n)
	for i := 0; i < n; i++ {
		keys[i] = []byte(fmt.Sprintf("https://host%d.example/p%d", i%97, i))
		ids[i] = uint32(i * 3)
	}
	return BuildDir(keys, ids, DefaultGamma), keys
}

// TestDirRoundTrip serializes a directory and reads it back, then checks the reloaded
// directory answers every member lookup with the same value the original did and
// rejects a non-member exactly as the original does. This is the property the
// persisted recrawl directory rests on: a directory loaded from disk is the same
// membership oracle as the one the build held in memory.
func TestDirRoundTrip(t *testing.T) {
	d, keys := makeDir(5000)
	b := d.Append(nil)
	got, n, err := ReadDir(b)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if n != len(b) {
		t.Fatalf("ReadDir consumed %d bytes, want %d", n, len(b))
	}
	if got.Len() != d.Len() {
		t.Fatalf("reloaded Len = %d, want %d", got.Len(), d.Len())
	}
	for i, k := range keys {
		wantV, wantOK := d.Lookup(k)
		gotV, gotOK := got.Lookup(k)
		if gotOK != wantOK || gotV != wantV {
			t.Fatalf("key %q: reloaded (%d,%v), original (%d,%v)", k, gotV, gotOK, wantV, wantOK)
		}
		if !gotOK || gotV != uint32(i*3) {
			t.Fatalf("key %q: value %d, want %d (member)", k, gotV, i*3)
		}
	}
	// Non-members are rejected on both sides, the membership check the recrawl dedup
	// keys off; a directory that admitted strangers would drop real new pages.
	for i := 0; i < 5000; i++ {
		k := []byte(fmt.Sprintf("https://absent%d.example/q%d", i, i))
		if _, ok := got.Lookup(k); ok {
			if _, orig := d.Lookup(k); !orig {
				t.Fatalf("reloaded admits non-member %q the original rejected", k)
			}
		}
	}
}

// TestDirRoundTripBytesStable checks the serialization is deterministic: the same
// directory serializes to the identical bytes twice, the reproducibility a build's
// byte-identical artifact rests on.
func TestDirRoundTripBytesStable(t *testing.T) {
	d, _ := makeDir(2000)
	a := d.Append(nil)
	b := d.Append(nil)
	if len(a) != len(b) {
		t.Fatalf("two serializations differ in length: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("serialization not deterministic at byte %d", i)
		}
	}
}

// TestReadDirRejectsTruncated checks a short or damaged buffer is refused rather than
// parsed into a partial directory, so a torn artifact triggers a rebuild.
func TestReadDirRejectsTruncated(t *testing.T) {
	d, _ := makeDir(500)
	full := d.Append(nil)
	for _, cut := range []int{0, 4, 16, len(full) / 2, len(full) - 1} {
		if _, _, err := ReadDir(full[:cut]); err == nil {
			t.Errorf("ReadDir accepted a buffer truncated to %d bytes", cut)
		}
	}
}

// TestDirRoundTripEmpty checks the degenerate empty directory survives the round trip,
// the no-documents edge a fresh collection or an all-deduped add can produce.
func TestDirRoundTripEmpty(t *testing.T) {
	d := BuildDir(nil, nil, DefaultGamma)
	got, _, err := ReadDir(d.Append(nil))
	if err != nil {
		t.Fatalf("ReadDir empty: %v", err)
	}
	if got.Len() != 0 {
		t.Fatalf("empty reloaded Len = %d, want 0", got.Len())
	}
	if _, ok := got.Lookup([]byte("https://x.example/")); ok {
		t.Fatalf("empty directory claims a member")
	}
}
