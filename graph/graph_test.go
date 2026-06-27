package graph

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// TestCodeRoundTrip exercises the bit-level codes over a wide range, including
// the zero and the boundaries where the code length steps up.
func TestCodeRoundTrip(t *testing.T) {
	vals := []uint64{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 100, 255, 256, 1023, 65535, 1 << 20, 1 << 40}
	for _, k := range []int{2, 3, 4} {
		w := &bitWriter{}
		for _, v := range vals {
			w.writeGamma(v)
			w.writeZeta(v, k)
		}
		for _, g := range []int64{0, 1, -1, 5, -5, 1000, -1000} {
			w.writeSignedGamma(g)
			w.writeSignedZeta(g, k)
		}
		buf := w.finish()
		r := newBitReader(buf, 0)
		for _, v := range vals {
			if got := r.readGamma(); got != v {
				t.Fatalf("k=%d gamma %d got %d", k, v, got)
			}
			if got := r.readZeta(k); got != v {
				t.Fatalf("k=%d zeta %d got %d", k, v, got)
			}
		}
		for _, g := range []int64{0, 1, -1, 5, -5, 1000, -1000} {
			if got := r.readSignedGamma(); got != g {
				t.Fatalf("signed gamma %d got %d", g, got)
			}
			if got := r.readSignedZeta(k); got != g {
				t.Fatalf("signed zeta %d got %d", g, got)
			}
		}
	}
}

// TestEFRoundTrip builds Elias-Fano over random monotone sequences and reads
// every value back, the offset-index contract the adjacency reader leans on.
func TestEFRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 50; trial++ {
		n := 1 + rng.Intn(500)
		vals := make([]uint64, n)
		var acc uint64
		for i := range vals {
			acc += uint64(rng.Intn(40))
			vals[i] = acc
		}
		e := buildEF(vals)
		got, err := decodeEF(e.encode())
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		for i := range vals {
			if v := got.get(i); v != vals[i] {
				t.Fatalf("trial %d idx %d got %d want %d", trial, i, v, vals[i])
			}
		}
	}
}

// naiveGraph mirrors what the builder should encode: sorted, deduped out-lists
// and the derived transpose.
func naiveGraph(n int, edges [][2]int) (out, in [][]int) {
	outSet := make([]map[int]bool, n)
	inSet := make([]map[int]bool, n)
	for i := 0; i < n; i++ {
		outSet[i] = map[int]bool{}
		inSet[i] = map[int]bool{}
	}
	for _, e := range edges {
		if e[0] == e[1] {
			continue
		}
		outSet[e[0]][e[1]] = true
		inSet[e[1]][e[0]] = true
	}
	out = make([][]int, n)
	in = make([][]int, n)
	for i := 0; i < n; i++ {
		out[i] = sortedKeys(outSet[i])
		in[i] = sortedKeys(inSet[i])
	}
	return
}

func sortedKeys(m map[int]bool) []int {
	if len(m) == 0 {
		return nil // match the decoder, which returns nil for an empty list
	}
	ks := make([]int, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Ints(ks)
	return ks
}

// TestAdjacencyRoundTrip is the M4 gate: a decoded region must reproduce the
// forward and transpose adjacency of the graph it was built from, exactly. It
// runs over random graphs with clustered targets so the reference, interval, and
// residual paths all fire.
func TestAdjacencyRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 40; trial++ {
		n := 2 + rng.Intn(400)
		b := NewBuilder(n)
		var edges [][2]int
		for u := 0; u < n; u++ {
			deg := rng.Intn(12)
			// Mix clustered runs (intervals) with scattered targets (residuals).
			base := rng.Intn(n)
			for k := 0; k < deg; k++ {
				var v int
				if rng.Intn(2) == 0 {
					v = (base + k) % n
				} else {
					v = rng.Intn(n)
				}
				b.AddEdge(u, v)
				edges = append(edges, [2]int{u, v})
			}
		}
		g, err := Open(b.Build())
		if err != nil {
			t.Fatalf("trial %d open: %v", trial, err)
		}
		wantOut, wantIn := naiveGraph(n, edges)
		for x := 0; x < n; x++ {
			if got := g.OutNeighbors(x); !reflect.DeepEqual(got, wantOut[x]) {
				t.Fatalf("trial %d out[%d]=%v want %v", trial, x, got, wantOut[x])
			}
			if got := g.InNeighbors(x); !reflect.DeepEqual(got, wantIn[x]) {
				t.Fatalf("trial %d in[%d]=%v want %v", trial, x, got, wantIn[x])
			}
			if g.OutDegree(x) != len(wantOut[x]) {
				t.Fatalf("trial %d outdeg[%d]=%d want %d", trial, x, g.OutDegree(x), len(wantOut[x]))
			}
			if g.InDegree(x) != len(wantIn[x]) {
				t.Fatalf("trial %d indeg[%d]=%d want %d", trial, x, g.InDegree(x), len(wantIn[x]))
			}
		}
	}
}

// TestIntervalHeavy forces long consecutive runs so the interval path dominates.
func TestIntervalHeavy(t *testing.T) {
	n := 1000
	b := NewBuilder(n)
	var edges [][2]int
	for u := 0; u < 50; u++ {
		for v := 100; v < 260; v++ { // one long run per source
			b.AddEdge(u, v)
			edges = append(edges, [2]int{u, v})
		}
	}
	g, err := Open(b.Build())
	if err != nil {
		t.Fatal(err)
	}
	wantOut, _ := naiveGraph(n, edges)
	for x := 0; x < 50; x++ {
		if got := g.OutNeighbors(x); !reflect.DeepEqual(got, wantOut[x]) {
			t.Fatalf("out[%d] mismatch", x)
		}
	}
}

// TestEmptyGraph builds a graph with nodes but no edges; every list is empty and
// every node is dangling.
func TestEmptyGraph(t *testing.T) {
	g, err := Open(NewBuilder(10).Build())
	if err != nil {
		t.Fatal(err)
	}
	if g.NodeCount() != 10 || g.EdgeCount() != 0 {
		t.Fatalf("nodes=%d edges=%d", g.NodeCount(), g.EdgeCount())
	}
	for x := 0; x < 10; x++ {
		if len(g.OutNeighbors(x)) != 0 || g.OutDegree(x) != 0 {
			t.Fatalf("node %d not empty", x)
		}
	}
}

// TestCorruptionRejected flips a header byte and truncates the region.
func TestCorruptionRejected(t *testing.T) {
	b := NewBuilder(20)
	for i := 0; i < 19; i++ {
		b.AddEdge(i, i+1)
	}
	good := b.Build()

	bad := append([]byte(nil), good...)
	bad[9] ^= 0xff // node count, inside header CRC
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	if _, err := Open(good[:headerLen-1]); err == nil {
		t.Fatal("short region accepted")
	}
}
