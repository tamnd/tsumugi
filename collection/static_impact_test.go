package collection

import "testing"

// TestQuantizeImpactMonotone checks the quantizer preserves the static-rank order: a higher
// rank never maps to a lower byte, the property the Block-Max Pruning walk terminates on.
func TestQuantizeImpactMonotone(t *testing.T) {
	rank := []float64{-0.4, 0.0, 0.1, 0.1, 0.55, 1.0, 0.9}
	q := quantizeImpact(rank)
	if len(q) != len(rank) {
		t.Fatalf("length: got %d want %d", len(q), len(rank))
	}
	for i := range rank {
		for j := range rank {
			if rank[i] < rank[j] && q[i] > q[j] {
				t.Fatalf("order inverted: rank[%d]=%v<rank[%d]=%v but q=%d>%d", i, rank[i], j, rank[j], q[i], q[j])
			}
			if rank[i] == rank[j] && q[i] != q[j] {
				t.Fatalf("equal ranks quantized apart: rank=%v q[%d]=%d q[%d]=%d", rank[i], i, q[i], j, q[j])
			}
		}
	}
}

// TestQuantizeImpactRange checks the quantizer uses the whole byte: the corpus min maps to 0
// and the max to 255, so the impacts spread across the full resolution the codec stores.
func TestQuantizeImpactRange(t *testing.T) {
	rank := []float64{-0.5, 0.25, 1.0}
	q := quantizeImpact(rank)
	if q[0] != 0 {
		t.Fatalf("min did not map to 0: got %d", q[0])
	}
	if q[2] != 255 {
		t.Fatalf("max did not map to 255: got %d", q[2])
	}
	if q[1] == 0 || q[1] == 255 {
		t.Fatalf("mid value collapsed to an endpoint: got %d", q[1])
	}
}

// TestQuantizeImpactDegenerate checks a shard whose ranks are all equal maps every document
// to the top of the range, so query-term coverage still orders the results rather than every
// score collapsing to zero.
func TestQuantizeImpactDegenerate(t *testing.T) {
	q := quantizeImpact([]float64{0.3, 0.3, 0.3})
	for i, v := range q {
		if v != impactQuantScale {
			t.Fatalf("degenerate doc %d = %d, want %d", i, v, impactQuantScale)
		}
	}
	if len(quantizeImpact(nil)) != 0 {
		t.Fatalf("empty input should quantize to empty")
	}
}
