package rank

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// sampleEnsemble builds a tiny two-leaf ensemble for the io tests.
func sampleEnsemble() *Ensemble {
	root := newSplit(0, 0.5, newLeaf(-1), newLeaf(1))
	return &Ensemble{trees: []*treeNode{root}, numFeatures: 4}
}

// TestModelStampRoundTrips checks Save then LoadEnsemble preserves the feature-schema
// stamp, so a serving node reads the schema the model was trained against.
func TestModelStampRoundTrips(t *testing.T) {
	e := sampleEnsemble()
	e.SetSchema(3, 0xdeadbeefcafef00d)

	var buf bytes.Buffer
	if err := e.Save(&buf); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := LoadEnsemble(&buf)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.SchemaVersion() != 3 || got.SchemaHash() != 0xdeadbeefcafef00d {
		t.Fatalf("stamp = v%d hash %016x want v3 hash deadbeefcafef00d", got.SchemaVersion(), got.SchemaHash())
	}
	if got.NumTrees() != 1 {
		t.Fatalf("tree count %d want 1", got.NumTrees())
	}
}

// TestCompilePropagatesStamp checks the compiled served model carries the ensemble's
// schema stamp, the value the broker's CheckModel reads.
func TestCompilePropagatesStamp(t *testing.T) {
	e := sampleEnsemble()
	e.SetSchema(2, 0x1122334455667788)
	m := e.Compile()
	if m.SchemaVersion() != 2 || m.SchemaHash() != 0x1122334455667788 {
		t.Fatalf("model stamp = v%d hash %016x want v2 hash 1122334455667788", m.SchemaVersion(), m.SchemaHash())
	}
	if m.NumFeatures() != 4 {
		t.Fatalf("num features %d want 4", m.NumFeatures())
	}
}

// TestUnstampedModelLoads checks an ensemble saved without a stamp round-trips with a
// zero schema, the unstamped sentinel the loader allows.
func TestUnstampedModelLoads(t *testing.T) {
	var buf bytes.Buffer
	if err := sampleEnsemble().Save(&buf); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := LoadEnsemble(&buf)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.SchemaHash() != 0 || got.SchemaVersion() != 0 {
		t.Fatalf("unstamped model carries a stamp v%d hash %016x", got.SchemaVersion(), got.SchemaHash())
	}
}

// TestLegacyV1ModelReadsUnstamped checks a version-1 TRNK stream, which carries no
// schema block, still loads and reports a zero stamp, so old artifacts keep working.
func TestLegacyV1ModelReadsUnstamped(t *testing.T) {
	// Hand-assemble a v1 model: magic, a 12-byte header with version 1, then one
	// leaf tree, with no schema block following the header.
	var buf bytes.Buffer
	buf.Write(modelMagic[:])
	var hdr [12]byte
	hdr[0] = 1
	binary.LittleEndian.PutUint32(hdr[4:], 4) // numFeatures
	binary.LittleEndian.PutUint32(hdr[8:], 1) // numTrees
	buf.Write(hdr[:])
	buf.WriteByte(0) // leaf tag
	var leaf [8]byte
	binary.LittleEndian.PutUint64(leaf[:], 0x3ff0000000000000) // float64(1.0)
	buf.Write(leaf[:])

	got, err := LoadEnsemble(&buf)
	if err != nil {
		t.Fatalf("load v1: %v", err)
	}
	if got.SchemaHash() != 0 || got.NumTrees() != 1 {
		t.Fatalf("v1 load = hash %016x trees %d want hash 0 trees 1", got.SchemaHash(), got.NumTrees())
	}
}
