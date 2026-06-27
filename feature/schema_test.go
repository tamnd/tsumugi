package feature

import "testing"

// TestSchemaHashStable pins that the default schema's fingerprint is reproducible
// across calls and that DefaultSchemaHash agrees with SchemaHash over DefaultSchema.
func TestSchemaHashStable(t *testing.T) {
	// Two independently built copies of the same layout must hash equal, the property
	// that lets a shard and a model compare fingerprints across processes.
	a := SchemaHash(append([]Column(nil), DefaultSchema()...))
	b := SchemaHash(append([]Column(nil), DefaultSchema()...))
	if a != b {
		t.Fatal("schema hash not reproducible")
	}
	if DefaultSchemaHash() != SchemaHash(DefaultSchema()) {
		t.Fatal("DefaultSchemaHash disagrees with SchemaHash(DefaultSchema())")
	}
	if DefaultSchemaHash() == 0 {
		t.Fatal("default schema hash is zero, the unstamped sentinel")
	}
}

// TestSchemaHashDetectsChange checks the fingerprint moves under every kind of layout
// change a model would misread: reordering columns, retyping one, changing a width,
// and adding a column.
func TestSchemaHashDetectsChange(t *testing.T) {
	base := DefaultSchema()
	want := SchemaHash(base)

	reordered := append([]Column(nil), base...)
	reordered[0], reordered[1] = reordered[1], reordered[0]
	if SchemaHash(reordered) == want {
		t.Fatal("reorder did not change the hash")
	}

	retyped := append([]Column(nil), base...)
	retyped[0].Quant = QuantLog
	if SchemaHash(retyped) == want {
		t.Fatal("quant change did not change the hash")
	}

	rewidth := append([]Column(nil), base...)
	rewidth[0].Width = 2
	if SchemaHash(rewidth) == want {
		t.Fatal("width change did not change the hash")
	}

	added := append(append([]Column(nil), base...), Column{FeatHostErrorRate, 1, QuantLinear})
	if SchemaHash(added) == want {
		t.Fatal("added column did not change the hash")
	}
}

// TestRegionColumnsRoundTrip checks a region reports the columns it was built with and
// that its SchemaHash matches the schema it was built from, the comparison the shard
// and broker loaders make.
func TestRegionColumnsRoundTrip(t *testing.T) {
	cols := DefaultSchema()
	r, err := Open(NewBuilder(cols, SchemaVersion).Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	got := r.Columns()
	if len(got) != len(cols) {
		t.Fatalf("column count %d want %d", len(got), len(cols))
	}
	for i := range cols {
		if got[i] != cols[i] {
			t.Fatalf("column %d = %+v want %+v", i, got[i], cols[i])
		}
	}
	if r.SchemaHash() != DefaultSchemaHash() {
		t.Fatalf("region schema hash %016x want %016x", r.SchemaHash(), DefaultSchemaHash())
	}
	if r.SchemaVersion() != SchemaVersion {
		t.Fatalf("region schema version %d want %d", r.SchemaVersion(), SchemaVersion)
	}
}

// TestRegionSchemaHashTracksLayout checks a region built from a non-default layout
// hashes to that layout, not the default, so a shard built against a different schema
// is distinguishable at load.
func TestRegionSchemaHashTracksLayout(t *testing.T) {
	cols := DefaultSchema()
	cols[0], cols[1] = cols[1], cols[0]
	r, err := Open(NewBuilder(cols, SchemaVersion).Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if r.SchemaHash() == DefaultSchemaHash() {
		t.Fatal("reordered region hashes equal to the default schema")
	}
	if r.SchemaHash() != SchemaHash(cols) {
		t.Fatal("region hash disagrees with its own layout")
	}
}
