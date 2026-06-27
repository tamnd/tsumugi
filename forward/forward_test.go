package forward

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func schema() []Column {
	return []Column{
		{Name: "url", Type: ColString, Codec: CodecZstd},
		{Name: "title", Type: ColString, Codec: CodecZstd},
		{Name: "snippet", Type: ColString, Codec: CodecZstd},
		{Name: "body", Type: ColString, Codec: CodecZstdDict, Flags: FlagBlob},
	}
}

// TestRoundTrip builds a small store and reads every value back, including a
// document that leaves some columns unset so the empty-value path is exercised.
func TestRoundTrip(t *testing.T) {
	b := NewBuilder(schema())
	b.Set(0, "url", []byte("https://a.example/one"))
	b.Set(0, "title", []byte("First"))
	b.Set(0, "snippet", []byte("the first page"))
	b.Set(0, "body", []byte("the first page has a longer body than the snippet"))
	b.Set(2, "url", []byte("https://c.example/three"))
	b.Set(2, "title", []byte("Third"))
	// docID 1 is left entirely unset; docID 2 leaves snippet and body unset.

	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if r.DocCount() != 3 {
		t.Fatalf("doc count: got %d want 3", r.DocCount())
	}

	got, ok := r.Column("title", 0)
	if !ok || string(got) != "First" {
		t.Fatalf("title[0]: %q ok=%v", got, ok)
	}
	got, ok = r.Column("body", 0)
	if !ok || string(got) != "the first page has a longer body than the snippet" {
		t.Fatalf("body[0]: %q ok=%v", got, ok)
	}
	// An unset value comes back empty and present.
	got, ok = r.Column("snippet", 2)
	if !ok || len(got) != 0 {
		t.Fatalf("snippet[2]: %q ok=%v want empty", got, ok)
	}
	got, ok = r.Column("url", 1)
	if !ok || len(got) != 0 {
		t.Fatalf("url[1]: %q ok=%v want empty", got, ok)
	}
	got, ok = r.Column("url", 2)
	if !ok || string(got) != "https://c.example/three" {
		t.Fatalf("url[2]: %q ok=%v", got, ok)
	}
}

// TestUnknownColumnAndRange covers the false-returning paths: an unknown column
// and an out-of-range docID.
func TestUnknownColumnAndRange(t *testing.T) {
	b := NewBuilder(schema())
	b.Set(0, "url", []byte("x"))
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, ok := r.Column("nope", 0); ok {
		t.Fatal("unknown column reported present")
	}
	if _, ok := r.Column("url", 99); ok {
		t.Fatal("out-of-range docID reported present")
	}
	if _, ok := r.Row(99); ok {
		t.Fatal("out-of-range row reported present")
	}
	// Setting an undeclared column is a no-op, not a panic.
	b.Set(0, "ghost", []byte("y"))
}

// TestSchemaPreserved checks the column descriptors survive the round trip.
func TestSchemaPreserved(t *testing.T) {
	in := schema()
	r, err := Open(NewBuilder(in).Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if !reflect.DeepEqual(r.Schema(), in) {
		t.Fatalf("schema\n got=%+v\nwant=%+v", r.Schema(), in)
	}
}

// TestRowMatchesColumns confirms Row returns the same values as per-column reads.
func TestRowMatchesColumns(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cols := schema()
	b := NewBuilder(cols)
	const n = 200
	want := make([]map[string]string, n)
	for d := 0; d < n; d++ {
		want[d] = map[string]string{}
		for _, c := range cols {
			if rng.Intn(3) == 0 {
				continue // leave some unset
			}
			v := fmt.Sprintf("%s-%d-%d", c.Name, d, rng.Intn(1<<20))
			b.Set(uint32(d), c.Name, []byte(v))
			want[d][c.Name] = v
		}
	}
	r, err := Open(b.Build())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for d := 0; d < n; d++ {
		row, ok := r.Row(uint32(d))
		if !ok {
			t.Fatalf("row %d missing", d)
		}
		for _, c := range cols {
			cv, _ := r.Column(c.Name, uint32(d))
			if string(row[c.Name]) != string(cv) {
				t.Fatalf("row vs column mismatch d=%d col=%s", d, c.Name)
			}
			if string(cv) != want[d][c.Name] {
				t.Fatalf("value mismatch d=%d col=%s got=%q want=%q", d, c.Name, cv, want[d][c.Name])
			}
		}
	}
}

// TestCorruptionRejected flips the header CRC region and a truncation and checks
// Open rejects both.
func TestCorruptionRejected(t *testing.T) {
	b := NewBuilder(schema())
	b.Set(0, "url", []byte("https://a.example"))
	good := b.Build()

	bad := append([]byte(nil), good...)
	bad[12] ^= 0xff // flip a row-count byte, inside the header CRC cover
	if _, err := Open(bad); err == nil {
		t.Fatal("corrupt header accepted")
	}
	if _, err := Open(good[:8]); err == nil {
		t.Fatal("truncated region accepted")
	}
}

// TestEmpty builds a store with the schema but no documents.
func TestEmpty(t *testing.T) {
	r, err := Open(NewBuilder(schema()).Build())
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	if r.DocCount() != 0 {
		t.Fatalf("empty doc count %d", r.DocCount())
	}
	if _, ok := r.Column("url", 0); ok {
		t.Fatal("empty store returned a value")
	}
}
