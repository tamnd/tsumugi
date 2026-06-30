package search

import (
	"context"
	"math"
	"reflect"
	"testing"
)

// sampleQueries returns the spread of Query shapes the wire codec must round-trip, the fields a
// distributed query actually carries: a bare term query, one with the pushed-down idf and field
// averages an aggregator sends a child, one with a dense vector and a sparse map, and the empty
// query. They exercise every nilable field in both its nil and its populated state, the
// distinction the codec preserves.
func sampleQueries() []Query {
	avg := [3]float64{12.5, 480.25, 7.0}
	return []Query{
		{},
		{Text: "raw string path", K: 10},
		{Terms: []string{"alpha", "beta", "common"}, K: 20, L0: 400},
		{Terms: []string{}, K: 5},
		{
			Terms:       []string{"data", "page"},
			TermIDF:     map[string]float64{"data": 3.5, "page": 1.25},
			AvgFieldLen: &avg,
			K:           15,
		},
		{
			Text:   "dense",
			Vector: []float32{0.1, -0.2, 3.5, 0, math.MaxFloat32, -math.SmallestNonzeroFloat32},
			Sparse: map[string]int{"x": 7, "y": -3, "z": 0},
			K:      25,
		},
		{Vector: []float32{}, Sparse: map[string]int{}},
	}
}

// sampleResults returns the Results shapes the codec must round-trip: a normal ranked top-k, a
// dropped-subtree result with a nil hit slice and the shard count charged, a degraded result,
// and an empty-but-non-nil hit slice, so the nil-versus-empty distinction is covered on the
// response side too.
func sampleResults() []Results {
	return []Results{
		{},
		{Hits: []Hit{{DocID: 42, Score: 1.5}, {DocID: 7, Score: 0.25}, {DocID: 99999, Score: -3.75}}, ShardsTotal: 8, ShardsOK: 8},
		{ShardsTotal: 4, ShardsOK: 0},
		{Hits: []Hit{{DocID: 1, Score: math.MaxFloat64}}, ShardsTotal: 3, ShardsOK: 2, Degraded: DegradeL2},
		{Hits: []Hit{}, ShardsTotal: 1, ShardsOK: 1},
	}
}

// TestBinaryCodecQueryRoundTrip checks the binary codec decodes a query to exactly the value it
// encoded, every field including the nil-versus-empty distinction reflect.DeepEqual is sensitive
// to, so the dense wire carries a query an aggregator pushes down without losing or changing a
// field. The float fields round-trip bit-for-bit because the codec writes their IEEE-754 bytes.
func TestBinaryCodecQueryRoundTrip(t *testing.T) {
	c := binaryCodec{}
	for i, q := range sampleQueries() {
		b, err := c.encodeQuery(q)
		if err != nil {
			t.Fatalf("query %d: encode: %v", i, err)
		}
		got, err := c.decodeQuery(b)
		if err != nil {
			t.Fatalf("query %d: decode: %v", i, err)
		}
		if !reflect.DeepEqual(got, q) {
			t.Fatalf("query %d round-trip mismatch:\n got  %#v\n want %#v", i, got, q)
		}
	}
}

// TestBinaryCodecResultsRoundTrip is the response-side counterpart: the binary codec decodes a
// Results to exactly the value it encoded, the hits in order with scores equal to the float, the
// shard counts and the degradation rung intact, and a nil hit slice still nil. The score
// equality is exact, not within a tolerance, because the merge up the tree compares scores and a
// codec that rounded a score would break it.
func TestBinaryCodecResultsRoundTrip(t *testing.T) {
	c := binaryCodec{}
	for i, r := range sampleResults() {
		b, err := c.encodeResults(r)
		if err != nil {
			t.Fatalf("results %d: encode: %v", i, err)
		}
		got, err := c.decodeResults(b)
		if err != nil {
			t.Fatalf("results %d: decode: %v", i, err)
		}
		if !reflect.DeepEqual(got, r) {
			t.Fatalf("results %d round-trip mismatch:\n got  %#v\n want %#v", i, got, r)
		}
	}
}

// TestBinaryCodecMapOrderDeterministic checks the encoding of a query with maps is byte-identical
// across encodes, the deterministic-wire property: Go iterates a map in random order, so a codec
// that wrote the map in iteration order would emit different bytes for the same query, but the
// codec sorts the keys, so the bytes are stable. This is the build's byte-identical property
// carried onto the wire.
func TestBinaryCodecMapOrderDeterministic(t *testing.T) {
	c := binaryCodec{}
	q := Query{
		Terms:   []string{"a", "b"},
		Sparse:  map[string]int{"q": 1, "w": 2, "e": 3, "r": 4, "t": 5, "y": 6},
		TermIDF: map[string]float64{"q": 1.1, "w": 2.2, "e": 3.3, "r": 4.4, "t": 5.5, "y": 6.6},
	}
	first, err := c.encodeQuery(q)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	for i := 0; i < 20; i++ {
		b, err := c.encodeQuery(q)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if !reflect.DeepEqual(b, first) {
			t.Fatalf("encode %d differs from first: map iteration order leaked onto the wire", i)
		}
	}
}

// TestBinaryCodecRejectsTruncated checks the decoder fails rather than panics on a truncated
// message: every prefix of a real encoded query and result is fed back through decode, and each
// either decodes (a prefix that happens to end on a field boundary) or returns an error, but
// never panics. A wire decoder reads bytes off the network, so a short or corrupt body must be a
// clean error, not a crash.
func TestBinaryCodecRejectsTruncated(t *testing.T) {
	c := binaryCodec{}
	q := Query{Terms: []string{"alpha", "beta"}, Vector: []float32{1, 2, 3}, TermIDF: map[string]float64{"alpha": 2.5}, K: 10}
	qb, _ := c.encodeQuery(q)
	for n := 0; n < len(qb); n++ {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("decodeQuery panicked on a %d-byte prefix: %v", n, p)
				}
			}()
			_, _ = c.decodeQuery(qb[:n])
		}()
	}

	r := Results{Hits: []Hit{{DocID: 1, Score: 1.5}, {DocID: 2, Score: 2.5}}, ShardsTotal: 4, ShardsOK: 4}
	rb, _ := c.encodeResults(r)
	for n := 0; n < len(rb); n++ {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("decodeResults panicked on a %d-byte prefix: %v", n, p)
				}
			}()
			_, _ = c.decodeResults(rb[:n])
		}()
	}
}

// TestBinaryWireSmallerThanJSON measures the point of the codec: on a query carrying a dense
// vector and on a result carrying a ranked top-k, the binary form is smaller than the JSON form,
// because a float32 is four bytes instead of a dozen decimal characters and a score is eight
// bytes instead of a long decimal. It logs both sizes so the ratio is visible, and fails only if
// the dense form is not actually denser, the regression guard on the codec's reason to exist.
func TestBinaryWireSmallerThanJSON(t *testing.T) {
	bin, js := binaryCodec{}, jsonCodec{}

	// A query with a 256-dim dense vector and the pushed-down idf, the shape a head sends a leaf
	// in a vector-enabled fleet, where JSON's decimal-text floats are most of the bytes.
	vec := make([]float32, 256)
	idf := map[string]float64{}
	for i := range vec {
		vec[i] = float32(i)*0.013 - 1.7
	}
	for _, term := range []string{"alpha", "beta", "gamma", "delta", "common"} {
		idf[term] = 3.14159 + float64(len(term))
	}
	q := Query{Terms: []string{"alpha", "beta", "gamma", "delta", "common"}, TermIDF: idf, Vector: vec, K: 100}
	qb, _ := bin.encodeQuery(q)
	qj, _ := js.encodeQuery(q)
	t.Logf("query (256-d vector): binary %d B, json %d B (%.2fx)", len(qb), len(qj), float64(len(qj))/float64(len(qb)))
	if len(qb) >= len(qj) {
		t.Fatalf("binary query (%d B) is not smaller than json (%d B)", len(qb), len(qj))
	}

	// A 1000-hit result, the deep top-k an aggregator pulls back from each leaf, where JSON writes
	// every document id and score as long decimal text.
	hits := make([]Hit, 1000)
	for i := range hits {
		hits[i] = Hit{DocID: uint32(1_000_000 + i), Score: 12.3456789 - float64(i)*0.001}
	}
	r := Results{Hits: hits, ShardsTotal: 100, ShardsOK: 100}
	rb, _ := bin.encodeResults(r)
	rj, _ := js.encodeResults(r)
	t.Logf("results (1000 hits): binary %d B, json %d B (%.2fx)", len(rb), len(rj), float64(len(rj))/float64(len(rb)))
	if len(rb) >= len(rj) {
		t.Fatalf("binary results (%d B) is not smaller than json (%d B)", len(rb), len(rj))
	}
}

// TestRemoteSearcherBinaryWireMatchesLocal is the dense-wire exactness proof: a RemoteSearcher
// built WithBinaryWire over an httptest server reproduces the in-process broker's own answer
// exactly, the same hits in the same order with scores equal to the float, the same way the JSON
// wire does in TestRemoteSearcherMatchesLocal. It runs over a real HTTP round trip, so it
// exercises the binary encode, the transport, the handler's content-type detection, and the
// binary decode, not a mock. The corpus is the cross-broker rank corpus so the top-k is a genuine
// score-ordered list, not a flat one a lossy codec could fake.
func TestRemoteSearcherBinaryWireMatchesLocal(t *testing.T) {
	const n, parts = 200, 4
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	rs := serveSearcher(t, broker, WithBinaryWire())

	ctx := context.Background()
	nontrivial := 0
	for _, q := range []Query{
		{Terms: []string{"common"}, K: 10},
		{Terms: []string{"alpha", "common"}, K: 20},
		{Terms: []string{"beta", "common"}, K: 15},
		{Terms: []string{"alpha", "beta", "common"}, K: 25},
	} {
		want := broker.SearchComplete(ctx, q)
		got := rs.SearchComplete(ctx, q)
		if got.ShardsTotal != want.ShardsTotal || got.ShardsOK != want.ShardsOK {
			t.Fatalf("query %v completeness %d/%d over binary wire, local %d/%d", q.Terms, got.ShardsOK, got.ShardsTotal, want.ShardsOK, want.ShardsTotal)
		}
		if got.Degraded != want.Degraded {
			t.Fatalf("query %v degraded %v over binary wire, local %v", q.Terms, got.Degraded, want.Degraded)
		}
		if len(got.Hits) != len(want.Hits) {
			t.Fatalf("query %v: binary wire returned %d hits, local %d", q.Terms, len(got.Hits), len(want.Hits))
		}
		for i := range want.Hits {
			if got.Hits[i].DocID != want.Hits[i].DocID || got.Hits[i].Score != want.Hits[i].Score {
				t.Fatalf("query %v hit %d: binary {%d,%v}, local {%d,%v}", q.Terms, i,
					got.Hits[i].DocID, got.Hits[i].Score, want.Hits[i].DocID, want.Hits[i].Score)
			}
		}
		if len(got.Hits) > 1 && got.Hits[0].Score != got.Hits[len(got.Hits)-1].Score {
			nontrivial++
		}
	}
	if nontrivial == 0 {
		t.Fatal("every query produced a flat ranking; the binary wire was not exercised on a real ranked top-k")
	}
}

// TestWireCodecsInteroperateOnOneHandler checks one handler answers a JSON client and a binary
// client off the same broker, each in its own wire: the handler reads the request's content type,
// so a JSON RemoteSearcher and a binary one dialed at one server both get the broker's exact
// answer. This is the back-compatibility guarantee, that turning the binary codec on at one peer
// does not require turning it on everywhere, and the two clients see identical results.
func TestWireCodecsInteroperateOnOneHandler(t *testing.T) {
	const n, parts = 160, 4
	docs := remoteRankCorpus(n)
	dir := t.TempDir()
	model := trainModel(t)

	broker, shards := buildBrokerFromDocs(t, dir, "s", docs, parts, model)
	defer func() {
		for _, sh := range shards {
			_ = sh.Close()
		}
	}()

	jsonRS := serveSearcher(t, broker)
	binRS := serveSearcher(t, broker, WithBinaryWire())

	ctx := context.Background()
	q := Query{Terms: []string{"alpha", "beta", "common"}, K: 20}
	jr := jsonRS.SearchComplete(ctx, q)
	br := binRS.SearchComplete(ctx, q)
	if len(jr.Hits) != len(br.Hits) || len(jr.Hits) == 0 {
		t.Fatalf("json client returned %d hits, binary client %d", len(jr.Hits), len(br.Hits))
	}
	for i := range jr.Hits {
		if jr.Hits[i].DocID != br.Hits[i].DocID || jr.Hits[i].Score != br.Hits[i].Score {
			t.Fatalf("hit %d: json {%d,%v}, binary {%d,%v}", i,
				jr.Hits[i].DocID, jr.Hits[i].Score, br.Hits[i].DocID, br.Hits[i].Score)
		}
	}
}
