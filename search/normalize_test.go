package search

import "testing"

// TestOnlineBM25NormalizesTitle proves the title field is length-normalized against
// its own fleet average, the per-field normalization gap #2 closes. Before per-field
// averages reached the extractor the title carried no average, so its BM25 used norm
// = 1 and a long title scored a term exactly as a short one. With a fleet title
// average a title longer than the average is normalized down, the standard BM25
// length penalty, now applied per field rather than only to the body.
func TestOnlineBM25NormalizesTitle(t *testing.T) {
	docs := []textDoc{{title: "apple filler filler filler filler"}}
	q := Query{Text: "apple"}
	idf := map[string]float64{"apple": 1.0}

	fwd := buildForward(t, docs)
	// avgField title set below the field's length, so the field is longer than the
	// fleet average and normalization must pull its score down.
	normalized := newOnlineExtractor(q, fwd, nil, idf, [3]float64{fTitle: 2}).features(0)
	// avgField title zero, the old no-normalization behavior, the baseline to beat.
	unnormalized := newOnlineExtractor(q, fwd, nil, idf, [3]float64{}).features(0)

	if normalized[OnBM25Title] <= 0 {
		t.Fatalf("title bm25 of a present term should be positive, got %.4f", normalized[OnBM25Title])
	}
	if normalized[OnBM25Title] >= unnormalized[OnBM25Title] {
		t.Fatalf("a title longer than the fleet average should normalize down: normalized %.4f, unnormalized %.4f",
			normalized[OnBM25Title], unnormalized[OnBM25Title])
	}
}

// TestOnlineBM25NormalizesURL is the same proof for the url-token field: a url longer
// than its fleet average is normalized down, where before it carried no average and
// scored unnormalized.
func TestOnlineBM25NormalizesURL(t *testing.T) {
	docs := []textDoc{{url: "https://example.com/apple/section/page/deep/path"}}
	q := Query{Text: "apple"}
	idf := map[string]float64{"apple": 1.0}

	fwd := buildForward(t, docs)
	normalized := newOnlineExtractor(q, fwd, nil, idf, [3]float64{fURL: 2}).features(0)
	unnormalized := newOnlineExtractor(q, fwd, nil, idf, [3]float64{}).features(0)

	if normalized[OnBM25URL] <= 0 {
		t.Fatalf("url bm25 of a present term should be positive, got %.4f", normalized[OnBM25URL])
	}
	if normalized[OnBM25URL] >= unnormalized[OnBM25URL] {
		t.Fatalf("a url longer than the fleet average should normalize down: normalized %.4f, unnormalized %.4f",
			normalized[OnBM25URL], unnormalized[OnBM25URL])
	}
}

// TestOnlineBM25FUsesPerFieldAverages proves the field-weighted total normalizes each
// field by its own fleet average rather than one shared denominator. With every field
// average set, a candidate whose title is longer than the fleet title average has its
// title contribution normalized by the title average, not by the body's, so two
// documents with identical body text but title lengths on opposite sides of the fleet
// average score differently on the title-weighted total.
func TestOnlineBM25FUsesPerFieldAverages(t *testing.T) {
	docs := []textDoc{
		{title: "apple", body: "apple banana cherry"},
		{title: "apple filler filler filler filler", body: "apple banana cherry"},
	}
	q := Query{Text: "apple"}
	idf := map[string]float64{"apple": 1.0}
	avg := [3]float64{fTitle: 2, fBody: 3, fURL: 1}

	fwd := buildForward(t, docs)
	e := newOnlineExtractor(q, fwd, nil, idf, avg)
	short := e.features(0)
	long := e.features(1)
	if short[OnBM25FTotal] <= long[OnBM25FTotal] {
		t.Fatalf("the shorter title should win the field-weighted total: short %.4f, long %.4f",
			short[OnBM25FTotal], long[OnBM25FTotal])
	}
}
