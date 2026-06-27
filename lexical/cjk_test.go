package lexical

import (
	"reflect"
	"testing"
)

// A run made entirely of dictionary words splits into exactly those words: the forward
// maximum matcher takes the longest word at each position, so "中国" is one term, not two
// characters. This is the precision the dictionary buys.
func TestSegmentDictionaryWords(t *testing.T) {
	s := newCJKSegmenter([]string{"中国", "公司", "产品"})
	got := s.segment([]rune("中国公司"))
	want := []string{"中国", "公司"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

// Maximum matching prefers the longer word: with both "北京" and "北京大学" in the
// dictionary, "北京大学" must come out whole rather than as "北京" plus a fallback, since
// the scan starts from the longest candidate.
func TestSegmentPrefersLongest(t *testing.T) {
	s := newCJKSegmenter([]string{"北京", "大学", "北京大学"})
	got := s.segment([]rune("北京大学"))
	want := []string{"北京大学"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

// A run the dictionary does not cover falls back to overlapping character bigrams, the
// recall path: three unknown characters yield two overlapping bigrams so a two-character
// query against the middle of the run still matches.
func TestSegmentBigramFallback(t *testing.T) {
	s := newCJKSegmenter(nil)
	got := s.segment([]rune("甲乙丙"))
	want := []string{"甲乙", "乙丙"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

// A single uncovered character with nothing after it is emitted alone rather than
// dropped, so a one-character run still produces a term.
func TestSegmentLoneFinalChar(t *testing.T) {
	s := newCJKSegmenter([]string{"中国"})
	got := s.segment([]rune("中国人"))
	want := []string{"中国", "人"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

// split keeps the Latin and digit spans of a mixed run as single tokens and segments
// only the CJK spans, so "iphone手機13" yields the Latin token, the segmented Han, and
// the digits, the case a real product page carries.
func TestSplitMixedRun(t *testing.T) {
	s := newCJKSegmenter([]string{"手機"})
	got := s.split("iphone手機13")
	want := []string{"iphone", "手機", "13"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("split = %v, want %v", got, want)
	}
}

// Korean is space-separated, so its runes are not CJK runes for segmentation and split
// leaves a Hangul run whole, the run tokenizer's job.
func TestSplitLeavesHangul(t *testing.T) {
	if IsCJKRune('한') {
		t.Error("Hangul must not be a CJK segmentation rune")
	}
}

// Segmentation is deterministic: the same run must split the same way every call, since
// the dictionary set is iterated only at fixed lengths, never by map order.
func TestSegmentDeterministic(t *testing.T) {
	s := newCJKSegmenter([]string{"搜索", "引擎", "结果"})
	run := []rune("搜索引擎返回最佳结果")
	first := s.segment(run)
	for i := 0; i < 100; i++ {
		got := s.segment(run)
		if !reflect.DeepEqual(got, first) {
			t.Fatalf("segment not deterministic: %v vs %v", got, first)
		}
	}
}

// The segmenting analyzer splits a CJK sentence into multiple terms where the plain
// analyzer would emit one, the whole point of the slice. The plain analyzer is the
// baseline: one run, one term.
func TestAnalyzerSegmentsCJK(t *testing.T) {
	plain := (&Analyzer{NFKC: true}).Analyze("搜索引擎返回最佳结果")
	if len(plain) != 1 {
		t.Fatalf("plain analyzer on a CJK run = %v, want one term", plain)
	}
	seg := (&Analyzer{NFKC: true, Segment: true}).Analyze("搜索引擎返回最佳结果")
	if len(seg) <= 1 {
		t.Fatalf("segmenting analyzer = %v, want more than one term", seg)
	}
	for _, tok := range seg {
		if len([]rune(tok)) > sharedCJK.maxLen {
			t.Errorf("term %q longer than the longest dictionary word", tok)
		}
	}
}

// The build side and the query side must produce identical terms for the same CJK text,
// the property the whole shared-analyzer design exists to guarantee. A document field
// and a query analyzed by the same ForLanguage("zh") analyzer agree term for term.
func TestSegmentBuildQueryConsistent(t *testing.T) {
	a := ForLanguage("zh")
	text := "中国公司提供搜索引擎服务"
	build := a.Analyze(text)
	query := a.Analyze(text)
	if !reflect.DeepEqual(build, query) {
		t.Fatalf("build %v != query %v", build, query)
	}
	// And a query for a word that segments out of the document finds it among the
	// document's terms, the recall the segmentation buys.
	docTerms := map[string]struct{}{}
	for _, tok := range build {
		docTerms[tok] = struct{}{}
	}
	for _, q := range a.Analyze("搜索引擎") {
		if _, ok := docTerms[q]; !ok {
			t.Errorf("query term %q not found among document terms %v", q, build)
		}
	}
}

// Turning segmentation on changes the analyzer_hash, so a shard built with segmentation
// cannot be silently queried by an analyzer without it, and the dictionary's identity is
// folded in so a dictionary change is a hash change too.
func TestSegmentChangesHash(t *testing.T) {
	off := (&Analyzer{NFKC: true}).Hash()
	on := (&Analyzer{NFKC: true, Segment: true}).Hash()
	if off == on {
		t.Error("segmentation did not change the analyzer hash")
	}
}

func BenchmarkSegmentCJK(b *testing.B) {
	run := []rune("搜索引擎返回最佳结果中国公司提供专业服务北京上海广州深圳")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sharedCJK.segment(run)
	}
}
