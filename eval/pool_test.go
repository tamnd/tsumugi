package eval

import (
	"context"
	"reflect"
	"testing"
)

func TestPoolUnionToDepth(t *testing.T) {
	runA := Run{"q1": {{Doc: "a", Score: 3}, {Doc: "b", Score: 2}, {Doc: "c", Score: 1}}}
	runB := Run{"q1": {{Doc: "b", Score: 3}, {Doc: "d", Score: 2}, {Doc: "e", Score: 1}}}
	// Depth 2 takes the top two of each run by score: {a,b} and {b,d}, union {a,b,d}.
	got := Pool([]Run{runA, runB}, 2)
	want := []string{"a", "b", "d"}
	if !reflect.DeepEqual(got["q1"], want) {
		t.Fatalf("pool = %v, want %v", got["q1"], want)
	}
}

func TestPoolDeterministicOrder(t *testing.T) {
	run := Run{"q": {{Doc: "z", Score: 1}, {Doc: "a", Score: 2}, {Doc: "m", Score: 3}}}
	first := Pool([]Run{run}, 10)["q"]
	for i := 0; i < 5; i++ {
		got := Pool([]Run{run}, 10)["q"]
		if !reflect.DeepEqual(got, first) {
			t.Fatalf("pool order not deterministic: %v then %v", first, got)
		}
	}
	want := []string{"a", "m", "z"}
	if !reflect.DeepEqual(first, want) {
		t.Fatalf("pool = %v, want sorted %v", first, want)
	}
}

func TestJudgePoolProducesQrels(t *testing.T) {
	queries := map[string]string{"q1": "rust async runtime"}
	pool := map[string][]string{"q1": {"d1", "d2", "d3"}}
	passages := map[string]Passage{
		"d1": {Title: "Rust async runtime internals", Body: "deep dive"},         // all terms in title: perfect
		"d2": {Title: "Concurrency", Body: "the rust async runtime in detail x"}, // all in body: highly
		"d3": {Title: "Banana bread", Body: "flour and sugar"},                   // none: irrelevant
	}
	passage := func(doc string) (Passage, bool) {
		p, ok := passages[doc]
		return p, ok
	}
	qrels, err := JudgePool(context.Background(), LexicalJudge{}, queries, pool, passage)
	if err != nil {
		t.Fatalf("JudgePool: %v", err)
	}
	want := map[string]float64{"d1": 3, "d2": 2, "d3": 0}
	if !reflect.DeepEqual(qrels["q1"], want) {
		t.Fatalf("qrels = %v, want %v", qrels["q1"], want)
	}
}

func TestJudgePoolSkipsMissingPassage(t *testing.T) {
	queries := map[string]string{"q1": "alpha beta"}
	pool := map[string][]string{"q1": {"present", "gone"}}
	passage := func(doc string) (Passage, bool) {
		if doc == "gone" {
			return Passage{}, false
		}
		return Passage{Title: "alpha beta", Body: "both"}, true
	}
	qrels, err := JudgePool(context.Background(), LexicalJudge{}, queries, pool, passage)
	if err != nil {
		t.Fatalf("JudgePool: %v", err)
	}
	if _, ok := qrels["q1"]["gone"]; ok {
		t.Fatal("a missing passage must be skipped, not judged on empty text")
	}
	if _, ok := qrels["q1"]["present"]; !ok {
		t.Fatal("the present document must be judged")
	}
}

func TestMergeQrelsGoldWins(t *testing.T) {
	base := Qrels{"q": {"d1": 1, "d2": 2}}
	gold := Qrels{"q": {"d2": 3}, "q2": {"x": 1}}
	got := MergeQrels(base, gold)
	if got["q"]["d2"] != 3 {
		t.Errorf("gold should override bulk: got %v", got["q"]["d2"])
	}
	if got["q"]["d1"] != 1 {
		t.Errorf("non-overlapping bulk label should survive: got %v", got["q"]["d1"])
	}
	if got["q2"]["x"] != 1 {
		t.Errorf("gold-only query should be added: got %v", got["q2"]["x"])
	}
	// base must not be mutated.
	if _, ok := base["q2"]; ok {
		t.Error("MergeQrels mutated base")
	}
	if base["q"]["d2"] != 2 {
		t.Error("MergeQrels mutated a base grade")
	}
}
