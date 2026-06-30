package eval

import (
	"testing"
)

func TestKendallTauB(t *testing.T) {
	cases := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"perfect agreement", []float64{1, 2, 3, 4, 5}, []float64{1, 2, 3, 4, 5}, 1},
		{"perfect inversion", []float64{1, 2, 3, 4, 5}, []float64{5, 4, 3, 2, 1}, -1},
		{"one discordant pair", []float64{1, 2, 3, 4, 5}, []float64{1, 2, 3, 5, 4}, 0.8},
		{"no variation is zero", []float64{2, 2, 2, 2}, []float64{1, 2, 3, 4}, 0},
		{"too few points is zero", []float64{1}, []float64{1}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := KendallTauB(c.a, c.b)
			if !approx(got, c.want) {
				t.Fatalf("KendallTauB = %.6f, want %.6f", got, c.want)
			}
		})
	}
}

func TestKendallTauBHandlesTies(t *testing.T) {
	// Heavy ties on the 0-3 grade scale: tau-b must stay in range and reward agreement.
	a := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	b := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	if got := KendallTauB(a, b); !approx(got, 1) {
		t.Fatalf("identical tied sequences should give tau-b 1, got %.6f", got)
	}
	// A judge that swaps one adjacent grade pair drops below 1 but stays high.
	c := []float64{0, 0, 1, 1, 2, 3, 2, 3}
	got := KendallTauB(a, c)
	if got <= 0 || got >= 1 {
		t.Fatalf("a single adjacent swap should give 0 < tau-b < 1, got %.6f", got)
	}
}

func TestCompareToGold(t *testing.T) {
	// The judge agrees with the human on most pairs and confuses one adjacent grade.
	gold := Qrels{
		"q1": {"d1": 3, "d2": 0},
		"q2": {"d3": 2, "d4": 1},
	}
	judge := Qrels{
		"q1": {"d1": 3, "d2": 0, "extra": 1}, // extra is not in gold, ignored
		"q2": {"d3": 3, "d4": 1},             // d3 off by one
	}
	ag := CompareToGold(judge, gold)
	if ag.N != 4 {
		t.Fatalf("compared %d pairs, want 4 (the gold pairs)", ag.N)
	}
	if ag.Missing != 0 {
		t.Fatalf("missing = %d, want 0; the judge labeled every gold pair", ag.Missing)
	}
	// Confusion is [human][judge]: human 3/judge 3, human 0/judge 0, human 2/judge 3, human 1/judge 1.
	if ag.Confusion[3][3] != 1 || ag.Confusion[0][0] != 1 || ag.Confusion[2][3] != 1 || ag.Confusion[1][1] != 1 {
		t.Fatalf("confusion matrix wrong: %v", ag.Confusion)
	}
}

func TestCompareToGoldCountsMissing(t *testing.T) {
	gold := Qrels{"q": {"d1": 3, "d2": 2}}
	judge := Qrels{"q": {"d1": 3}} // d2 unlabeled
	ag := CompareToGold(judge, gold)
	if ag.N != 1 {
		t.Fatalf("compared %d, want 1", ag.N)
	}
	if ag.Missing != 1 {
		t.Fatalf("missing = %d, want 1", ag.Missing)
	}
}

func TestAgreementGate(t *testing.T) {
	// A judge that reproduces the human ranking clears the UMBRELA gate.
	gold := Qrels{"q": {}}
	judge := Qrels{"q": {}}
	for i := 0; i < 40; i++ {
		doc := string(rune('a'+i%26)) + string(rune('0'+i/26))
		grade := float64(i % 4)
		gold["q"][doc] = grade
		judge["q"][doc] = grade // perfect agreement
	}
	ag := CompareToGold(judge, gold)
	if !ag.Passes() {
		t.Fatalf("perfect agreement (tau %.4f) must pass the %.2f gate", ag.KendallTau, UmbrelaTauGate)
	}
	// A judge that grades at random fails the gate.
	bad := Qrels{"q": {}}
	for doc := range gold["q"] {
		bad["q"][doc] = float64((int(gold["q"][doc]) + 2) % 4) // shift every grade, breaks the order
	}
	if CompareToGold(bad, gold).Passes() {
		t.Fatal("a judge that distorts every grade must not pass the trust gate")
	}
}
