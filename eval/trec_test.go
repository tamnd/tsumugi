package eval

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func approxEq(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
}

// TestParseRunGroupsAndIgnoresRankTag checks the six-column parse groups by query
// and reads the score, the field that orders the list, ignoring the advisory rank
// and the run tag the way trec_eval does.
func TestParseRunGroupsAndIgnoresRankTag(t *testing.T) {
	in := "q1 Q0 docA 1 2.5 sys\nq1 Q0 docB 2 1.0 sys\nq2 Q0 docC 1 9.0 sys\n\n"
	run, err := ParseRun(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(run) != 2 || len(run["q1"]) != 2 || len(run["q2"]) != 1 {
		t.Fatalf("grouping wrong: %#v", run)
	}
	if run["q1"][0].Doc != "docA" || run["q1"][0].Score != 2.5 {
		t.Fatalf("q1 first = %#v", run["q1"][0])
	}
}

// TestParseRunRejectsMalformed checks a wrong column count and an unparseable score
// are errors, not silently dropped lines that would quietly lower every metric.
func TestParseRunRejectsMalformed(t *testing.T) {
	if _, err := ParseRun(strings.NewReader("q1 Q0 docA 1 2.5")); err == nil {
		t.Fatal("want error on five-field line")
	}
	if _, err := ParseRun(strings.NewReader("q1 Q0 docA 1 high sys")); err == nil {
		t.Fatal("want error on non-numeric score")
	}
}

// TestParseQrels checks the four-column parse and the last-grade-wins rule for a
// repeated pair.
func TestParseQrels(t *testing.T) {
	in := "q1 0 docA 3\nq1 0 docB 0\nq1 0 docA 2\nq2 0 docC 1\n"
	q, err := ParseQrels(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if q["q1"]["docA"] != 2 {
		t.Fatalf("repeated pair kept %v, want last 2", q["q1"]["docA"])
	}
	if q["q1"]["docB"] != 0 || q["q2"]["docC"] != 1 {
		t.Fatalf("qrels = %#v", q)
	}
}

// TestWriteRunRoundTrips checks WriteRun emits canonical six-column lines in sorted
// order that ParseRun reads back to the same scored order, the property a committed
// run file and the reproducibility check rest on.
func TestWriteRunRoundTrips(t *testing.T) {
	run := Run{
		"q2": {{Doc: "z", Score: 1}},
		"q1": {{Doc: "a", Score: 1.0}, {Doc: "b", Score: 3.0}},
	}
	var buf bytes.Buffer
	if err := WriteRun(&buf, run, "tag"); err != nil {
		t.Fatal(err)
	}
	want := "q1 Q0 b 1 3 tag\nq1 Q0 a 2 1 tag\nq2 Q0 z 1 1 tag\n"
	if buf.String() != want {
		t.Fatalf("WriteRun =\n%q\nwant\n%q", buf.String(), want)
	}
	back, err := ParseRun(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if back["q1"][0].Doc != "b" || sortedByScore(back["q1"])[0].Doc != "b" {
		t.Fatalf("round-trip order wrong: %#v", back["q1"])
	}
}

// TestScoreTieBreaksByDocDescending pins the trec_eval tie rule: equal-score
// documents order by descending identifier, so a metric over a run with ties is
// reproducible rather than dependent on the file's line order.
func TestScoreTieBreaksByDocDescending(t *testing.T) {
	docs := []RankedDoc{{Doc: "a", Score: 1}, {Doc: "c", Score: 1}, {Doc: "b", Score: 1}}
	got := sortedByScore(docs)
	if got[0].Doc != "c" || got[1].Doc != "b" || got[2].Doc != "a" {
		t.Fatalf("tie order = %v %v %v, want c b a", got[0].Doc, got[1].Doc, got[2].Doc)
	}
}
