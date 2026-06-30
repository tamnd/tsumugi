package eval

import (
	"context"
	"strings"
	"testing"
)

func TestUmbrelaPromptCarriesQueryAndPassage(t *testing.T) {
	p := Passage{Doc: "7", Title: "Go memory model", Body: "happens-before and goroutines"}
	prompt := UmbrelaPrompt("golang memory model", p)
	for _, want := range []string{
		"golang memory model",     // the query
		"Go memory model",         // the title
		"happens-before",          // the body
		"integer scale of 0 to 3", // the scale instruction
		"Final score:",            // the parse anchor
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q\n%s", want, prompt)
		}
	}
}

func TestParseGrade(t *testing.T) {
	cases := []struct {
		reply string
		want  Grade
		err   bool
	}{
		{"Final score: 3", GradePerfect, false},
		{"3", GradePerfect, false},
		{"The answer is 0.", GradeIrrelevant, false},
		{"M: 2, T: 1, O: 2\nFinal score: 2", GradeHighly, false}, // last integer wins
		{"score is 7", GradePerfect, false},                      // out of range clamps to top
		{"grade: 1 (related)", GradeRelated, false},
		{"no number here", 0, true},
		{"", 0, true},
	}
	for _, c := range cases {
		got, err := ParseGrade(c.reply)
		if c.err {
			if err == nil {
				t.Errorf("ParseGrade(%q) = %d, want error", c.reply, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseGrade(%q) error: %v", c.reply, err)
			continue
		}
		if got != c.want {
			t.Errorf("ParseGrade(%q) = %d, want %d", c.reply, got, c.want)
		}
	}
}

func TestLexicalJudgeGrades(t *testing.T) {
	j := LexicalJudge{}
	ctx := context.Background()
	cases := []struct {
		name  string
		query string
		p     Passage
		want  Grade
	}{
		{
			name:  "no overlap is irrelevant",
			query: "kubernetes networking",
			p:     Passage{Title: "Banana bread recipe", Body: "flour sugar butter"},
			want:  GradeIrrelevant,
		},
		{
			name:  "partial overlap is related",
			query: "rust async runtime scheduler",
			p:     Passage{Title: "Async basics", Body: "an async primer"},
			want:  GradeRelated,
		},
		{
			name:  "most terms present is highly relevant",
			query: "rust async runtime",
			p:     Passage{Title: "Concurrency", Body: "the rust async runtime explained at length with extra detail"},
			want:  GradeHighly,
		},
		{
			name:  "all terms in title is perfect",
			query: "rust async runtime",
			p:     Passage{Title: "Rust async runtime internals", Body: "deep dive"},
			want:  GradePerfect,
		},
		{
			name:  "all terms but only in body is highly not perfect",
			query: "rust async runtime",
			p:     Passage{Title: "Internals", Body: "the rust async runtime in detail"},
			want:  GradeHighly,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := j.Grade(ctx, c.query, c.p)
			if err != nil {
				t.Fatalf("Grade error: %v", err)
			}
			if got != c.want {
				t.Fatalf("Grade = %d, want %d", got, c.want)
			}
		})
	}
}

func TestLexicalJudgeDeterministic(t *testing.T) {
	j := LexicalJudge{}
	ctx := context.Background()
	p := Passage{Title: "Distributed consensus", Body: "raft and paxos compared"}
	first, _ := j.Grade(ctx, "raft consensus protocol", p)
	for i := 0; i < 5; i++ {
		got, _ := j.Grade(ctx, "raft consensus protocol", p)
		if got != first {
			t.Fatalf("LexicalJudge is not deterministic: %d then %d", first, got)
		}
	}
}
