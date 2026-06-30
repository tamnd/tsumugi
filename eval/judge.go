package eval

import (
	"context"
	"fmt"
	"strings"
	"unicode"
)

// Grade is a graded relevance judgment on the TREC zero-to-three scale UMBRELA
// labels against: zero the document has nothing to do with the query, one it is
// related but does not answer it, two it carries an answer but buried among other
// material, three it is dedicated to the query and holds the exact answer. The
// scale is the one the harness's NDCG gain of 2^grade-1 is defined over, so a
// judge's labels drop straight into a qrels file the metrics already read.
type Grade int

// The four points of the UMBRELA relevance scale, named so the judge code reads as
// the prompt does rather than as bare integers.
const (
	GradeIrrelevant Grade = 0
	GradeRelated    Grade = 1
	GradeHighly     Grade = 2
	GradePerfect    Grade = 3
)

// MaxGrade is the top of the scale, the bound the confusion matrix is sized to and
// every grade is clamped into.
const MaxGrade = 3

// Passage is a document presented to a judge: its identifier and the text a judge
// reads to grade it, the title and the body the forward store holds. The judge sees
// the same text a reader would, so a label it assigns is a judgment of the page and
// not of an identifier.
type Passage struct {
	Doc   string
	Title string
	Body  string
}

// Judge assigns a graded relevance label to a query-passage pair. It is the seam the
// quality harness pools and labels through: the LLM judge that calls a model with the
// UMBRELA prompt and the deterministic lexical judge CI runs both satisfy it, so the
// pooling, the qrels generation, and the gold-set agreement check are written once
// against the interface and run against either grader.
type Judge interface {
	Grade(ctx context.Context, query string, p Passage) (Grade, error)
}

// UmbrelaPrompt builds the UMBRELA judging prompt for a query-passage pair, the
// careful prompt doc 14 rests the LLM-label trust on. It states the four-point scale
// in the words UMBRELA uses, asks the model to reason about the query's intent before
// it scores, and pins the output to a single trailing integer so the reply parses
// without the model's prose getting in the way. The same prompt over the same model
// at temperature zero is what makes the labels reproducible enough to commit and
// re-evaluate every configuration against.
func UmbrelaPrompt(query string, p Passage) string {
	var b strings.Builder
	b.WriteString("Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n")
	b.WriteString("0 = represents that the passage has nothing to do with the query,\n")
	b.WriteString("1 = represents that the passage seems related to the query but does not answer it,\n")
	b.WriteString("2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information, and\n")
	b.WriteString("3 = represents that the passage is dedicated to the query and contains the exact answer.\n\n")
	b.WriteString("Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if the passage presents something very important related to the entire topic but also has some extra information, and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies, give it category 0.\n\n")
	b.WriteString("Query: ")
	b.WriteString(query)
	b.WriteString("\nPassage: ")
	if p.Title != "" {
		b.WriteString(p.Title)
		b.WriteString(". ")
	}
	b.WriteString(p.Body)
	b.WriteString("\n\nSplit this problem into steps:\n")
	b.WriteString("Consider the underlying intent of the search.\n")
	b.WriteString("Measure how well the content matches a likely intent of the query (M).\n")
	b.WriteString("Measure how trustworthy the passage is (T).\n")
	b.WriteString("Consider the aspects above and the relative importance of each, and decide on a final score (O).\n")
	b.WriteString("Produce a final score with no explanation, as the last line, in the exact form: Final score: <0, 1, 2, or 3>\n")
	return b.String()
}

// ParseGrade extracts the zero-to-three grade from a judge model's free-text reply.
// It reads the last integer in the text, which is where the prompt asks the model to
// put its final score, so a model that reasons in prose before answering still parses,
// and it clamps a stray out-of-range value into the scale rather than failing, since a
// judge that says four means the top grade. A reply with no integer at all is an error,
// since that is a judge that did not answer and a silent zero would label a document
// irrelevant on a parse failure.
func ParseGrade(reply string) (Grade, error) {
	last := -1
	for i := 0; i < len(reply); i++ {
		c := reply[i]
		if c < '0' || c > '9' {
			continue
		}
		j := i
		for j < len(reply) && reply[j] >= '0' && reply[j] <= '9' {
			j++
		}
		n := 0
		for k := i; k < j; k++ {
			n = n*10 + int(reply[k]-'0')
			if n > 1000 {
				break
			}
		}
		last = n
		i = j
	}
	if last < 0 {
		return 0, fmt.Errorf("eval: no grade in judge reply %q", strings.TrimSpace(reply))
	}
	return clampGrade(last), nil
}

// clampGrade folds an integer into the zero-to-three scale, the bound the metrics and
// the confusion matrix both assume a grade holds to.
func clampGrade(n int) Grade {
	if n < 0 {
		return GradeIrrelevant
	}
	if n > MaxGrade {
		return GradePerfect
	}
	return Grade(n)
}

// LexicalJudge is a deterministic judge that grades a passage by how completely it
// covers the query's terms, the offline grader CI uses where calling a live LLM would
// be nondeterministic and unavailable. It is a real relevance signal, not a stub: a
// passage that mentions none of the query's terms grades zero, one that mentions some
// grades related, one that mentions them all grades highly, and one that mentions them
// all in its title grades perfect, the title being the strongest single evidence a page
// is about a query. It does not claim the UMBRELA Kendall-tau agreement with human
// judgment the LLM judge is validated for; it makes the pooling, qrels, ladder, and
// agreement machinery runnable and testable without a model, and the gold-set check is
// what measures any real judge's agreement before its labels are trusted.
type LexicalJudge struct{}

// Grade scores a passage against a query by query-term coverage, the deterministic
// relevance proxy described on the type. It never errors, so it composes into the
// pooling and ladder paths the same way the LLM judge does without a network failure
// mode.
func (LexicalJudge) Grade(_ context.Context, query string, p Passage) (Grade, error) {
	qterms := termSet(query)
	if len(qterms) == 0 {
		return GradeIrrelevant, nil
	}
	body := termSet(p.Title + " " + p.Body)
	title := termSet(p.Title)
	matched, inTitle := 0, 0
	for t := range qterms {
		if _, ok := body[t]; ok {
			matched++
		}
		if _, ok := title[t]; ok {
			inTitle++
		}
	}
	cov := float64(matched) / float64(len(qterms))
	switch {
	case matched == 0:
		return GradeIrrelevant, nil
	case cov < 0.5:
		return GradeRelated, nil
	case cov < 1.0:
		return GradeHighly, nil
	case inTitle == len(qterms):
		return GradePerfect, nil
	default:
		return GradeHighly, nil
	}
}

// termSet tokenizes text into the set of its distinct lowercased alphanumeric terms,
// the bag the lexical judge measures coverage over. It splits on every non-letter,
// non-digit rune so it tokenizes the same way regardless of punctuation, and the set
// shape means a term repeated in a long body counts once, so coverage measures how
// many of the query's distinct terms the passage mentions rather than how often.
func termSet(text string) map[string]struct{} {
	set := map[string]struct{}{}
	for _, f := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		set[f] = struct{}{}
	}
	return set
}
