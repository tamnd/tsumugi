package eval

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// RankedDoc is one document in a query's result list: its identifier and the score
// the engine gave it. The harness orders a query's documents by this score, so the
// rank a run file states is advisory and the score is authoritative, the same rule
// trec_eval follows when it re-sorts a run before scoring.
type RankedDoc struct {
	Doc   string
	Score float64
}

// Run is an engine's output over a query set: per query, the documents it returned
// keyed by the query identifier. The slice is the ranking, but the harness sorts it
// by score before scoring so a run read off disk in any order scores the same.
type Run map[string][]RankedDoc

// Qrels is the graded relevance judgments: per query, each judged document's grade
// on the TREC zero-to-three scale. A document absent from a query's map is
// unjudged, which the pooling assumption treats as irrelevant (grade zero), so the
// map holds only the judged documents and the harness defaults the rest.
type Qrels map[string]map[string]float64

// ParseRun reads a run file in TREC six-column format, the columns query, the
// literal Q0, document, rank, score, and run tag, whitespace separated, one ranked
// document per line. The rank and tag columns are read and ignored the way
// trec_eval ignores them, since the score is what orders the list. Blank lines are
// skipped; a line without six fields or an unparseable score is an error, since a
// malformed run would silently drop results and quietly lower every metric.
func ParseRun(r io.Reader) (Run, error) {
	run := Run{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for line := 1; sc.Scan(); line++ {
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 6 {
			return nil, fmt.Errorf("eval: run line %d has %d fields, want 6", line, len(fields))
		}
		score, err := strconv.ParseFloat(fields[4], 64)
		if err != nil {
			return nil, fmt.Errorf("eval: run line %d score %q: %w", line, fields[4], err)
		}
		run[fields[0]] = append(run[fields[0]], RankedDoc{Doc: fields[2], Score: score})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return run, nil
}

// WriteRun writes a run in TREC six-column format, one query at a time in sorted
// query order and each query's documents in scored order, assigning the rank column
// from that order. tag labels the run in the sixth column, the run-tag convention
// that lets a pooled qrels record which system contributed a document. The output
// is deterministic, the property a committed run file and a reproducibility check
// both need.
func WriteRun(w io.Writer, run Run, tag string) error {
	bw := bufio.NewWriter(w)
	for _, q := range sortedQueries(run) {
		docs := sortedByScore(run[q])
		for i, d := range docs {
			if _, err := fmt.Fprintf(bw, "%s Q0 %s %d %s %s\n",
				q, d.Doc, i+1, strconv.FormatFloat(d.Score, 'g', -1, 64), tag); err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}

// ParseQrels reads a judgments file in TREC four-column format, the columns query,
// iteration, document, and grade, whitespace separated. The iteration column is the
// TREC convention's unused field and is ignored. A repeated query-document pair
// keeps the last grade. A line without four fields or an unparseable grade is an
// error for the same reason a malformed run is: a dropped judgment silently changes
// a metric.
func ParseQrels(r io.Reader) (Qrels, error) {
	q := Qrels{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for line := 1; sc.Scan(); line++ {
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 4 {
			return nil, fmt.Errorf("eval: qrels line %d has %d fields, want 4", line, len(fields))
		}
		grade, err := strconv.ParseFloat(fields[3], 64)
		if err != nil {
			return nil, fmt.Errorf("eval: qrels line %d grade %q: %w", line, fields[3], err)
		}
		if q[fields[0]] == nil {
			q[fields[0]] = map[string]float64{}
		}
		q[fields[0]][fields[2]] = grade
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return q, nil
}

// sortedByScore returns a query's documents in scored order, highest score first,
// breaking a tie by the larger document identifier. The tie rule matches trec_eval,
// which sorts equal-score documents by descending document name so the metric over
// a run with score ties is reproducible rather than dependent on the file order, and
// it does not mutate the caller's slice.
func sortedByScore(docs []RankedDoc) []RankedDoc {
	out := append([]RankedDoc(nil), docs...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Doc > out[j].Doc
	})
	return out
}

// sortedQueries returns a run's query identifiers in ascending order, the stable
// order WriteRun and the report iterate in.
func sortedQueries[V any](m map[string]V) []string {
	qs := make([]string, 0, len(m))
	for q := range m {
		qs = append(qs, q)
	}
	sort.Strings(qs)
	return qs
}
