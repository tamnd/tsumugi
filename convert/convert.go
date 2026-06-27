// Package convert turns a crawl export into the document stream the build pipeline
// ingests. A crawl lands as Parquet or newline-delimited JSON; this package reads
// either behind one Source interface so the rest of the build does not care which
// format the crawl arrived in.
//
// The document model is deliberately small: the url and host the ranking signals key
// off, the body text the lexical and feature stages read, and the crawl date the
// freshness signal uses. Everything richer, the title, the per-document features, the
// link graph, is derived downstream from these fields rather than carried here, so
// the source format stays a thin adapter and the derivation lives in one place.
package convert

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Document is one crawled page reduced to the fields the build pipeline reads. Body
// is the extracted text, markdown in the ccrawl export, which the lexical analyzer
// tokenizes and the feature stage measures.
type Document struct {
	URL       string
	Host      string
	Body      string
	CrawlDate string
}

// Source yields crawl documents one at a time. Next returns the next document and
// true, or a zero document and false at the end of the stream; an error ends
// iteration. A Source holds an open file and must be closed.
type Source interface {
	Next() (Document, bool, error)
	Close() error
}

// OpenSource opens a crawl export, choosing the reader by file extension: .parquet
// for the columnar ccrawl markdown export, .jsonl or .json for newline-delimited
// records, .jsonl.gz for the gzipped form. It is the one entry point the build
// orchestrator calls, so a new format is added here and nowhere else.
func OpenSource(path string) (Source, error) {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".parquet"):
		return openParquet(path)
	case strings.HasSuffix(lower, ".jsonl.gz"), strings.HasSuffix(lower, ".json.gz"):
		return openJSONL(path, true)
	case strings.HasSuffix(lower, ".jsonl"), strings.HasSuffix(lower, ".json"):
		return openJSONL(path, false)
	default:
		return nil, fmt.Errorf("convert: unsupported source %q, want .parquet or .jsonl%s", filepath.Base(path), "")
	}
}
