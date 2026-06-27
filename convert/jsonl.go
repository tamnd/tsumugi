package convert

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
)

// jsonlRecord is one line of a newline-delimited JSON crawl export. It accepts either
// markdown or body for the text and falls back gracefully, so an export from a
// different crawler that names the field body still reads.
type jsonlRecord struct {
	URL       string `json:"url"`
	Host      string `json:"host"`
	Markdown  string `json:"markdown"`
	Body      string `json:"body"`
	CrawlDate string `json:"crawl_date"`
}

// jsonlSource streams documents from a newline-delimited JSON file, optionally
// gzipped. Each line is one record; the scanner buffer is grown so a long markdown
// body on one line does not overflow it.
type jsonlSource struct {
	closers []io.Closer
	scan    *bufio.Scanner
}

func openJSONL(path string, gz bool) (Source, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var r io.Reader = f
	closers := []io.Closer{f}
	if gz {
		zr, err := gzip.NewReader(f)
		if err != nil {
			_ = f.Close()
			return nil, err
		}
		r = zr
		closers = append([]io.Closer{zr}, closers...)
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	return &jsonlSource{closers: closers, scan: sc}, nil
}

func (s *jsonlSource) Next() (Document, bool, error) {
	for s.scan.Scan() {
		line := s.scan.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec jsonlRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			return Document{}, false, err
		}
		body := rec.Markdown
		if body == "" {
			body = rec.Body
		}
		return Document{URL: rec.URL, Host: rec.Host, Body: body, CrawlDate: rec.CrawlDate}, true, nil
	}
	if err := s.scan.Err(); err != nil {
		return Document{}, false, err
	}
	return Document{}, false, nil
}

func (s *jsonlSource) Close() error {
	var first error
	for _, c := range s.closers {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
