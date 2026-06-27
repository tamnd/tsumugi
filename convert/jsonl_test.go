package convert

import (
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

// drain reads a source to the end and returns every document, failing the test on any
// read error so the round-trip assertions stay readable.
func drain(t *testing.T, s Source) []Document {
	t.Helper()
	var docs []Document
	for {
		d, ok, err := s.Next()
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		docs = append(docs, d)
	}
	return docs
}

func TestJSONLSource(t *testing.T) {
	const body = `{"url":"https://a.example/p1","host":"a.example","markdown":"# One\nbody one","crawl_date":"2026-01-02"}
{"url":"https://b.example/p2","host":"b.example","markdown":"# Two\nbody two","crawl_date":"2026-01-03"}

{"url":"https://c.example/p3","host":"c.example","body":"plain body three"}
`
	path := filepath.Join(t.TempDir(), "crawl.jsonl")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	src, err := OpenSource(path)
	if err != nil {
		t.Fatalf("OpenSource: %v", err)
	}
	defer func() { _ = src.Close() }()

	docs := drain(t, src)
	if len(docs) != 3 {
		t.Fatalf("got %d docs, want 3 (blank line should be skipped)", len(docs))
	}
	if docs[0].URL != "https://a.example/p1" || docs[0].Host != "a.example" {
		t.Errorf("doc0 url/host = %q/%q", docs[0].URL, docs[0].Host)
	}
	if docs[0].Body != "# One\nbody one" {
		t.Errorf("doc0 body = %q", docs[0].Body)
	}
	if docs[0].CrawlDate != "2026-01-02" {
		t.Errorf("doc0 crawl date = %q", docs[0].CrawlDate)
	}
	// Third record has no markdown, so the body field is the fallback text.
	if docs[2].Body != "plain body three" {
		t.Errorf("doc2 body fallback = %q, want the body field", docs[2].Body)
	}
}

func TestJSONLGzip(t *testing.T) {
	const line = `{"url":"https://g.example/x","host":"g.example","markdown":"gzipped text"}` + "\n"
	path := filepath.Join(t.TempDir(), "crawl.jsonl.gz")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := gzip.NewWriter(f)
	if _, err := zw.Write([]byte(line)); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	src, err := OpenSource(path)
	if err != nil {
		t.Fatalf("OpenSource: %v", err)
	}
	defer func() { _ = src.Close() }()

	docs := drain(t, src)
	if len(docs) != 1 || docs[0].Body != "gzipped text" {
		t.Fatalf("gzip round-trip = %+v, want one doc with the body", docs)
	}
}

func TestOpenSourceUnsupported(t *testing.T) {
	if _, err := OpenSource("crawl.txt"); err == nil {
		t.Fatal("OpenSource on an unsupported extension should error")
	}
}
