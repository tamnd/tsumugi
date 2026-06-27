package convert

import (
	"os"

	"github.com/parquet-go/parquet-go"
)

// ccrawlRow mirrors the columns of the ccrawl markdown Parquet export. Only the
// fields the document model needs are read; the reader projects to these columns and
// skips the rest, so the html length and record id sitting in the file cost nothing.
type ccrawlRow struct {
	URL       string `parquet:"url"`
	Host      string `parquet:"host"`
	CrawlDate string `parquet:"crawl_date"`
	Markdown  string `parquet:"markdown"`
}

// parquetSource streams documents from a Parquet file in row batches, so a
// multi-gigabyte export never lands in memory at once. The batch is refilled as it
// drains.
type parquetSource struct {
	f      *os.File
	reader *parquet.GenericReader[ccrawlRow]
	buf    []ccrawlRow
	pos    int
	n      int
	done   bool
}

const parquetBatch = 256

func openParquet(path string) (Source, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	pf, err := parquet.OpenFile(f, st.Size())
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	return &parquetSource{
		f:      f,
		reader: parquet.NewGenericReader[ccrawlRow](pf),
		buf:    make([]ccrawlRow, parquetBatch),
	}, nil
}

func (s *parquetSource) Next() (Document, bool, error) {
	for {
		if s.pos < s.n {
			r := s.buf[s.pos]
			s.pos++
			return Document{URL: r.URL, Host: r.Host, Body: r.Markdown, CrawlDate: r.CrawlDate}, true, nil
		}
		if s.done {
			return Document{}, false, nil
		}
		n, err := s.reader.Read(s.buf)
		s.n = n
		s.pos = 0
		if err != nil {
			// io.EOF arrives with the last partial batch, so drain n before stopping.
			s.done = true
			if n == 0 {
				return Document{}, false, nil
			}
		}
	}
}

func (s *parquetSource) Close() error {
	_ = s.reader.Close()
	return s.f.Close()
}
