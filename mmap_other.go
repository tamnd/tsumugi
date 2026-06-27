//go:build !unix

package tsumugi

import "os"

// mmap is the fallback for platforms without unix mmap (notably Windows). It
// reads the whole file into memory, which keeps the reader identical everywhere
// at the cost of not sharing pages with the page cache.
type mmap struct {
	data []byte
}

func mmapOpen(path string) (*mmap, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, ErrShortFile
	}
	return &mmap{data: data}, nil
}

func (m *mmap) Close() error { return nil }
