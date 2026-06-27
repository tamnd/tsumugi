//go:build unix

package tsumugi

import (
	"os"

	"golang.org/x/sys/unix"
)

// mmap holds a read-only memory mapping of a shard file. On unix the file is
// mapped so a shard serves straight out of the page cache and a broker can hold
// many shards resident without copying their bytes into the heap.
type mmap struct {
	data []byte
	f    *os.File
}

func mmapOpen(path string) (*mmap, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := fi.Size()
	if size == 0 {
		f.Close()
		return nil, ErrShortFile
	}
	data, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}
	// Hint sequential-ish random access; ignore the error, it is advisory.
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	return &mmap{data: data, f: f}, nil
}

func (m *mmap) Close() error {
	err := unix.Munmap(m.data)
	if cerr := m.f.Close(); err == nil {
		err = cerr
	}
	return err
}
