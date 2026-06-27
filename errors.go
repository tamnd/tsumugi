package tsumugi

import "errors"

var (
	// ErrBadMagic is returned when the leading or trailing magic does not match
	// TSM1, which means the file is not a tsumugi shard or has been truncated.
	ErrBadMagic = errors.New("tsumugi: bad magic")

	// ErrShortFile is returned when the file is too small to hold a header and a
	// trailer.
	ErrShortFile = errors.New("tsumugi: file too short")

	// ErrBadVersion is returned when the major version is one this build does not
	// understand.
	ErrBadVersion = errors.New("tsumugi: unsupported major version")

	// ErrHeaderCRC is returned when the header checksum does not match its bytes.
	ErrHeaderCRC = errors.New("tsumugi: header crc mismatch")

	// ErrFooterCRC is returned when the footer checksum in the trailer does not
	// match the footer bytes.
	ErrFooterCRC = errors.New("tsumugi: footer crc mismatch")

	// ErrRegionCRC is returned when a region's bytes do not match the checksum in
	// its descriptor. The error names the region kind so corruption is localized.
	ErrRegionCRC = errors.New("tsumugi: region crc mismatch")

	// ErrCorruptFooter is returned when the footer cannot be parsed.
	ErrCorruptFooter = errors.New("tsumugi: corrupt footer")

	// ErrNoRegion is returned when a requested region kind is not present.
	ErrNoRegion = errors.New("tsumugi: region not present")
)
