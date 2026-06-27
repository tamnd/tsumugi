package lexical

import "errors"

// errCorrupt is returned when a region's bytes do not parse: a truncated stream,
// a bad sub-offset, or a magic mismatch. The container CRC catches bit flips
// before a reader gets here, so this fires on a structurally malformed region.
var errCorrupt = errors.New("tsumugi/lexical: corrupt region")

// errBadMagic is returned when the lexical region header does not start with the
// LEX1 magic.
var errBadMagic = errors.New("tsumugi/lexical: bad region magic")
