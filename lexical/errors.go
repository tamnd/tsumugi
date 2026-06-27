package lexical

import "errors"

// errCorrupt is returned when a region's bytes do not parse: a truncated stream,
// a bad sub-offset, or a magic mismatch. The container CRC catches bit flips
// before a reader gets here, so this fires on a structurally malformed region.
var errCorrupt = errors.New("tsumugi/lexical: corrupt region")

// errBadMagic is returned when the lexical region header does not start with the
// LEX1 magic.
var errBadMagic = errors.New("tsumugi/lexical: bad region magic")

// errLegacyBlockMax is returned when a lexical region predates the idf-free block-max
// format and bakes a shard-local idf into its bounds. Scoring such a region with the
// broker's pushed-down collection-wide idf would double-count the idf, so the region is
// refused and the shard has to be rebuilt rather than served with silently wrong scores.
var errLegacyBlockMax = errors.New("tsumugi/lexical: region predates idf-free block-max, rebuild the shard")
