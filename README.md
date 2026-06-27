# tsumugi

[![ci](https://github.com/tamnd/tsumugi/actions/workflows/ci.yml/badge.svg)](https://github.com/tamnd/tsumugi/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/tamnd/tsumugi)](https://github.com/tamnd/tsumugi/releases/latest)
[![Go Reference](https://pkg.go.dev/badge/github.com/tamnd/tsumugi.svg)](https://pkg.go.dev/github.com/tamnd/tsumugi)
[![Go Report Card](https://goreportcard.com/badge/github.com/tamnd/tsumugi)](https://goreportcard.com/report/github.com/tamnd/tsumugi)
[![License](https://img.shields.io/github/license/tamnd/tsumugi)](./LICENSE)

**tsumugi** (紡) weaves a crawl into a web-scale search and ranking engine on
compact single-file shards. One `.tsumugi` file is a self-describing shard: a
single container holding the inverted index, the stored fields, a quantized
feature matrix, the compressed link graph, and quantized vectors for one slice of
a corpus. A broker maps many shards, retrieves candidates with block-max pruned
postings, ranks them through a learned cascade, and merges an exact fleet-wide
top-k, with a ten-millisecond budget at a hundred thousand shards as the goal it
is built to hit.

It is built for the rest of the fleet. A crawler like
[ami](https://github.com/tamnd/ami) or a corpus like
[Common Crawl via ccrawl-cli](https://github.com/tamnd/ccrawl-cli) provides the
documents, a builder packs them into many `.tsumugi` shards, and a broker serves
ranked results off the same bytes.

## Status

The engine runs end to end. The shard container (M0), the lexical index with
BM25F and BlockMax-WAND (M1), the forward store and quantized feature matrix
(M2-M3), the compressed link graph and its signals (M4-M5), the learned-sparse
impact index and quantized vectors (M6-M7), the ranking cascade (M8), the
LambdaMART trainer and graded-metrics eval (M9), the per-shard search and the
fleet-wide broker (M10), and the build tooling (M11) are all implemented and
tested. A `.tsumugi` file is a self-describing single file with a CRC-checked
header, footer, and per-region integrity, written append-then-footer with an
atomic rename so a torn write is rejected at open, and read through a memory map
so an uncompressed region serves with no heap copy.

Proven end to end on a real [ccrawl-cli](https://github.com/tamnd/ccrawl-cli)
export: `build` packs 20,246 documents from 18,777 hosts into 11 shards (92.4 MB)
in 11s, `train` fits a 150-tree model, `serve` answers queries across all shards
in under a millisecond each, and `compact` merges the 11 shards into 3 (84.3 MB)
in 9s. The broker reproduces a single-index top-k bit for bit across a
partitioned collection, and a worst-case all-matching query over 50,000 documents
in 16 shards returns in 6.9 ms, inside the ten-millisecond budget the design is
built to hold at a hundred thousand shards.

## Install

```bash
go install github.com/tamnd/tsumugi/cmd/tsumugi@latest
```

Prebuilt binaries, `.deb`/`.rpm`/`.apk` packages, a multi-arch GHCR image, and
checksums ship on [releases](https://github.com/tamnd/tsumugi/releases) once the
first version is tagged.

## Quick start

Build a collection from a crawl export, fit a bootstrap model, and serve it:

```bash
# Pack a Parquet or JSONL crawl export into shards under ./data.
tsumugi build --source crawl.parquet --out ./data --shard-size 2000

# Fit a cold-start ranking model over the collection's features.
tsumugi train ./data --out ./data/model.bin

# Serve ranked results over HTTP with a per-request latency budget.
tsumugi serve --dir ./data --model ./data/model.bin --addr :8080
curl 'localhost:8080/search?q=education&k=10'
```

Manage a collection as later crawls arrive:

```bash
tsumugi collection list ./data           # shards, bases, sizes
tsumugi collection add ./data --source fresh.jsonl
tsumugi collection compact ./data        # merge shards back down
```

Point `inspect` at any shard to see its layout:

```bash
tsumugi inspect shard.tsumugi
```

```
file:     shard.tsumugi
version:  1.0
docs:     20246
flags:    lexical,forward,feature
node_base:500000
regions:
  kind     codec  on-disk  raw     ratio
  lexical  zstd   ...      ...     ...x
  forward  zstd   ...      ...     ...x
  feature  none   ...      ...     1.00x
stats:
  avg_doc_len    207.4
  doc_count      20246
  term_count     1.4e+06
  token_count    4.2e+06
```

## How a shard is laid out

A `.tsumugi` file is a header, a run of regions, a footer, and a trailer. Only the
header (at offset zero) and the trailer (at the end) sit at fixed offsets;
everything else is reached through region descriptors in the footer, so the
physical order of regions is not load bearing and a new region kind can be added
without breaking an old reader. The footer is written last and the trailing magic
is the completeness marker: a file without it, or with a footer that fails its
CRC, is a torn write and is refused. Integrity is layered, the header, the footer,
and each region carry their own CRC32C, and a region's CRC is checked lazily on
first access so opening a shard costs only the header and footer no matter how
large the shard is.

## License

MIT. See [LICENSE](./LICENSE).
