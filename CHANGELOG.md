# Changelog

All notable changes to tsumugi are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-06-27

The first release. tsumugi packs a web crawl into compact single-file shards and
serves an exact, ranked, fleet-wide top-k in milliseconds, from one pure-Go
binary.

### Added

- **The `.tsumugi` shard format.** One self-describing file holds the inverted index, the stored fields, a quantized feature matrix, the compressed link graph, and quantized vectors, behind a CRC-checked header and footer. It is written append-then-footer into a temporary path and renamed into place, so the final name only ever points at a complete shard, and it is read through a memory map with no heap copy for uncompressed regions.
- **`tsumugi build`** turns a Parquet or newline-delimited JSON crawl export into a directory of shards, ordering documents by host for locality and writing the lexical index, the feature matrix, and the forward store into each shard.
- **`tsumugi train`** fits a LambdaMART ranking model over a collection's feature matrix using the static-rank prior as a cold-start label, in pure Go.
- **`tsumugi serve`** answers ranked queries over HTTP. A broker routes each query to the shards that can contribute, gathers their candidates, and runs one global rerank under a per-request latency budget, so the merged top-k is bit-for-bit what a single index over every shard would return.
- **`tsumugi collection list`, `add`, and `compact`** manage a shard directory: `add` extends the collection without rewriting existing shards, and `compact` merges accumulated shards back into fewer, larger ones, rebuilding losslessly from each shard's forward store.
- **`tsumugi inspect`** prints a shard's header, region table, and statistics.
- **Retrieval and ranking.** A doc-side learned-sparse impact index queried with Block-Max pruning, an optional dense plane over quantized vectors, and a ranking cascade that retrieves, cuts over the feature matrix, and reranks with the trained model.
- **Packaged everywhere.** Archives, `.deb`/`.rpm`/`.apk` packages, a multi-arch GHCR image, Homebrew and Scoop, checksums, SBOMs, and a cosign signature.

[Unreleased]: https://github.com/tamnd/tsumugi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tamnd/tsumugi/releases/tag/v0.1.0
