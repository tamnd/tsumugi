---
title: "Release notes"
description: "What changed in each tsumugi release."
weight: 30
---

The authoritative, commit-level history lives in [`CHANGELOG.md`](https://github.com/tamnd/tsumugi/blob/main/CHANGELOG.md) and on the [releases page](https://github.com/tamnd/tsumugi/releases). This page summarises each version.

## v0.1.0

The first release. tsumugi packs a web crawl into compact single-file shards and serves an exact, ranked, fleet-wide top-k in milliseconds, from one pure-Go binary.

- **`tsumugi build`** turns a Parquet or newline-delimited JSON crawl export into a directory of `.tsumugi` shards, ordering documents by host for locality and writing the lexical index, the feature matrix, and the forward store into each shard.
- **`tsumugi train`** fits a LambdaMART ranking model over a collection's features, using the static-rank prior as a cold-start label, in pure Go with no LightGBM or Python in the loop.
- **`tsumugi serve`** stands a collection up behind HTTP. A broker routes each query to the shards that can answer it, gathers their candidates, and runs one global rerank under a per-request latency budget, so the merged top-k is bit-for-bit what a single index over every shard would return.
- **`tsumugi collection`** lists, extends, and compacts a shard directory: `add` brings a later crawl in without rewriting existing shards, and `compact` merges accumulated shards back into fewer, larger ones, both safe to run against a live directory because shards are immutable once written.
- **`tsumugi inspect`** prints a shard's header, region table, and statistics.
- **The `.tsumugi` shard format.** One self-describing file holds the inverted index, the stored fields, a quantized feature matrix, the compressed link graph, and quantized vectors, behind a CRC-checked header and footer, written append-then-footer with an atomic rename and read through a memory map with no heap copy for uncompressed regions.
- **Retrieval and ranking.** A doc-side learned-sparse impact index queried with Block-Max pruning, an optional dense plane over quantized vectors, and a ranking cascade that retrieves, cuts over the feature matrix, and reranks with the trained model.
- **Proven end to end.** On a real crawl export, `build` packs 20,246 documents from 18,777 hosts into 11 shards (92.4 MB) in 11 seconds, `train` fits a 150-tree model, `serve` answers queries in under a millisecond each, and `compact` merges the shards down to 3 (84.3 MB) in 9 seconds. A worst-case all-matching query over 50,000 documents in 16 shards returns in 6.9 milliseconds.
- **Packaged everywhere.** Archives, `.deb`/`.rpm`/`.apk` packages, a multi-arch GHCR image, Homebrew and Scoop, checksums, SBOMs, and a cosign signature.
