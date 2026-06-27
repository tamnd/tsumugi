---
title: "Shard format"
description: "What is inside a .tsumugi file: the header, the regions, the footer, and the integrity model."
weight: 20
---

A `.tsumugi` file is one self-describing shard. It holds everything a query touches for its slice of the corpus, behind a layout designed so it can be opened cheaply, read without copying, and extended with new region kinds without breaking older readers. Use [`tsumugi inspect`](/reference/cli/) to print any file's header, regions, and stats.

## The overall layout

A file is four parts in order:

1. a **header** at offset zero;
2. a run of **regions**, one per subsystem;
3. a **footer** with a descriptor per region;
4. a **trailer** with the closing magic.

Only the header and the trailer sit at fixed offsets. Every region is reached through a descriptor in the footer, which records the region's kind, its offset, its on-disk and raw sizes, its codec, and its CRC. Because nothing points at a region by a fixed position, the physical order of regions is not load-bearing, and a new region kind can be added without breaking a reader that does not know about it. The file opens with the magic `TSM1` at both ends.

## The header

The header carries the shard's place in the world and what it contains:

- the **node base**, the global document id of the shard's first document, which is how a shard's local id space `[0, N)` maps into the collection's global ids;
- the **document count** `N`;
- the **build epoch**, a timestamp for when the shard was built;
- the **capability flags**, a bitset saying which regions and modes the shard carries.

The flags let a reader know what a shard holds before touching the footer: whether it has a lexical, forward, feature, graph, vector, or dictionary region, whether it is search-only, and whether its lexical region is an impact-quantized learned-sparse index rather than a classic BM25 index.

## The regions

Each region is one subsystem's data, compressed independently:

| Region | Holds |
|--------|-------|
| **Lexical** | The inverted index. Either a classic BM25 index or a learned-sparse impact index, distinguished by the impact-postings flag rather than by a separate kind. |
| **Forward** | The stored fields: the url, the derived title, and the body, so the shard carries the text it was built from. |
| **Feature** | The quantized feature matrix, one row of per-document signals per document, the rows the ranking model scores. |
| **Graph** | The compressed link graph, forward and transpose, for the link-based signals. |
| **Vector** | The quantized dense vectors and the index over them, for the optional dense retrieval plane. |
| **Dictionary** | A shared dictionary the other regions' compression references. |

A shard need not carry every region. A search-only shard, for instance, can drop the forward store; a collection without a link graph leaves the graph region out. A reader skips a region a shard does not have, and the same query runs against a minimal shard and a full one without special-casing, just using fewer planes.

## Compression and the document id space

A shard owns a dense document id space `[0, N)`, and documents are placed in that space in host order so a host's pages sit adjacent. That ordering is the locality the per-region compression and the shared dictionary exploit: adjacent documents share vocabulary, link structure, and boilerplate, so they delta and dictionary-compress well. The node base then lifts the local ids into the collection's global space, which is what lets many shards merge into one ranked result without id collisions.

## Integrity

Integrity is layered. The header, the footer, and each region carry their own CRC32C. A region's CRC is verified lazily, on first access, so opening a shard costs only the header and the footer no matter how large the file is, and reading a region pays for its own check only when you touch it.

The footer is written last, and the trailing magic is the completeness marker. A file without it, or with a footer that fails its CRC, is a torn write and is refused at open. A build writes the file append-then-footer into a temporary path and renames it into place, so the final name only ever points at a complete, valid shard. Combined with the fact that shards are never mutated after they are written, this is what makes [adding and compacting](/guides/maintaining-a-collection/) safe to run against a live directory.

## Reading without copying

A shard is read through a memory map. An uncompressed region is served straight from the mapping with no heap copy, so a reader aliases the file's bytes rather than allocating its own. This is why a `serve` process can hold a large collection in mapped files and why several servers can map the same immutable shards at once with no coordination.
