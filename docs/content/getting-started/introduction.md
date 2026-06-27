---
title: "Introduction"
description: "Why a search engine fits in one file per shard, and how tsumugi retrieves and ranks without a transformer on the query path."
weight: 10
---

A classic web search stack is a cluster of services. The inverted index lives in one system, the document store in another, the feature store in a third, and the ranker in a fourth, each with its own process, its own on-disk format, and its own scaling story. Getting a single ranked result means a fan-out across all of them, and keeping them consistent is most of the operational work.

tsumugi collapses that into a file. A crawl is split into shards, and a shard is one `.tsumugi` file that already holds everything a query touches: the inverted index, the stored fields, the per-document features, the link graph, and the vectors. One binary builds these files, trains a ranking model over them, and serves them. There is no separate index server to run and no model runtime to install.

## A shard is a single file

A `.tsumugi` file is a header, a run of regions, a footer, and a trailing magic. Each region is one subsystem's data: the lexical index, the forward store of stored fields, the quantized feature matrix, the compressed link graph, the quantized vectors, and a shared dictionary. The header and trailer sit at fixed offsets; every region is reached through a descriptor in the footer, so a new region kind can be added without breaking an older reader.

The file is written once, append-then-footer, into a temporary path and renamed into place, so the final name only ever points at a complete shard. It is read through a memory map with each region's CRC checked lazily on first access, so opening a shard costs only the header and footer no matter how large the file is, and an uncompressed region is served straight from the mapping with no heap copy. A shard owns a dense document id space `[0, N)`, and its node base places that range in the collection's global id space, which is what lets many shards merge into one ranked result.

## Retrieval is learned-sparse, ranking is a cascade

tsumugi retrieves with a doc-side learned-sparse index. At build time a model assigns impact weights to the terms of each document, and those weights are stored as quantized postings. At query time the engine walks them with Block-Max pruning, so it never scores a document that cannot reach the top-k. There is no transformer run on the query, which is what keeps retrieval cheap on a CPU. An optional dense plane retrieves over quantized vectors and is fused with the lexical results.

Ranking is a cascade, each stage cutting the candidate set the next one sees:

1. **Retrieve** a few hundred candidates per shard from the pruned postings.
2. **Cut** them with a fast linear pass over the feature matrix down to a smaller set.
3. **Rerank** that set with a LambdaMART model, evaluated with vectorized tree inference, to produce the final order.

The model is a gradient-boosted set of regression trees fit to a ranking objective. tsumugi trains it in pure Go, so there is no LightGBM or Python in the loop, and the trained model compiles to the same vectorized scorer the serve path uses.

## Many shards, one exact answer

A collection is a directory of shards. A broker opens them, builds a routing index from each shard's vocabulary, and computes fleet-wide statistics once. A query routes only to the shards whose terms intersect it, so a selective query touches a handful of shards rather than the whole fleet. Each routed shard returns its candidates with their feature rows attached, and the broker runs one global rerank over the union.

That last step is what makes the result exact rather than approximate. A document's final score is the global model evaluated over that document's own feature row, and the feature row is identical whether the document sits in a shard or in a single combined index. So given recall-complete retrieval, the broker's top-k is bit-for-bit the top-k a monolithic index would return, not a best-effort blend of incomparable per-shard scores.

## The shape of a build

The input is a crawl export, a Parquet or newline-delimited JSON file with a url, a host, and the page text per record. tsumugi orders the documents by host so a host's pages share a shard and sit adjacent, which is the locality the compression relies on, then cuts the ordered stream into shards and writes each one. A later crawl is brought in with `collection add`, which extends the global id space without rewriting existing shards, and `collection compact` merges accumulated shards back down into fewer, larger ones.

## Then what?

Once a collection is built and a model is trained, `tsumugi serve` stands it up behind HTTP, answering ranked queries under a per-request latency budget. The [quick start](/getting-started/quick-start/) walks the whole loop end to end.

Next: [install tsumugi](/getting-started/installation/).
