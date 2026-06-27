---
title: "tsumugi"
description: "tsumugi (紡) weaves a crawl into a web-scale search and ranking engine on compact single-file shards. Build, train, and serve ranked search over tens of millions of pages from one pure-Go binary, no GPU on the hot path."
heroTitle: "Search a crawl, woven into shards"
heroLead: "tsumugi packs a web crawl into compact .tsumugi files, one self-describing shard each, then serves ranked results across the whole fleet in milliseconds. The inverted index, the stored fields, the feature matrix, the link graph, and the vectors all live in one file, and a broker merges an exact top-k over many of them."
heroPrimaryURL: "/getting-started/quick-start/"
heroPrimaryText: "Get started"
---

A search engine is usually a cluster: an index service, a document store, a feature service, a ranking service, each with its own process, its own format, and its own failure modes. tsumugi (紡, "to spin thread") takes the opposite shape. One crawl becomes a set of `.tsumugi` shards, each a single file that already holds everything a query needs, and one binary builds them, trains a ranker over them, and serves them.

Point it at a crawl export and three commands take you from raw pages to a live search endpoint:

```bash
tsumugi build --source crawl.parquet --out ./data
tsumugi train ./data --out ./data/model.bin
tsumugi serve --dir ./data --model ./data/model.bin
```

## What it does

- **One file is a whole shard.** A `.tsumugi` file holds the inverted index, the stored fields, a quantized feature matrix, the compressed link graph, and quantized vectors for one slice of the corpus, behind a CRC-checked header and footer. Open it with one `mmap` and an uncompressed region serves with no heap copy.
- **Modern relevance, CPU-first.** Retrieval is a doc-side learned-sparse impact index queried with Block-Max pruning, with an optional dense plane over quantized vectors. Ranking is a cascade: cheap retrieval, a linear cut over the feature matrix, then a LambdaMART model with vectorized inference. No transformer on the query path, no GPU.
- **An exact fleet-wide top-k.** A broker routes a query to the shards that can answer it, gathers their candidates with features attached, and runs one global rerank, so the merged top-k is bit-for-bit what a single index over every shard would return.
- **Built for scale.** The design targets tens of millions of documents on a single box or a small fleet, with a ten-millisecond query budget at a hundred thousand shards.
- **Pure Go, one binary.** No cgo, no external index server, no model runtime. `CGO_ENABLED=0` builds for every platform, and the LambdaMART trainer is pure Go too.

## Where it fits

tsumugi is the ranking brain on top of the fleet's storage and crawl layers. A crawler like [ami](https://github.com/tamnd/ami) or a corpus like [Common Crawl through ccrawl-cli](https://github.com/tamnd/ccrawl-cli) provides the documents as a Parquet or JSONL export, tsumugi packs them into shards, and the broker serves ranked results off the same bytes.

## Where to go next

- New here? Start with the [introduction](/getting-started/introduction/), then the [quick start](/getting-started/quick-start/).
- Want to install it? See [installation](/getting-started/installation/).
- Looking for a specific task? The [guides](/guides/) cover building a collection, training a model, serving search, and keeping a collection fresh.
- Need every flag? The [CLI reference](/reference/cli/) is the full surface, and the [shard format](/reference/shard-format/) page documents what is inside a `.tsumugi` file.
