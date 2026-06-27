---
title: "Quick start"
description: "From a crawl export to a live search endpoint answering ranked queries, in four commands."
weight: 30
---

This walks the core loop: build a collection from a crawl export, fit a ranking model over it, serve it, and run a query. A real crawl arrives as a Parquet file, but to keep this self-contained we start from a tiny newline-delimited JSON file you can paste straight into a terminal.

## 1. Make a small crawl export

A crawl export is one JSON record per line, each with a `url`, a `host`, and the page text (`markdown` or `body`):

```bash
cat > crawl.jsonl <<'EOF'
{"url":"https://a.example/intro","host":"a.example","markdown":"# Introduction to search\nLearned sparse retrieval ranks documents by impact."}
{"url":"https://a.example/ranking","host":"a.example","markdown":"# Ranking with gradient boosted trees\nLambdaMART optimizes a ranking objective."}
{"url":"https://b.example/vectors","host":"b.example","markdown":"# Dense vectors\nQuantized embeddings power approximate nearest neighbor search."}
{"url":"https://b.example/graph","host":"b.example","markdown":"# The link graph\nPageRank scores a page by the pages that link to it."}
EOF
```

## 2. Build a collection

```bash
tsumugi build --source crawl.jsonl --out ./data --shard-size 2
```

tsumugi reads the export, orders the documents by host for locality, and writes them into `.tsumugi` shards under `./data`:

```
built 4 docs from 2 hosts into 2 shards (0.1 MB) in 12ms
```

Look at what landed:

```bash
tsumugi collection list ./data
```

```
SHARD                BASE  DOCS  SIZE
shard-00000.tsumugi  0     2     0.1 MB
shard-00001.tsumugi  2     2     0.1 MB
total                      4     0.1 MB
```

Each shard owns a contiguous slice of the global document id space, shown by its base. You can look inside any one file:

```bash
tsumugi inspect ./data/shard-00000.tsumugi
```

## 3. Train a ranking model

```bash
tsumugi train ./data --out ./data/model.bin
```

This fits a LambdaMART model over the collection's feature matrix using the static-rank prior as a cold-start label, the model the serve command ranks with until real relevance judgments replace it:

```
trained 200 trees over 4 documents in 1 queries, wrote ./data/model.bin
```

## 4. Serve it

```bash
tsumugi serve --dir ./data --model ./data/model.bin --addr :8080
```

```
serving 2 shards (4 docs) on :8080
```

In another terminal, run a query. The endpoint returns the ranked top-k as JSON, with the elapsed time and the number of shards it touched:

```bash
curl 'localhost:8080/search?q=ranking&k=3'
```

```json
{"hits":[{"doc_id":1,"score":2.71},{"doc_id":0,"score":1.04}],"shards":2,"took_ms":0.4}
```

`/healthz` reports the collection size:

```bash
curl localhost:8080/healthz
```

```json
{"docs":4,"shards":2,"status":"ok"}
```

## What just happened

`build` packed the crawl into shards, `train` fit a model over their features, and `serve` stood up a broker that routes each query to the shards that can answer it, gathers their candidates, and runs one global rerank so the merged top-k is exact across the collection. On a real crawl the same four commands scale to tens of millions of documents across thousands of shards, with selective queries still answered in well under ten milliseconds.

## Where to go next

- The [guides](/guides/) cover [building a collection](/guides/building-a-collection/), [training a model](/guides/training-a-model/), [serving search](/guides/serving-search/), and [keeping a collection fresh](/guides/maintaining-a-collection/) in depth.
- The [CLI reference](/reference/cli/) lists every command and flag.
- The [shard format](/reference/shard-format/) page documents what is inside a `.tsumugi` file.
