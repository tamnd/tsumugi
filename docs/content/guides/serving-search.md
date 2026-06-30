---
title: "Serving search"
description: "Stand up a search endpoint over a collection, query it, and understand the routing, the latency budget, and why the merged top-k is exact."
weight: 30
---

`tsumugi serve` opens a collection, builds a broker over its shards, and answers ranked queries over HTTP.

```bash
tsumugi serve --dir ./data --model ./data/model.bin --addr :8080
```

```
serving 11 shards (20246 docs) on :8080
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--dir` | | Directory of `.tsumugi` shards to serve |
| `--model` | | Trained ranking model file |
| `--addr` | `:8080` | Address to listen on |
| `--timeout` | `10ms` | Per-request deadline |
| `--cache` | `0` | Result-cache size in entries (0 disables) |
| `--max-inflight` | `0` | Maximum concurrent searches before shedding with 503 (0 disables) |
| `--reload-interval` | `0` | Poll the shard directory at this interval to publish and retire shards (0 disables polling) |

## Querying

The `/search` endpoint takes a query string `q` and an optional cutoff `k`:

```bash
curl 'localhost:8080/search?q=open+source+search&k=10'
```

It returns the ranked top-k as JSON: each hit's global document id and score, the number of shards the query touched, and the elapsed time in milliseconds.

```json
{
  "hits": [
    {"doc_id": 15906, "score": 7.42},
    {"doc_id": 17100, "score": 6.88}
  ],
  "shards": 11,
  "took_ms": 0.6
}
```

`/healthz` reports the collection size and is the endpoint to point a load balancer or a readiness probe at:

```bash
curl localhost:8080/healthz
```

```json
{"docs": 20246, "shards": 11, "status": "ok"}
```

## Reloading shards without a restart

A running server can take on freshly built shards and drop retired ones without stopping. Build a new `.tsumugi` file into the served directory, or remove one, then tell the server to reconcile its served set with what is on disk:

```bash
curl -X POST localhost:8080/admin/reload
```

```json
{"published": 2, "retired": 1, "shards": 12, "docs": 21430}
```

`reload` globs the directory, publishes every file not yet served, and retires every served shard whose file is gone. To act on a single shard by name instead, use `publish` and `retire`:

```bash
curl -X POST 'localhost:8080/admin/publish?shard=shard-00012.tsumugi'
curl -X POST 'localhost:8080/admin/retire?shard=shard-00003.tsumugi'
```

A published shard whose recorded analyzer does not match the server's is refused, the same check the server applies at startup, so a shard built with a different analysis chain can never be queried against tokens the server does not produce. A retired shard's documents stop appearing in new results immediately; a query already in flight finishes against the set it started with.

The admin endpoints are POST only. To reconcile unattended instead of on demand, set `--reload-interval` and the server polls the directory on that interval:

```bash
tsumugi serve --dir ./data --model ./data/model.bin --reload-interval 30s
```

## What a query does

When the broker starts, it builds two things over the shards once: a routing index that maps each term to the shards holding a posting for it, and the fleet-wide statistics (the total document and token counts, and the collection-wide average document length the term normalization needs).

A query then:

1. **routes** to the shards whose vocabulary intersects it, so a selective query touches a handful of shards rather than the whole fleet;
2. **fans out** to those shards concurrently, bounded by a worker semaphore, each shard returning its candidates with their feature rows already attached and ids shifted into the global space;
3. **reranks** the union once with the global model and returns the top-k.

A query with no matching terms anywhere returns nothing; a query whose terms are everywhere touches every shard, which is the worst case the latency budget is sized against.

## The latency budget

`--timeout` is a per-request deadline that cancels the fan-out, so a single slow shard cannot hold a query past its budget. The default is ten milliseconds, the figure the engine is designed around: a worst-case all-matching query over fifty thousand documents in sixteen shards returns in under seven milliseconds, and a typical selective query over a real crawl returns in well under one. Raise the timeout for an analytical workload that prefers completeness over latency; lower it to shed tail latency under load.

## Why the merged result is exact

The broker does the rerank, and the shards only retrieve. A document's final score is the global model evaluated over that document's own feature row, and the feature row is identical whether the document sits in a shard or in a single combined index. So the score never depends on the partitioning. The fusion and the cut only choose which candidates reach the reranker; the final order is the model's score, ties broken by id, both deterministic.

The consequence is that, given recall-complete retrieval, the broker's top-k is bit-for-bit the top-k a single index over every shard would return. Sharding here is a layout decision, not an approximation: you split the corpus for size and parallelism without trading away the exactness of the ranking.

## Running behind a proxy

The server speaks plain HTTP and holds the collection read-only in memory-mapped files, so it scales out by running more copies over the same shard directory behind a load balancer. Because shards are immutable once written, several `serve` processes can map the same files at once with no coordination. To refresh the served data, [add or compact shards](/guides/maintaining-a-collection/) and restart the servers against the updated directory.
