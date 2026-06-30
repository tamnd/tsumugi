---
title: "Keeping a collection fresh"
description: "Bring later crawls into a collection with add, and merge accumulated shards back down with compact, without rewriting what is already there."
weight: 40
---

A crawl is not a one-time event. New pages appear, old ones change, and you re-crawl. tsumugi keeps a collection current without rebuilding it from scratch, because a `.tsumugi` shard is immutable once written: every operation either appends new shards or rewrites the whole set into a fresh one, and never mutates a file in place.

## Inspecting what is there

```bash
tsumugi collection list ./data
```

```
SHARD                BASE   DOCS   SIZE
shard-00000.tsumugi  0      50000  44.3 MB
shard-00001.tsumugi  50000  50000  38.5 MB
shard-00002.tsumugi  100000 12431  9.8 MB
total                       112431 92.6 MB
```

The shards are listed in global id order, so the listing reads as the collection's id layout: each shard's base is where its slice of the document id space begins.

## Adding a later crawl

`collection add` brings a new crawl export into an existing collection:

```bash
tsumugi collection add ./data --source fresh-crawl.parquet
```

It reads the collection's highest existing document id, continues the global id space past it, and names its new shards after the existing ones, so the new crawl extends the collection rather than rewriting a byte of it. This is the freshness path the immutable-shard discipline makes safe: a running `serve` process keeps mapping the old shards while `add` writes new files alongside them. To make the new shards live without a restart, POST `/admin/reload` to the running server, or start it with `--reload-interval` so it picks them up on a poll. See [serving search](/guides/serving-search/#reloading-shards-without-a-restart).

`add` takes the same `--source`, `--shard-size`, and `--limit` flags as `build`. The trade-off is fragmentation: each add appends its own shards in its own host order, so after many small adds the collection is many small, separately-ordered shards. That is what compaction fixes.

## Compacting

`collection compact` merges the accumulated shards back into fewer, larger ones:

```bash
tsumugi collection compact ./data
```

```
compacted 112431 docs into 3 shards (84.3 MB) in 9s
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--shard-size` | `50000` | Documents per merged shard |

Compaction reads every document back out of the shards' forward stores, reorders the whole set by host, and rebuilds at the chosen shard size. Because it reorders across the entire collection rather than within each old shard, the result compresses better than the sum of its inputs, which is why the merged size above is smaller than the fragmented total. It is also how you change a collection's shard size after the fact: compact at a new `--shard-size`.

This is why the forward store keeps the body. A shard carries the text it was built from, so a compaction is lossless for ranking: it rebuilds the lexical index, the feature matrix, and the forward store from the stored documents alone, with no need to go back to the original crawl export.

## The atomic swap

A compaction writes the rebuilt shards into a staging directory and swaps them in only once every new shard is written: it removes the old shards, moves the staged ones up, and drops the staging directory. A compaction that fails partway leaves the original collection untouched, so the operation is safe to run against a directory you care about. To pick up the compacted data in a running server, restart it against the same directory once the swap completes.

## A typical cadence

A common rhythm is to `add` after each incremental crawl and `compact` on a schedule, say nightly or weekly, to fold the day's additions back into the main shards. Between compactions the collection stays queryable the whole time, since both `add` and `compact` leave a complete, consistent set of shards on disk at every moment.

## Retraining after large changes

Adding a lot of new documents shifts the feature distribution the ranker was fit to. After a substantial add or compact, [retrain the model](/guides/training-a-model/) over the updated collection and restart `serve` with the new model file so the ranking reflects the current corpus.
