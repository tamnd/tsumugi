---
title: "Building a collection"
description: "Turn a Parquet or JSONL crawl export into a directory of .tsumugi shards, and choose a shard size that fits your corpus."
weight: 10
---

A collection is a directory of `.tsumugi` shards built from a crawl export. `tsumugi build` is the command that creates one.

## The input

A crawl export is a Parquet file or a newline-delimited JSON file. tsumugi picks the reader by extension: `.parquet` for the columnar form, `.jsonl` or `.json` for the line form, and `.jsonl.gz` or `.json.gz` for the gzipped line form. Each record needs three things:

- a **url**, which the ranking signals key off and the forward store keeps;
- a **host**, which the build orders on so a host's pages share a shard;
- the page **text**, read from a `markdown` field, falling back to a `body` field.

A record with no text is skipped, since it carries nothing to index. The Parquet reader follows the [open-index/open-markdown](https://huggingface.co/datasets/open-index/open-markdown) column layout that crawlers like [ami](https://github.com/tamnd/ami) and [ccrawl-cli](https://github.com/tamnd/ccrawl-cli) export, so an export from either drops straight in.

## Building

```bash
tsumugi build --source crawl.parquet --out ./data
```

tsumugi reads the whole export, orders the documents by host then url, and cuts the ordered stream into shards under `./data`. The output reports the corpus shape:

```
built 20246 docs from 18777 hosts into 11 shards (92.4 MB) in 11s
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | | Crawl export to read (`.parquet` or `.jsonl`) |
| `--out` | | Output directory for the shards |
| `--shard-size` | `50000` | Documents per shard |
| `--limit` | `0` | Cap documents read, zero for all |

`--limit` is handy for a trial run over the first slice of a large export before committing to the whole thing.

## What a build writes

For each slice of documents, a shard gets three regions:

- a **lexical index** over the title, body, and url fields, so a query matches against all three;
- a **feature matrix** of the per-document signals the analyzer derives (a static-rank prior, the body and title lengths, the url depth and length, an https flag, a text-quality score, and a language flag);
- a **forward store** keeping the url, the derived title, and the body, so the shard holds the text it was built from.

The title is derived from the page itself: the first markdown heading, or the first line if there is none. The build assigns dense global document ids in host order, the id space every later stage and the broker key off.

## Choosing a shard size

Shard size trades off two things. Smaller shards mean more files and a larger routing index, but each shard opens faster and a query that routes to one touches less data. Larger shards compress better, because more of a host's pages land together, and the per-shard fixed costs amortize over more documents.

The default of fifty thousand documents per shard is a reasonable middle for a web crawl. If you are serving a latency-critical workload and your queries are very selective, smaller shards can lower tail latency; if you are optimizing for on-disk size and have many small hosts, larger shards pack tighter. You can always change your mind later: [`collection compact`](/guides/maintaining-a-collection/) rewrites the whole collection at a new shard size.

## Locality is the point of host ordering

Ordering by host is not cosmetic. A host's pages share vocabulary, share link structure, and often share boilerplate, so placing them adjacent is what lets the per-region compression and the shared dictionary do their work. It is also what makes the routing index effective: a host's distinctive terms cluster into one or two shards, so a query for them routes narrowly instead of fanning out.

## Next

Once a collection exists, [train a model](/guides/training-a-model/) over it, then [serve it](/guides/serving-search/).
