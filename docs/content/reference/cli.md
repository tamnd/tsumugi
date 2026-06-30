---
title: "CLI reference"
description: "Every tsumugi command and flag."
weight: 10
---

```
tsumugi [command] [flags]
```

Five command areas: `build` packs a crawl export into shards, `train` fits a ranking model over them, `serve` answers queries over a collection, `collection` manages the shard directory, and `inspect` prints a shard's internals. Run `tsumugi <command> --help` for the canonical, up-to-date list.

## tsumugi build

```
tsumugi build [flags]
```

Reads a crawl export, a Parquet or newline-delimited JSON file, orders the documents by host for locality, and writes them into `.tsumugi` shards under the output directory. The result is a collection the serve command can answer queries over directly.

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | | Crawl export to read (`.parquet` or `.jsonl`) |
| `--out` | | Output directory for the shards |
| `--shard-size` | `50000` | Documents per shard |
| `--limit` | `0` | Cap documents read, zero for all |

Supported source extensions: `.parquet`, `.jsonl`, `.json`, `.jsonl.gz`, `.json.gz`. A record needs a `url`, a `host`, and page text (a `markdown` field, falling back to `body`); records with no text are skipped.

## tsumugi train

```
tsumugi train <dir> [flags]
```

Fits a LambdaMART model over a collection's feature matrix using the static-rank prior as a bootstrap label, the cold-start model the serve command ranks with until real relevance judgments replace it. It reads every shard's features, groups documents into synthetic queries, fits the model, and writes it to a file.

| Flag | Default | Meaning |
|------|---------|---------|
| `--out` | | Model output file (required) |
| `--group-size` | `16` | Documents per synthetic query group |
| `--rounds` | `200` | Boosting rounds |

## tsumugi serve

```
tsumugi serve [flags]
```

Opens every `.tsumugi` shard in a directory, builds the routing index and the fleet-wide statistics, and answers ranked queries over HTTP. Each request fans out to the shards that can contribute, gathers their candidates, and runs one global rerank, so the merged top-k is the result a single index over every shard would give.

| Flag | Default | Meaning |
|------|---------|---------|
| `--dir` | | Directory of `.tsumugi` shards to serve |
| `--model` | | Trained ranking model file |
| `--addr` | `:8080` | Address to listen on |
| `--timeout` | `10ms` | Per-request deadline |
| `--cache` | `0` | Result-cache size in entries (0 disables) |
| `--max-inflight` | `0` | Maximum concurrent searches before shedding with 503 (0 disables) |
| `--reload-interval` | `0` | Poll the shard directory at this interval to publish and retire shards (0 disables polling) |

### Endpoints

| Path | Meaning |
|------|---------|
| `GET /search?q=<query>&k=<n>` | Ranked top-k as JSON: `hits` (each `doc_id` and `score`), `shards`, `took_ms`. `k` defaults to 10. |
| `GET /healthz` | Collection size as JSON: `docs`, `shards`, `status`. |
| `POST /admin/reload` | Reconcile the served set with the shard directory: publish new files, retire gone ones. Returns `published`, `retired`, `shards`, `docs`. |
| `POST /admin/publish?shard=<name>` | Publish one named shard from the directory into the running server. |
| `POST /admin/retire?shard=<name>` | Retire one named shard from the running server. |

## tsumugi collection

```
tsumugi collection [command]
```

Lists, extends, and compacts a directory of `.tsumugi` shards.

### tsumugi collection list

```
tsumugi collection list <dir>
```

Lists the shards in a collection ordered by node base, showing each shard's file, its base in the global id space, its document count, and its size on disk.

### tsumugi collection add

```
tsumugi collection add <dir> [flags]
```

Adds a crawl export to an existing collection. It continues the global id space past the highest existing id and names its shards after the existing ones, so the new shards extend the collection rather than rewrite it.

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | | Crawl export to add (`.parquet` or `.jsonl`) |
| `--shard-size` | `50000` | Documents per shard |
| `--limit` | `0` | Cap documents read, zero for all |

### tsumugi collection compact

```
tsumugi collection compact <dir> [flags]
```

Merges a collection's shards into fewer, larger ones. It reads the documents back from each shard's forward store, reorders the whole set by host, and rebuilds into a staging directory, swapping it in only once every new shard is written, so a failed compact leaves the original collection untouched.

| Flag | Default | Meaning |
|------|---------|---------|
| `--shard-size` | `50000` | Documents per merged shard |

## tsumugi inspect

```
tsumugi inspect <file.tsumugi>
```

Prints a shard's header, its region table with on-disk and raw sizes per region, and its statistics. The read side of the format: it shows the version, document count, node base, capability flags, the regions the file carries, and the per-shard stats like the average document length and the term and token counts.

## tsumugi version

```
tsumugi version
```

Prints the version, the commit it was built from, and the build date.
