---
title: "Training a model"
description: "Fit a LambdaMART ranking model over a collection, and understand the bootstrap label that stands in until real relevance judgments exist."
weight: 20
---

The serve path ranks with a LambdaMART model: a gradient-boosted set of regression trees that produces the final order in the last stage of the cascade. `tsumugi train` fits one over a collection.

```bash
tsumugi train ./data --out ./data/model.bin
```

```
trained 200 trees over 20246 documents in 1266 queries, wrote ./data/model.bin
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--out` | | Model output file (required) |
| `--group-size` | `16` | Documents per synthetic query group |
| `--rounds` | `200` | Boosting rounds |

## The bootstrap label

Training a ranker needs labeled data: queries, documents, and a relevance grade per document. A fresh crawl has none of that. `train` solves the cold-start problem by fitting against a prior instead of real judgments.

It reads every shard's feature matrix, takes the static-rank feature (the prior the analyzer derives from body length and url depth), buckets it into a 0..4 graded label, and groups the documents into fixed-size synthetic queries. The model then learns to reconstruct that static-rank ordering from the other features. The result is a sane default ranking, not a learned relevance, and it is honest about being a prior: it is the model the serve command uses until you replace it with one trained on real judgments.

`--group-size` sets how many documents make up each synthetic query group, and `--rounds` sets how many boosting rounds (trees) to fit. More rounds fit the prior more tightly; the default of 200 is a good starting point.

## Moving past the prior

When you have real relevance data, whether human judgments or click logs, the path forward is to swap the label, not the pipeline. The trainer reads a dataset of feature rows, grades, and query groups; the bootstrap simply manufactures that dataset from the static-rank prior. Feeding it a dataset built from real judgments produces a model in the exact same format, and `serve` loads it the same way. The graded-relevance metrics the engine reports (NDCG and friends) are the yardstick for whether a new model beats the prior.

## How the model is used

The trained file is a set of trees. At serve time tsumugi compiles them into a vectorized scorer and evaluates it over each candidate's feature row in the rerank stage. Because the score is a pure function of the feature row, it does not depend on which shard a document lives in, which is what makes the broker's cross-shard top-k exact. The trainer and the serve path share the same tree representation, so a model that trains is a model that serves, with no conversion step.

## Next

With a model in hand, [serve the collection](/guides/serving-search/).
