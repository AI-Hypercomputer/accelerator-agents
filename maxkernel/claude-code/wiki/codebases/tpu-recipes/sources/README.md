---
title: "Source: README.md"
type: source-page
doc: README.md
updated: 2026-07-03
---
# README.md

The repository's top-level orientation doc. tpu-recipes is a collection of recipes reproducing
specific, point-in-time **benchmark** measurements (throughput, step time) on Cloud TPUs — for
customers/partners validating hardware performance or informing purchasing decisions, not
general-purpose tutorials
([intended-audience](src:README.md#intended-audience)). It lays out the repo's four top-level
directories — `training/`, `inference/`, `microbenchmarks/`, `utils/` — and the workflow for using
them: pick a model/TPU/framework combination, navigate to the matching recipe directory, and follow
its procedure end to end
([repository-organization](src:README.md#repository-organization),
[how-to-use-these-recipes](src:README.md#how-to-use-these-recipes)). Benchmarks are point-in-time and
not repeated once complete ([maintenance-policy](src:README.md#maintenance-policy)); the actual
confidential benchmark report itself is out of scope for this repo
([repository-scope](src:README.md#repository-scope)).

## Feeds
- [tpu-recipes overview](../overview.md) — this doc is the root orientation page the overview
  builds on directly.
