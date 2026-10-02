---
title: "scaling-book — How To Scale Your Model"
type: codebase
tags: [book, reference]
commit: 6cda371aa898f6ef1de92d617e2fe5bbff45aaa5
created: 2026-04-22
updated: 2026-04-22
---

> **Grounded front door → [scaling-book overview](scaling-book/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

`scaling-book` is the Jekyll-based source for the book *How To Scale Your Model: A Systems View of LLMs on TPUs* by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, and Reiner Pope (Google DeepMind, 2025). It is **prose, not a library** — the wiki value is in the individual chapters, each of which will be ingested as its own `source` page in a later wave. This page is a thin index that frames the repo and maps chapters to their planned source-page slugs.

## Overview

The repo is a Jekyll site that renders to `https://jax-ml.github.io/scaling-book`. Each chapter is a top-level `*.md` file with Distill-style frontmatter, built by `bundle exec jekyll serve`. There is no runtime code to ingest here — the eleven numbered chapters plus a conclusion are the content. Treat this codebase page as a navigation shim: the per-chapter source pages are where claims, equations, roofline formulas, and parallelism recipes get extracted and linked to concepts and hypotheses.

## Architecture

Flat Jekyll site. Chapters live at the repo root as Markdown with Liquid includes (`{% include figure.liquid %}`) and Distill citation/footnote extensions. `_layouts/`, `_includes/`, `_sass/`, and `_plugins/` are theme machinery from the al-folio Jekyll theme and carry no perf content.

## Key abstractions

None in the code sense. The conceptual abstractions the book develops — roofline analysis, sharded matmul notation, the 4 parallelism axes (data, tensor, pipeline, expert), memory-saving techniques (rematerialization, ZeRO, host offload, gradient accumulation) — live in the chapter prose and will get `concept` pages when ingested.

## Entry points

- [`index.md`](../../raw/code/scaling-book/index.md) — landing page with the full table of contents and chapter blurbs.
- [`README.md`](../../raw/code/scaling-book/README.md) — build instructions and citation.
- [`bin/convert_to_single_md.py`](../../raw/code/scaling-book/bin/convert_to_single_md.py) — concatenates all chapters into `scaling-book-combined.md` (useful for bulk ingest).

## Dependencies

Ruby 3.4.5+, Jekyll (see [`Gemfile`](../../raw/code/scaling-book/Gemfile)), ImageMagick, Jupyter, the al-folio Distill theme. Build-only — nothing runtime-relevant to TPU performance.

## Notable files

- [`_config.yml`](../../raw/code/scaling-book/_config.yml) — site config, baseurl `/scaling-book`.
- [`conclusion.md`](../../raw/code/scaling-book/conclusion.md) — further-reading list; useful for discovering follow-on sources to ingest.
- [`assets/img/`](../../raw/code/scaling-book/assets/) — figures referenced by chapters (roofline plots, transformer diagrams, sharding cartoons).

## Performance-relevant surfaces

The "surfaces" here are the chapters themselves. Each will be ingested as a standalone source page; this table is the canonical mapping used to back-link from those pages.

| # | Chapter | Repo file | 1-line description | Planned source slug |
|---|---|---|---|---|
| 1 | A Brief Intro to Roofline Analysis | [`roofline.md`](../../raw/code/scaling-book/roofline.md) | Algorithms are bounded by compute, communication, and memory; roofline analysis approximates how fast they run. | `sources/2025-scaling-book-roofline.md` |
| 2 | How to Think About TPUs | [`tpus.md`](../../raw/code/scaling-book/tpus.md) | How TPUs work as chips and how that constrains which models can train and serve on them. | `sources/2025-scaling-book-tpus.md` |
| 3 | Sharded Matrices and How to Multiply Them | [`sharding.md`](../../raw/code/scaling-book/sharding.md) | Model sharding and multi-TPU parallelism explained via sharded matrix multiplication. | `sources/2025-scaling-book-sharding.md` |
| 4 | All the Transformer Math You Need to Know | [`transformers.md`](../../raw/code/scaling-book/transformers.md) | Counts parameters, FLOPs, and KV-cache sizes for Transformer forward and backward passes. | `sources/2025-scaling-book-transformers.md` |
| 5 | How to Parallelize a Transformer for Training | [`training.md`](../../raw/code/scaling-book/training.md) | FSDP, Megatron sharding, and pipeline parallelism — picking the efficient scheme for a given chip count and batch. | `sources/2025-scaling-book-training.md` |
| 6 | Training LLaMA 3 on TPUs | [`applied-training.md`](../../raw/code/scaling-book/applied-training.md) | Applied walkthrough of training LLaMA 3 on TPUs — time, cost, parallelism choice. | `sources/2025-scaling-book-applied-training.md` |
| 7 | All About Transformer Inference | [`inference.md`](../../raw/code/scaling-book/inference.md) | Inference adds latency constraints and reshapes memory; covers disaggregated serving and KV caches. | `sources/2025-scaling-book-inference.md` |
| 8 | Serving LLaMA 3 on TPUs | [`applied-inference.md`](../../raw/code/scaling-book/applied-inference.md) | Applied walkthrough of serving LLaMA 3 on TPU v5e — latency/throughput tradeoffs and cost. | `sources/2025-scaling-book-applied-inference.md` |
| 9 | How to Profile TPU Code | [`profiling.md`](../../raw/code/scaling-book/profiling.md) | JAX + XLA stack and the JAX/TensorBoard profiler for debugging real performance issues. | `sources/2025-scaling-book-profiling.md` |
| 10 | Programming TPUs in JAX | [`jax-stuff.md`](../../raw/code/scaling-book/jax-stuff.md) | JAX parallelism APIs (shard_map, jit, pjit) with worked examples. | `sources/2025-scaling-book-jax-stuff.md` |
| 11 | How to Think About GPUs | [`gpus.md`](../../raw/code/scaling-book/gpus.md) | Bonus chapter on NVIDIA GPUs — how they work, how they're networked, how their rooflines differ from TPUs. | `sources/2025-scaling-book-gpus.md` |

Conclusion material lives in [`conclusion.md`](../../raw/code/scaling-book/conclusion.md); it will be folded into whichever chapter source page cites it rather than getting its own page.

## Connections

No concept, hypothesis, or experiment pages reference this codebase yet. Once chapter source pages are filed, expected connections include roofline analysis, FSDP, tensor/Megatron parallelism, pipeline parallelism, expert parallelism, rematerialization, ZeRO/optimizer sharding, host offload, gradient accumulation, KV-cache layout, disaggregated serving, and the JAX profiler workflow.

## See also

- (none yet — chapter source pages pending)

## Sources

- [`raw/code/scaling-book/README.md`](../../raw/code/scaling-book/README.md)
- [`raw/code/scaling-book/index.md`](../../raw/code/scaling-book/index.md)
- [`raw/code/scaling-book/_config.yml`](../../raw/code/scaling-book/_config.yml)
- [Ultra-Scale Playbook (2025)](../sources/2025-ultrascale-playbook.md) — parallel GPU-side reference covering much of the same ground (5D parallelism, FSDP/ZeRO, FlashAttention, mixed precision, rematerialization, overlap math); cross-check chapter-by-chapter when scaling-book chapters are ingested in Wave 3.
