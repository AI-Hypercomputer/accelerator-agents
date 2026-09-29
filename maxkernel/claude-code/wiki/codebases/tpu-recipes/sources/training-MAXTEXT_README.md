---
title: "Source: training/MAXTEXT_README.md"
type: source-page
doc: training/MAXTEXT_README.md
updated: 2026-07-03
---
# training/MAXTEXT_README.md

The shared MaxText environment-prep doc every MaxText leaf recipe points to for steps 1-4 of its
"Prep for MaxText" section: clone MaxText at a specific `tpu-recipes-v0.1.x` tag, install
dependencies into a `uv`-managed Python 3.12 virtualenv, build the Docker image from a pinned
`jax-stable-stack`/`jax-ai-image` base image, and push it to Container Registry
([prep-for-maxtext-workloads-on-gke](src:training/MAXTEXT_README.md#prep-for-maxtext-workloads-on-gke)).
Each leaf recipe then overrides only the tag and base image to pin its own JAX version. A
troubleshooting FAQ covers two common failure modes: a missing `python3-venv` package breaking `uv
venv` creation, and a pip hash-mismatch from a stale local cache during the Docker build
([faq](src:training/MAXTEXT_README.md#faq)).

## Feeds
- [MaxText training recipe pattern](../topics/maxtext-training-recipes.md) — the common
  environment-setup step every MaxText leaf recipe in that topic's comparison table shares.
