---
title: "autoresearch (Karpathy)"
type: codebase
tags: [methodology, reference]
commit: 228791fb499afffb54b46200aca536f79142f117
created: 2026-04-22
updated: 2026-04-22
---

Karpathy's reference implementation of the autonomous research loop: a single H100, a simplified nanochat pretraining setup, a fixed 5-minute wall-clock budget, and `val_bpb` as the single metric. An agent autonomously edits `train.py`, runs a 5-minute experiment, and advances or rewinds the branch based on whether `val_bpb` improved. **This wiki ingests autoresearch as a methodology reference only** — there is no TPU-specific content in the repo. What we take from it is the *shape* of the loop (agent-editable file, human-editable prompt, fixed budget, fair metric, append-only results log). The wiki's own SCHEMA.md is an adaptation of that pattern to TPU performance optimization.

## Overview

The repo is deliberately tiny: five text files plus a lockfile and a teaser image. It is not infrastructure; it is a demonstration. The agent is pointed at `program.md`, reads `README.md`, `prepare.py`, and `train.py`, and then begins an indefinite loop of: edit `train.py` -> `uv run train.py > run.log 2>&1` -> grep `val_bpb` -> keep (commit advances) or discard (`git reset`). Experiments are logged to an untracked `results.tsv`. One experiment per ~5 minutes, ~100 per overnight sleep.

The repo's only hardware assumption is a single NVIDIA GPU (H100 by default); the metric `val_bpb` (bits per byte) is deliberately vocab-size-independent so that architectural changes stay comparable. **None of this is TPU content.** We are borrowing the *protocol*, not the code.

## Architecture

Three-role split:

- **Human role** — edits `program.md`. Sets the scope, the metric, the rules, the logging format. Not touched during a run.
- **Agent role** — edits `train.py`. Everything within that file is fair game: model architecture, optimizer, hyperparameters, batch size. No other file may be edited.
- **Fixed role** — `prepare.py` and the evaluation harness inside it. Read-only. Defines the data, the tokenizer, `MAX_SEQ_LEN`, `TIME_BUDGET`, and `evaluate_bpb` — the ground-truth metric.

The loop is explicit prose in `program.md` under "The experiment loop": commit -> run -> parse -> record -> advance-or-reset.

## Key abstractions

- **Run tag / branch** — every autonomous run lives on its own `autoresearch/<tag>` branch so history is preserved and comparisons are traceable.
- **Fixed time budget** (`TIME_BUDGET = 300` seconds in `prepare.py`) — every experiment trains for the same wall-clock duration, excluding startup/compilation. This makes runs with different model sizes, batch sizes, and architectures directly comparable.
- **Single-number metric** (`val_bpb` from `evaluate_bpb` in `prepare.py`) — lower is better, vocab-independent, computed on a pinned validation shard.
- **`results.tsv`** — tab-separated, append-only, not tracked by git. Columns: `commit`, `val_bpb`, `memory_gb`, `status` (`keep`/`discard`/`crash`), `description`.
- **Simplicity criterion** — a change that improves the metric but adds ugly code may still be discarded; a change that simplifies without regressing is kept.

## Entry points

- [raw/code/autoresearch/program.md](../../raw/code/autoresearch/program.md) — agent's entry point. Sets up the branch, establishes the baseline, then enters the forever loop.
- [raw/code/autoresearch/prepare.py](../../raw/code/autoresearch/prepare.py) — one-time data prep: `uv run prepare.py` downloads shards and trains the BPE tokenizer into `~/.cache/autoresearch/`.
- [raw/code/autoresearch/train.py](../../raw/code/autoresearch/train.py) — the experiment: `uv run train.py > run.log 2>&1`.

## Dependencies

PyTorch 2.9.1 on CUDA 12.8; `kernels` for Hopper FlashAttention-3; `rustbpe` + `tiktoken` for the BPE tokenizer; `pyarrow` for shard reads. Pinned in [pyproject.toml](../../raw/code/autoresearch/pyproject.toml). No distributed runtime, no config system, no CLI flags — the agent edits module-level constants directly.

## Notable files

- [README.md](../../raw/code/autoresearch/README.md) — framing, design choices, platform caveats.
- [program.md](../../raw/code/autoresearch/program.md) — the "skill" that drives the agent. Treat this as the canonical reference for the loop shape.
- [train.py](../../raw/code/autoresearch/train.py) — 630 lines, one file: GPT model, MuonAdamW optimizer, training loop, FA3 attention, `torch.compile` wrap. All hyperparameters live as top-level constants (`DEPTH`, `TOTAL_BATCH_SIZE`, `MATRIX_LR`, etc. around line 432-451) so the agent can edit without plumbing.
- [prepare.py](../../raw/code/autoresearch/prepare.py) — fixed: data, tokenizer, `MAX_SEQ_LEN = 2048`, `TIME_BUDGET = 300`, `evaluate_bpb`.
- `analysis.ipynb` — notebook for inspecting `results.tsv`.

## Structural surfaces we borrow

autoresearch is methodological content, not TPU-perf content, so there are no knobs, kernels, or flags to enumerate. Instead, this is what the wiki's own protocol inherits from it:

- **Agent/human file split.** `program.md` is human-edited (scope, rules, metric); `train.py` is agent-edited (the artifact under optimization). This wiki mirrors the split as `SCHEMA.md` (human) versus the `wiki/**` pages the agent writes. See [program.md](../../raw/code/autoresearch/program.md) section "What you CAN do / CANNOT do".
- **Fair-comparison metric convention.** One primary number, vocab/config-independent, explicitly named as the ground truth (`val_bpb`). For this wiki the primary number is different — step time / MFU / tokens-per-second — and `val_bpb` is explicitly *not* one of our metrics, but the discipline of naming one ground-truth metric and pinning its harness is taken from here. See [prepare.py](../../raw/code/autoresearch/prepare.py) `evaluate_bpb` and the "DO NOT CHANGE" comment above it.
- **Wall-clock budget idea.** autoresearch's fixed 5-minute budget makes experiments directly comparable across architectural changes. This wiki adapts the idea — a TPU experiment has a declared step-count or wall-clock bound on each experiment page so its profile and headline number are interpretable. See [program.md](../../raw/code/autoresearch/program.md) section "Experimentation" and the `TIME_BUDGET` constant in [prepare.py](../../raw/code/autoresearch/prepare.py).
- **Iteration-log format.** `results.tsv` with `commit | metric | memory | status | description` is the minimal viable experiment log. This wiki keeps a richer log (`wiki/log.md` for events, per-experiment pages with tables and profile links) but the header row — commit, metric, memory, verdict, description — is the same five-column skeleton. See [program.md](../../raw/code/autoresearch/program.md) section "Logging results".
- **Log-of-experiments / keep-or-discard discipline.** Every run gets a verdict (`keep`, `discard`, `crash`); keeps advance the branch, discards `git reset`. This wiki generalises the three verdicts to `supported`, `refuted`, `inconclusive`, `invalid` but the norm — every run resolves to one verdict, no silent drops — is taken from here. See [program.md](../../raw/code/autoresearch/program.md) section "The experiment loop" steps 7-9.

## Connections

- [SCHEMA.md](../../SCHEMA.md) — this wiki's operating rules, explicitly an adaptation of the autoresearch pattern.
- The `experiment`, `hypothesis`, and `model` page templates in `SCHEMA.md` owe their keep/discard/verdict shape to this repo.
- Divergences from autoresearch that matter for this wiki:
  - Metric axis differs — this wiki tracks step time, MFU, tokens/sec, and peak HBM. `val_bpb` and any model-quality metric are **out of scope** (SCHEMA.md §Scope).
  - Surface area differs — we optimise compiler flags, parallelism, rematerialisation, attention kernels, layouts, fusions, scheduling, and precision; autoresearch optimises hyperparameters and model architecture.
  - Semantics check is mandatory — because our optimisations must not change the model, a `supported` verdict requires a loss-trajectory or output-parity check that autoresearch does not need (autoresearch is allowed to change what the model is).
  - Ingested knowledge is first-class — this wiki has `sources/`, `concepts/`, `observations/`, `hypotheses/`, `analyses/` directories on top of an autoresearch-style experiment loop. autoresearch has only `results.tsv`.

## See also

- [SCHEMA.md](../../SCHEMA.md)

## Sources

- [raw/code/autoresearch/README.md](../../raw/code/autoresearch/README.md)
- [raw/code/autoresearch/program.md](../../raw/code/autoresearch/program.md)
- [raw/code/autoresearch/train.py](../../raw/code/autoresearch/train.py)
- [raw/code/autoresearch/prepare.py](../../raw/code/autoresearch/prepare.py)
- [raw/code/autoresearch/pyproject.toml](../../raw/code/autoresearch/pyproject.toml)
- [Ultra-Scale Playbook (2025)](../sources/2025-ultrascale-playbook.md) — Section 8 ("Finding the Best Training Configuration") describes essentially the same search procedure — fit-in-memory → target-GBS → maximise-throughput, benchmarked over thousands of configs — that this wiki's loop is designed to execute for TPU.
