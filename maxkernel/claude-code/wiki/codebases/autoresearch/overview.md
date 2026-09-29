---
title: autoresearch — overview
type: overview
tags: [autoresearch, gpt, pretraining, single-gpu, h100, mfu, bpb, flash-attention]
created: 2026-07-04
updated: 2026-07-04
---
# autoresearch — what it is and how it fits together

## In one paragraph
`autoresearch` is Karpathy's minimal single-GPU GPT-pretraining reference: two small Python files
that train a language model for a **fixed wall-clock time budget** on one H100 and report a single
comparable number — validation **bits-per-byte (BPB)** — alongside a hardware-efficiency readout
(**MFU**). It exists to make optimization experiments *comparable*: the metric is frozen, the
validation set is pinned and excluded from training, and the first warmup/compile steps are excluded
from timing, so any change's effect on quality-per-second is measured cleanly. It is the
methodological model for this wiki's own loop — the same "one number, held-constant harness, measure
the delta" discipline, scaled down to a single accelerator. Everything lives in two concerns:
[`prepare`](concepts/prepare.md) builds the data + tokenizer and the runtime data plane, and
[`train`](concepts/train.md) is the time-budgeted loop that consumes them.

## Core architecture
```mermaid
flowchart LR
  subgraph prep["prepare.py (data plane)"]
    SH["download shards"]:::io --> TOK["train BPE tokenizer"]:::io
    TOK --> DL["packed dataloader<br/>(best-fit, 100% token util)"]:::io
    VAL["pinned last-shard<br/>validation set"]:::io --> BPB["frozen BPB metric"]:::metric
  end
  subgraph tr["train.py (time-budgeted loop)"]
    LOOP["train to TIME_BUDGET<br/>(warmup steps excluded)"]:::loop
    FA["FA3 flash-attention<br/>(causal + sliding window, QK-norm)"]:::perf
    MFU["FLOP accounting → MFU /<br/>steady_state_mfu vs H100 bf16 peak"]:::perf
    LOOP --> FA
    LOOP --> MFU
  end
  DL --> LOOP
  BPB --> LOOP
  classDef io fill:#e0f7fa,stroke:#00acc1;
  classDef metric fill:#fff7e0,stroke:#f9ab00;
  classDef loop fill:#e6f4ea,stroke:#34a853;
  classDef perf fill:#fce8e6,stroke:#ea4335;
```

Node → concept page: [data plane + metric](concepts/prepare.md) · [time-budgeted loop + MFU](concepts/train.md).

## Main concepts

### The time-budgeted training loop
[`train`](concepts/train.md) trains for a fixed **wall-clock budget**, not a step count, and reports
one number (val BPB) plus an MFU readout. The perf-load-bearing surfaces: the **FA3 flash-attention**
kernel (causal + per-layer sliding window, QK-norm), window-aware **FLOP accounting** feeding
measured `mfu` / `steady_state_mfu` against a hard-coded H100 bf16 peak, bf16 mixed precision, and
the deliberate exclusion of the first ~10 compile/warmup steps from both timing and MFU. See
[train](concepts/train.md).

### The data plane and the frozen metric
[`prepare`](concepts/prepare.md) has two lives: a one-time **shard download + BPE tokenizer**
training, and the runtime data plane `train.py` imports — a **best-fit packed dataloader** (100%
token utilization, no padding) with a pinned host buffer + `non_blocking` H2D copy into preallocated
CUDA buffers. Its guardrails are what make experiments comparable: the pinned last-shard validation
set is excluded from both training data and tokenizer statistics, and the byte-normalized **BPB**
metric is frozen. See [prepare](concepts/prepare.md).

## Map of the wiki
- **How is a run scored / what's the harness discipline?** → [train](concepts/train.md) (time budget, BPB, MFU) + [prepare](concepts/prepare.md) (pinned val set, frozen BPB).
- **Where is the attention kernel / MFU accounting?** → [train](concepts/train.md).
- **How is data fed (packing, tokenizer, H2D)?** → [prepare](concepts/prepare.md).
- **What is symbol X — signature, callers?** → the per-module structural index under [`catalog/`](catalog/).
