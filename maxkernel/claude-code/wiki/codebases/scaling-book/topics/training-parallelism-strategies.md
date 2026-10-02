---
title: "Topic: Training parallelism strategies"
type: topic
updated: 2026-07-03
---
# Training parallelism strategies

## Overview

Given a model size and chip count, the book derives closed-form communication-bound thresholds for
each core training parallelism strategy — data parallelism, FSDP, and tensor parallelism — by
applying [Sharding notation and collective communication](sharding-notation-and-collectives.md)'s
collective-cost formulas to each strategy's required AllGather/AllReduce/ReduceScatter pattern
([what-do-we-mean-by-scaling](src:training.md#what-do-we-mean-by-scaling)). LLaMA 3-70B on a full
TPU v5p pod is the book's worked example applying every formula to a real model
([how-to-shard-llama-3-70b-for-training](src:applied-training.md#how-to-shard-llama-3-70b-for-training)).

## Diagram

```mermaid
flowchart TD
  DP["Data Parallelism / FSDP"] -->|"comms-bound when per-device batch < 2550/M_X"| DPBOUND["bandwidth-bound"]
  TP["Tensor Parallelism"] -->|"comms-bound when Y > M_Y*F/2550"| TPBOUND["typically 8-16 way max"]
  COMBO["FSDP + Tensor Parallelism combined"] -->|"optimal FSDP degree X_opt = sqrt((B/F)(M_X/M_Y)N)"| SMALLBATCH["handles per-chip batch as low as 2550^2/2F"]
  LLAMA["LLaMA 3-70B on full v5p pod"] -->|"1024-way DP + 2-way seq-parallel + 4-way tensor-parallel"| RESULT["4M-token batch, not comms-bound"]
```

## Key results

**Both pure data parallelism and FSDP become bandwidth-bound on TPU v5 once per-device batch size
drops below $2550/M_X$** ($M_X$ = number of mesh axes used)
([fully-sharded-data-parallelism-fsdp](src:training.md#fully-sharded-data-parallelism-fsdp)).
Counter-intuitively, *smaller* total batch size (at fixed chip count) makes you *more*
communication-bound — but larger batch sizes give diminishing gradient-noise-reduction returns and
can cause training instability, so real recipes start from a scaling-law-determined batch size and
a fixed (large) chip count, then search for a partitioning that fits.

**Tensor parallelism becomes communication-bound when $Y > M_Y \cdot F/2550$ — typically an 8-16-way
ceiling for most model sizes**
([tensor-parallelism](src:training.md#tensor-parallelism)). This is a much tighter ceiling than
FSDP's, which is why tensor parallelism is almost always combined with FSDP/data parallelism rather
than used alone at scale.

**Combining FSDP with tensor parallelism drops the viable per-chip batch size floor to
$2550^2/(2F)$ — about 8x lower than FSDP could achieve alone**, letting you scale to more chips for
the same batch size before hitting the communication-bound wall
([combining-fsdp-and-tensor-parallelism](src:training.md#combining-fsdp-and-tensor-parallelism)).
The optimal FSDP degree in this combined regime is $X_{opt} = \sqrt{(B/F)(M_X/M_Y)N}$
([combining-fsdp-and-tensor-parallelism](src:training.md#combining-fsdp-and-tensor-parallelism)).

**Context/sequence parallelism is "just data parallelism over the sequence dimension" for the MLP —
tokens are tokens, regardless of which sequence they belong to — but attention needs explicit
cross-sequence handling (ring attention), gathering KVs or Qs with careful FLOPs/comms overlap**
([what-do-we-mean-by-scaling](src:training.md#what-do-we-mean-by-scaling)). This reframing is why
sequence parallelism can be folded into the same "batch size" accounting as data parallelism for
every non-attention computation.

**LLaMA 3-70B's 3 largest MLP weight matrices (56B of 70B total params) dominate its FLOPs/memory
accounting almost entirely** — nearly every other parameter can be ignored for back-of-envelope
estimates
([counting-parameters-and-flops](src:applied-training.md#counting-parameters-and-flops)). Applying
the formulas above, the concrete recipe that avoids being communication-bound at a 4M-token batch on
a full TPU v5p pod is **1024-way data parallelism × 2-way sequence parallelism × 4-way tensor
parallelism** — pure FSDP or FSDP+sequence-parallelism alone *would* be communication-bound at this
scale
([how-to-shard-llama-3-70b-for-training](src:applied-training.md#how-to-shard-llama-3-70b-for-training)).

## See also
- [Sharding notation and collective communication](sharding-notation-and-collectives.md) — the
  collective-cost primitives every threshold here is derived from.
- [Transformer FLOPs and memory accounting](transformer-flops-and-memory-accounting.md) — the
  parameter/FLOPs counts these formulas are applied to.
- [Inference serving, latency and throughput](inference-serving-latency-throughput.md) — the
  inference-side counterpart, where the optimal parallelism strategy differs because generation is
  memory-bandwidth-bound rather than compute-bound.

## Sources
- [training.md](../sources/training.md)
- [applied-training.md](../sources/applied-training.md)
