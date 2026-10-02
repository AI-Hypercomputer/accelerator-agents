---
title: "multi-shard sequence-parallel correction for linear recurrences"
type: concept
tags: [pallas, sequence-parallelism, linear-recurrence, lru, mamba, recurrentgemma, stub]
created: 2026-04-23
updated: 2026-04-23
---

Pattern for running a linear recurrence (LRU / Mamba / any `h_t = a_t · h_{t-1} + x_t`) under sequence parallelism: run the scan locally per shard, broadcast each shard's final `h` and `a_prod` across the sequence-axis group, then recompose the globally-correct `h_t` with one additional sweep. First-party reference: RecurrentGemma `pallas.py`. *Stub — expand when more sources are available.*

## Definition

Naive sequence parallelism is wrong for linear recurrences: each shard only has a prefix of the sequence, so its local `h` at shard boundaries doesn't reflect cross-shard dependency. The **multi-shard correction** computes the global `h_t` as a composition of (local scan on shard k) with (a prefix product of all earlier shards' carry-multiplier × their final state).

Algebraically: `h_t^global = a_prod_[0..k-1] · h_start^k + h_local^k`, where `a_prod_[0..k-1]` is the product of all earlier shards' cumulative `a` factors.

## Why it matters for TPU perf

Sequence-parallel training of Griffin / RG-LRU / Mamba-class models needs this correction or the output is wrong. Without it, linear recurrences are not sequence-parallelizable. The correction adds one extra cross-shard broadcast + one small pass — cheap compared to the savings from parallelizing the main scan.

## Mechanism

1. Each shard k runs `lru_pallas_scan` on its local sequence slice, producing `h_local^k` and local `a_prod^k = prod(a_t)` over the shard.
2. `lax.all_gather` `(h_local^k_end, a_prod^k)` across the sequence axis.
3. Compute `a_prod_[0..k-1] = prod_{j<k}(a_prod^j)` and `h_start^k = a_prod_[0..k-1] · h_init + sum_{j<k}(...)` — a cumulative-product prefix sweep.
4. Recombine: `h^global_t = a_prod_prefix^k · h_start^k + h_local_t^k`.

## When it applies / when it doesn't

- **Applies** whenever a linear recurrence (`h_t = a_t · h_{t-1} + b_t`) is sequence-parallelized across devices.
- **Does not apply** to attention (no carried state) or data-parallel scans (each device has the full sequence).

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `multi_shard_correction` in `recurrentgemma/jax/pallas.py` | [recurrentgemma](../codebases/recurrentgemma.md) | First-party reference; uses `sequence_shard_index` mesh axis |

## Connections

- [sequence-parallelism](sequence-parallelism.md)
- [scan-over-layers](scan-over-layers.md)
- [pallas-kernel](pallas-kernel.md)

## Sources

- [recurrentgemma codebase](../codebases/recurrentgemma.md) "Performance-relevant surfaces §2".
- [Pallas kernel directory §4.2](../analyses/pallas-kernel-directory/04-research-labs.md#42-google-deepmindrecurrentgemma).
