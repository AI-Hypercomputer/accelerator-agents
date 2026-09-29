---
title: "Retrospective — qwen3_cc5 / jax as of 2026-06-12 (#3, post-v043 frontier)"
type: analysis
tags: [retrospective, qwen3-cc5, jax]
model: qwen3_cc5
lane: jax
created: 2026-06-12
updated: 2026-06-12
---

# qwen3_cc5 / jax retrospective — 2026-06-12 (#3)

**Mode**: INCREMENTAL (prior: [#2](2026-06-12-qwen3_cc5-jax-retrospective-2.md), through v032; +10 new experiments v033–v043 deep-known)
**Scope**: 8B/v6e-8
**Period**: 2026-06-02 → 2026-06-12
**Experiments**: 41 total — 14 supported, 19 refuted, 4 inconclusive, 4 invalid
**Frontier**: **2,615.0 ms / 6,265 tok/s/chip / ≈36.0% MFU @ bs2/seq8192** (v043, 50-step certified; trunk `23aa5e7`) — MaxText gap **−9.9%**

## Prior-retrospective accountability (#2 → now)

| #2 recommendation | Outcome |
|---|---|
| #1 Manual weight prefetch (L) | **Refuted comprehensively** (v034–v037): unroll×offload = XLA S(5) bug (upstream-fileable); unroll-no-offload −3.24% (scheduler ignores the window); carry-prefetch +16 GB infeasible. Crucially, the target itself dissolved: the 624 ms AG bucket was a bs4+offload op-point artifact — it is **0.9 ms** at the current frontier. |
| #2 TP=2 probe (S) | **Refuted** (v033): −23.6%; tp collectives 20.2% of step; axis closed with a measured number. |
| #3 int8-AQT (L) | Still open — the sole surviving direction. |

Unplanned wins the probes surfaced (the retrospective's case for probing):
v037's *control arm* discovered bs2+save_attn (v038, +3.0%); v039's OOM
allocation table exposed the never-ablated attn_out save → save_qkv
(v041/v043, +0.7%, merged).

## Search-strategy assessment

- ✅ Frontier moved **3× since #2** (6,040 → 6,221 → 6,265). Probing-with-controls is paying.
- ✅ All single-axis spaces now closed with measured walls (see anti-recommendations).
- ⚠️ The remaining gap (−9.9%) decomposes into exactly two measured deltas vs MaxText at comparable shapes: **MXU efficiency** (55.5% vs 61.9%; our conv bucket is ~49% HBM-BW-util vs their 26.9% — layout/fusion difference, NOT matmul-tile quality) and **remat recompute** (250 ms loop-fusion at the frontier).
- 🔵 Unexplored axes (cross-checked vs taxonomy): int8/fp8 precision (#3), **fused QKV / fused gate-up projections** (MaxText has config options we never ported — fewer, larger matmuls is the canonical MXU-util lever and matches the measured conv-BW signature), CE-path alternatives at bs2, optimizer-state host offload (memory, not speed). A parallel MaxText code inventory is running to rank these (→ companion analysis).

## Mechanism tree (delta since #2; v033–v043)

- **Tensor parallelism** (1) — ❌ v033 (−23.6%; closed with measured number; stack tp-compatible)
- **FSDP/collectives — manual restructuring** (4) — ⚠️ v034 (XLA S(5) bug ×3 attempts; + stale-image-cache observation), ❌ v035 (pipeliner-less ACF ablation: flags worth ~0 — 21-flag standard; bug reproduces), ❌ v037 (unroll −3.24% no-offload; mechanism dead), [obs: the AG bucket itself was op-point-specific]
- **Remat / save-list** (4) — 🏆 v038 (bs2+save_attn 6,221, frontier), ❌ v039 (bs3+save_attn OOM by 612 MiB — wall measured), 🏆 v041-A/v043 (save_qkv: attn_out save unnecessary, grads bit-identical; **6,265 certified, merged**), ❌ v041-B (bs3+save_qkv fits but −12.6%/token — v6e bandwidth ridge)
- 🔵 **Fused QKV projection** — UNEXPLORED (MaxText `fused_qkv`; targets the 48.6% conv bucket's BW-util signature)
- 🔵 **Fused gate-up MLP** — UNEXPLORED (same class)
- 🔵 **int8 weight-only (AQT/qwix)** — UNEXPLORED (L; halves conv weight traffic)
- 🔵 **CE-path at bs2** — chunked-CE while-loop costs ~70 ms vs MaxText's plain CE (bucket-diff); per-chip f32 logits at bs2 = 9.9 GB — still infeasible plain; vocab-sharded variant remains L/deprioritized

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | **Fused QKV (+ fused gate-up) projections** | op-fusion | up to ~+2-4% (MXU util lever; matches measured conv-BW delta) | medium (pending code inventory) | M | ~0.5 | local-gap + MaxText config (`fused_qkv`); companion inventory in flight |
| 2 | **int8 weight-only AQT/qwix** | precision | ~+2 pp MFU | low-med | L | ~0.12 | retrospective #2 #3; taxonomy precision section |
| 3 | Optimizer-state host offload | memory | reopens batch rungs only (batch closed → low) | low | M | <0.1 | MaxText `optimizer_memory_host_offload` |

## Anti-recommendations (cumulative, all measured)

Batch (3 substrates: offload bs5 −0.5%, save_attn bs3 OOM-by-612-MiB,
save_qkv bs3 −12.6%/token); TP (−23.6%); splash kernel config (×3 flat;
our splash BEATS MaxText flash by 89 ms); AG-overlap flags (absent /
fire-and-hurt / no-op) and unroll/prefetch (XLA S(5) bug — file upstream,
revisit on libtpu update); ACF flags (worth 0); unrolled loop with offload
(2× liveness OOM); attn_out saving (strictly worse); host-offload at bs≤2
(no-offload wins); vmem >98304 (capped); QK-norm+RoPE Pallas (pre-filter
ceiling ≤1-2% at L).

## Cross-lane brief

- **qwen3-cc5-maxtext**: reference 6,953 @ bs3 (their recipe). Remaining transferables are now ARCHITECTURE-level (fused projections, AQT) — flag/remat/kernel parity is achieved or exceeded. Companion analysis (in flight) inventories the rest.
- **qwen3-cc5-torchax**: dormant at 19.2% @ seq2048; no inbound transfers.
- **llama3-8b-jax** (cross-model): its remaining unique lever (SC-offload flags) is compile-fatal on current libtpu (×2 confirmations).

## Experiment ledger (machine-readable)

Rows v000–v032 unchanged from [#2](2026-06-12-qwen3_cc5-jax-retrospective-2.md). New:

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| v033 | 2026-06-12 | 8B/v6e-8 | tensor-parallelism | refuted | no (axis closed: −23.6%) |
| v034 | 2026-06-12 | 8B/v6e-8 | fsdp-collectives | inconclusive | no (XLA S(5) bug; 3 attempts) |
| v035 | 2026-06-12 | 8B/v6e-8 | libtpu-xla-flags | refuted | no (ACF flags worth 0; bug reproduced) |
| v037 | 2026-06-12 | 8B/v6e-8 | fsdp-collectives | refuted | no (unroll −3.24%; control arm → v038) |
| v038 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | supported | **yes (6,221 / 35.7%)** |
| v039 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no (OOM by 612 MiB) |
| v041 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | refuted | no (arm A → v043; bs3 −12.6%/tok) |
| v043 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | supported | **yes (6,265 / ≈36.0%, certified, merged 23aa5e7)** |

## Sources

- Lane log; model page; [#2](2026-06-12-qwen3_cc5-jax-retrospective-2.md); [phase summary](2026-06-12-maxtext-gap-closing-phase-summary.md); [bucket diff](2026-06-12-maxtext-vs-jax-bucket-diff.md); experiment pages v033–v043.

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [MaxText feature comparison (companion, same date)](2026-06-12-maxtext-feature-gap-inventory.md)
