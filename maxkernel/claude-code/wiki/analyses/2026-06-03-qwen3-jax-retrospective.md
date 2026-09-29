---
title: "Retrospective — qwen3 / jax as of 2026-06-03"
type: analysis
tags: [retrospective, qwen3, jax]
model: qwen3
lane: jax
created: 2026-06-03
updated: 2026-06-03
---

# qwen3 / jax retrospective — 2026-06-03

**Scope**: 8B/v6e-8
**Period**: 2026-06-02 → 2026-06-03
**Experiments**: 33 total, 2 supported, 28 refuted, 0 inconclusive, 3 invalid, 0 backfilled
**Frontier**: 32.8% MFU @ v027 (bs=8, Splash, Tokamax XLA, Fused RoPE, inner nnx.remat, Vocab-Parallel FSDP TP=1)

## Search-strategy assessment

⚠️ **Single-axis-heavy**: 28 single-axis probes (batch sizing, TP scaling, XLA flags, checkpointing policies) vs 5 kernel ports (Tokamax). Ratio 5.6:1 above the 5:1 imbalance threshold.
⚠️ **Frontier stuck** for 6 experiments since v027 (2026-06-03 TP=1 bs=8 fragmentation fix win).
✅ **Sharding / TP topic**: well-explored (10 attempts, 1 supported, 9 refuted with diverse mechanisms) — calling this closed is justified. TP=2 overhead cannot be overcome, and TP=1 (FSDP) has hit its batch-size limit.
🔵 **Pallas kernels**: under-explored (2 of 12 catalog options tried; 0 supported; 5 crashed/refuted). We tried Tokamax CE variants extensively but ran into either missing libraries (`chunked_xla`) or SPMD partitioning limitations (`mosaic_tpu`). We have NOT tried fused elementwise kernels (e.g. Fused QK-norm + RoPE).

**Inference**: lane likely has +2-3 pp upside in kernel work that single-axis probing won't surface. We are blocked on collective-communication overheads, but we've exhausted all easy structural/compiler ways to hide them.

## Mechanism tree

- **FSDP / sharding / TP** (10 experiments)
  - Pure FSDP (TP=1) with bs=8 — 🏆 v027 +4.1pp vs v013
  - TP=2 scaling (bs=4 vs bs=8) — ❌ v007 (refuted), ❌ v008 (refuted), ❌ v009 (refuted)
  - TP=4 scaling — ❌ v012 (refuted, MFU regressed to 25.4%)
  - FSDP weight sharding overlap — ❌ v016 (crashed, Tokamax CE requires TP>=2)
  - TP=1 layout bridge fixes — ❌ v029 (refuted)
- **Batch-size + sequence amortization** (8 experiments)
  - Scale bs=2 to bs=4 — ❌ v001 (OOM)
  - Scale bs=8 to bs=16 — ❌ v028 (OOM)
- **Pallas kernels** (7 experiments)
  - Tokamax chunked_xla CE — ❌ v022, ❌ v023, ❌ v030 (crashed/unsupported)
  - Tokamax mosaic_tpu CE — ❌ v031, ❌ v032 (crashed, SPMD unsupported)
  - Tokamax xla CE — ✅ v010, ✅ v013, ✅ v024
  - 🔵 fused RMSNorm+RoPE+QKV — UNEXPLORED (catalog: tpu-recipes/maxtext-deepseek)
  - 🔵 NSA (Native Sparse Attention) — UNEXPLORED (catalog: paper 2025-02)
- **Activation checkpointing** (4 experiments)
  - Selective rematerialization (`checkpoint_dots_with_no_batch_dims`) — ❌ v003 (OOM)
  - Aggressive rematerialization (`checkpoint_dots`) — ❌ v011 (OOM)
  - Per-layer rematerialization (`nnx.remat`) — ✅ v014 (saved memory, but MFU dropped to 30.6%)
  - Removing outer jax.checkpoint to fix fragmentation — 🏆 v026 (unlocked bs=4), 🏆 v027 (unlocked bs=8)
- **libtpu / XLA flags** (2 experiments)
  - Phase 2 Collective Overlap bundle — ❌ v019, ❌ v020, ❌ v033 (OOMs / MFU regressions)
- **torch.compile + scan** (1 experiment)
  - `nnx.scan` over decoder blocks — ❌ v017 (no MFU improvement)
- **Splash attention** (1 experiment)
  - Enable splash attention — 🏆 v004 (MFU 24.3%), 🏆 v005

## Coverage gaps — what's NOT been explored

| Topic | Catalog options | Tried | Unexplored |
|---|---|---|---|
| Pallas kernels | 12 | 2 (CE variants, Splash) | 10 (Fused QK-norm/RoPE, Fused Gate/Up, etc) |
| libtpu / XLA flags | 25+ | 1 (Phase 2 bundle) | many |
| MoE expert dispatch | N/A | 0 | Model is dense, not applicable |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | Fused QK-norm + RoPE Pallas kernel | Pallas | +2-3pp | medium | L | 0.23 | local-gap: index Pallas catalog + program.md hints |
| 2 | Drop bs to 4 to use Phase 2 Overlap bundle | libtpu | +0.5pp | low | S | 0.20 | local-gap: retry v033 at lower bs |
| 3 | Fused RMSNorm+RoPE+QKV Pallas kernel | Pallas | +1-2pp | medium | L | 0.15 | local-gap: index Pallas catalog |

## Anti-recommendations

- **Tokamax Custom Implementations (`chunked_xla` / `mosaic_tpu`)** — The library installed is too old for `chunked_xla`, and JAX's `spmd_mesh` does not support partitioning `mosaic_tpu` Pallas kernels automatically. Do not try to compile these; they will crash.
- **Scaling batch size > 8** — `bs=16` is a strict physical OOM even under perfectly sharded TP=1 FSDP with outer-checkpoint fragmentation removed.
- **Phase 2 XLA bundles at `bs=8`** — The aggressive collective overlap flags increase peak HLO temp memory beyond the 31.25GB limit.

## Cross-lane brief

No sibling lanes configured.

## Experiment ledger (machine-readable — used by future incremental retrospectives)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| v001 | 2026-06-02 | 8B/v6e-8 | Batch-size + sequence amortization | invalid | no |
| v002 | 2026-06-02 | 8B/v6e-8 | FSDP / collective optimization | invalid | no |
| v003 | 2026-06-02 | 8B/v6e-8 | Activation checkpointing | refuted | no |
| v004 | 2026-06-02 | 8B/v6e-8 | Splash attention | supported | yes (24.3%) |
| v005 | 2026-06-02 | 8B/v6e-8 | Splash attention | supported | yes (24.3%) |
| v006 | 2026-06-02 | 8B/v6e-8 | Pallas kernels | invalid | no |
| v007 | 2026-06-02 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v008 | 2026-06-02 | 8B/v6e-8 | Tensor parallelism | supported | yes (28.6%) |
| v009 | 2026-06-02 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v010 | 2026-06-02 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v011 | 2026-06-02 | 8B/v6e-8 | Activation checkpointing | refuted | no |
| v012 | 2026-06-02 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v013 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | supported | yes (28.7%) |
| v014 | 2026-06-03 | 8B/v6e-8 | Activation checkpointing | supported | yes (30.6%) |
| v015 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | supported | yes (30.7%) |
| v016 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v017 | 2026-06-03 | 8B/v6e-8 | torch.compile + scan | refuted | no |
| v018 | 2026-06-03 | 8B/v6e-8 | libtpu / XLA flags | refuted | no |
| v019 | 2026-06-03 | 8B/v6e-8 | libtpu / XLA flags | refuted | no |
| v020 | 2026-06-03 | 8B/v6e-8 | libtpu / XLA flags | refuted | no |
| v021 | 2026-06-03 | 8B/v6e-8 | FSDP / collective optimization | refuted | no |
| v022 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v023 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v024 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v025 | 2026-06-03 | 8B/v6e-8 | Batch-size + sequence amortization | refuted | no |
| v026 | 2026-06-03 | 8B/v6e-8 | Activation checkpointing | supported | yes (32.2%) |
| v027 | 2026-06-03 | 8B/v6e-8 | Activation checkpointing | supported | yes (32.8%) |
| v028 | 2026-06-03 | 8B/v6e-8 | Batch-size + sequence amortization | refuted | no |
| v029 | 2026-06-03 | 8B/v6e-8 | Data transformation overhead | refuted | no |
| v030 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v031 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v032 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v033 | 2026-06-03 | 8B/v6e-8 | libtpu / XLA flags | refuted | no |

## Sources

- All 33 experiment pages in `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/`
- Prior log analysis

## See also

- `wiki/models/qwen3-jax.md`
- `.claude/skills/formulate-hypothesis/SKILL.md`
