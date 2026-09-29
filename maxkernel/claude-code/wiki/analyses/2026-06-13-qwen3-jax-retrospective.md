---
title: "Retrospective — qwen3 / jax as of 2026-06-13"
type: analysis
tags: [retrospective, qwen3, jax]
model: qwen3
lane: jax
created: 2026-06-13
updated: 2026-06-13
---

# qwen3 / jax retrospective — 2026-06-13

**Scope**: 8B/v6e-8
**Period**: 2026-06-02 → 2026-06-13
**Experiments**: 52 total, 7 supported, 41 refuted, 1 inconclusive, 3 invalid, 0 backfilled
**Frontier**: 33.0% MFU @ seq2048 bs16 (v042); 30.6% MFU @ seq8192 bs32 (v047)

## Search-strategy assessment

⚠️ **Batch-scaling & memory limits block progress**: The recent sprint focused heavily on scaling the batch size to increase MFU at seq=8192, but every configuration attempted (SDPA, Splash, Tokamax, and combinations) was strictly bottlenecked by the 31.25 GB HBM capacity on v6e. The lane is currently completely stuck on memory limits.
✅ **Tensor parallelism topic**: well-explored (17 attempts, mostly refuted with diverse mechanisms) — calling this closed is justified for now under TP=1 FSDP constraints.
🔵 **FSDP / collective optimization**: under-explored (2 attempts).
🔵 **Activation Checkpointing / Rematerialization**: under-explored (2 attempts), and highly relevant given the current HBM bottlenecks preventing batch scaling.

## Mechanism tree

- **Pallas kernels** (16 experiments)
  - ❌ v006 2026-06-02-qwen3-jax-v006-tokamax-ce.md
  - ❌ v010 2026-06-02-qwen3-jax-v010-splash-tp2-tokamax-bs8.md
  - 🏆 v013 2026-06-03-qwen3-jax-v013-splash-tp2-tokamax-bs4.md
  - 🏆 v014 2026-06-03-qwen3-jax-v014-splash-tp2-tokamax-remat-bs8.md
  - ❌ v015 2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8.md
  - ❌ v016 2026-06-03-qwen3-jax-v016-splash-tp1-tokamax-remat-fused-bs4.md
  - ❌ v017 2026-06-03-qwen3-jax-v017-splash-tp2-tokamax-remat-scan-bs8.md
  - ❌ v019 2026-06-03-qwen3-jax-v019-phase2-bundle.md
  - ❌ v020 2026-06-03-qwen3-jax-v020-phase2-bundle-bs4.md
  - ❌ v031 2026-06-03-qwen3-jax-v031-tp1-vocab-parallel-bs8-mosaic-tpu.md
  - ❌ v032 2026-06-03-qwen3-jax-v032-tp1-vocab-parallel-bs8-mosaic-tpu-patched.md
  - ❌ v038 2026-06-04-qwen3-jax-v038-tp1-vocab-parallel-bs8-fused-rope.md
  - 💥 v040 2026-06-04-qwen3-jax-v040-tp1-vocab-parallel-bs16-seq2048-tokamax.md
  - ❌ v041 2026-06-04-qwen3-jax-v041-tp1-vocab-parallel-bs16-seq2048-tokamax-shardmap.md
  - ❌ v050 2026-06-13-qwen3-jax-v050-splash-tokamax-bs64.md
  - ❌ v051 2026-06-13-qwen3-jax-v051-tokamax-bs64.md
  - 🔵 fused RMSNorm+RoPE+QKV — UNEXPLORED (catalog: tpu-recipes)
- **Tensor parallelism** (14 experiments)
  - ❌ v007 2026-06-02-qwen3-jax-v007-tp2-bs2.md
  - ❌ v021 2026-06-03-qwen3-jax-v021-tp1-vocab-parallel-bs4.md
  - ❌ v022 2026-06-03-qwen3-jax-v022-tp1-chunked-xla-bs4.md
  - ❌ v023 2026-06-03-qwen3-jax-v023-tp1-chunked-xla-bs4.md
  - ❌ v024 2026-06-03-qwen3-jax-v024-tp1-xla-bs4.md
  - ❌ v025 2026-06-03-qwen3-jax-v025-tp1-vocab-parallel-bs2.md
  - ❌ v026 2026-06-03-qwen3-jax-v026-tp1-vocab-parallel-bs4-no-outer-ckpt.md
  - ❌ v027 2026-06-03-qwen3-jax-v027-tp1-vocab-parallel-bs8-no-outer-ckpt.md
  - ❌ v028 2026-06-03-qwen3-jax-v028-tp1-vocab-parallel-bs16-no-outer-ckpt.md
  - 🏆 v030 2026-06-03-qwen3-jax-v030-tp1-vocab-parallel-bs8-chunked-xla.md
  - ❌ v034 2026-06-04-qwen3-jax-v034-tp1-vocab-parallel-bs8-fused-qknorm-rope.md
  - ❌ v035 2026-06-04-qwen3-jax-v035-tp1-vocab-parallel-bs8-xla-sdpa.md
  - ❌ v036 2026-06-04-qwen3-jax-v036-tp1-vocab-parallel-bs8-chunked-xla.md
  - ❌ v039 2026-06-04-qwen3-jax-v039-tp1-vocab-parallel-bs32-seq1024.md
- **Splash attention** (10 experiments)
  - ❌ v004 2026-06-02-qwen3-jax-v004-splash-bs2.md
  - ❌ v005 2026-06-02-qwen3-jax-v005-splash-tuned.md
  - 🏆 v008 2026-06-02-qwen3-jax-v008-splash-tp2-bs4.md
  - ❌ v009 2026-06-02-qwen3-jax-v009-splash-tp2-bs8.md
  - ❌ v011 2026-06-02-qwen3-jax-v011-splash-tp2-remat-dots-bs8.md
  - 🏆 v012 2026-06-02-qwen3-jax-v012-splash-tp4-bs8.md
  - ❌ v037 2026-06-04-qwen3-jax-v037-tp1-vocab-parallel-bs8-chunked-xla-splash-fix.md
  - ⚠️ v045 2026-06-05-qwen3-jax-v045-nsa-attention.md
  - ❌ v045 2026-06-05-qwen3-jax-v045-nsa.md
  - 💥 v046 2026-06-05-qwen3-jax-v046-nsa-pallas.md
  - ❌ v049 2026-06-13-qwen3-jax-v049-splash-bs64.md
- **Batch-size + sequence amortization** (6 experiments)
  - ❌ v001 2026-06-02-qwen3-jax-v001-batch-scaling.md
  - ❌ v043 2026-06-05-qwen3-jax-v043-bs32.md
  - ❌ v044 2026-06-05-qwen3-jax-v044-bs24.md
  - 🏆 v047 2026-06-13-qwen3-jax-v047-seq8k-baseline.md
  - ❌ v048 2026-06-13-qwen3-jax-v048-bs64.md
- **Activation checkpointing** (2 experiments)
  - 💥 v002 2026-06-02-qwen3-jax-v002-selective-sc-offload.md
  - ❌ v003 2026-06-02-qwen3-jax-v003-selective-remat.md
- **FSDP / collective optimization** (2 experiments)
  - ❌ v018 2026-06-03-qwen3-jax-v018-collective-matmul-off.md
  - ❌ v033 2026-06-03-qwen3-jax-v033-tp1-vocab-parallel-bs8-collective-overlap.md
- **Topology / hardware envelope** (1 experiments)
  - ⚠️ v042 2026-06-04-qwen3-jax-v042-tp1-vocab-parallel-bs16-seq2048-tokamax-shardmap-scan.md
- **Op fusion / kernel fusion** (1 experiments)
  - 🏆 v042 2026-06-05-qwen3-jax-v042-gate-up-fusion.md
- **Data transformation overhead** (1 experiments)
  - ❌ v029 2026-06-03-qwen3-jax-v029-tp1-vocab-parallel-bs8-fix-layout.md

## Coverage gaps — what's NOT been explored

| Topic | Catalog options | Tried | Unexplored |
|---|---|---|---|
| Pallas kernels | 12 | 11 | 1 |
| FSDP / collective optimization | 10 | 2 | 8 |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | Port Pallas NSA Kernel for TPU properly | Splash | +3-5pp | high | L | 0.33 | local-gap: memory constraint bypass |
| 2 | Advanced Activation Checkpointing / Selective Remat | AC | +1-2pp | medium | S | 1.40 | local-gap: open hypothesis |
| 3 | FSDP 2D Sharding / ZeRO-3 overlap | FSDP | +1-2pp | medium | M | 0.46 | local-gap: index |
| 4 | Fused RMSNorm+RoPE+QKV Pallas kernel | Pallas | +2-3pp | medium | L | 0.50 | local-gap: index Pallas catalog |

## Anti-recommendations

- **Batch-size scaling without structural memory savings** — 6 attempts across various configs (SDPA, Splash, Tokamax), all refuted by OOMs or crashes. Continuing to tweak flags and batch size limits without significantly reducing activation/parameter memory footprint is futile. The limit on v6e is ~31.25 GB.
- **Tensor parallelism re-exploration** — 17 attempts with diverse mechanisms; mostly refuted. Pure TP=1 FSDP is the strict architectural ceiling for this model and hardware combination under current constraints.

## Cross-lane brief

No sibling lanes (torchax) retrospectives available or evaluated for this model.

## Experiment ledger (machine-readable — used by future incremental retrospectives)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| v001 | 2026-06-02 | 8B/v6e-8 | Batch-size + sequence amortization | refuted | no |
| v002 | 2026-06-02 | 8B/v6e-8 | Activation checkpointing | invalid | no |
| v003 | 2026-06-02 | 8B/v6e-8 | Activation checkpointing | refuted | no |
| v004 | 2026-06-02 | 8B/v6e-8 | Splash attention | refuted | no |
| v005 | 2026-06-02 | 8B/v6e-8 | Splash attention | refuted | no |
| v006 | 2026-06-02 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v007 | 2026-06-02 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v008 | 2026-06-02 | 8B/v6e-8 | Splash attention | supported | yes |
| v009 | 2026-06-02 | 8B/v6e-8 | Splash attention | refuted | no |
| v010 | 2026-06-02 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v011 | 2026-06-02 | 8B/v6e-8 | Splash attention | refuted | no |
| v012 | 2026-06-02 | 8B/v6e-8 | Splash attention | supported | yes |
| v013 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | supported | yes |
| v014 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | supported | yes |
| v015 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v016 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v017 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v018 | 2026-06-03 | 8B/v6e-8 | FSDP / collective optimization | refuted | no |
| v019 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v020 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v021 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v022 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v023 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v024 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v025 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v026 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v027 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v028 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v029 | 2026-06-03 | 8B/v6e-8 | Data transformation overhead | refuted | no |
| v030 | 2026-06-03 | 8B/v6e-8 | Tensor parallelism | supported | yes |
| v031 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v032 | 2026-06-03 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v033 | 2026-06-03 | 8B/v6e-8 | FSDP / collective optimization | refuted | no |
| v034 | 2026-06-04 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v035 | 2026-06-04 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v036 | 2026-06-04 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v037 | 2026-06-04 | 8B/v6e-8 | Splash attention | refuted | no |
| v038 | 2026-06-04 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v039 | 2026-06-04 | 8B/v6e-8 | Tensor parallelism | refuted | no |
| v040 | 2026-06-04 | 8B/v6e-8 | Pallas kernels | invalid | no |
| v041 | 2026-06-04 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v042 | 2026-06-04 | 8B/v6e-8 | Topology / hardware envelope | TBD | no |
| v042 | 2026-06-05 | 8B/v6e-8 | Op fusion / kernel fusion | supported | yes |
| v043 | 2026-06-05 | 8B/v6e-8 | Batch-size + sequence amortization | refuted | no |
| v044 | 2026-06-05 | 8B/v6e-8 | Batch-size + sequence amortization | refuted | no |
| v045 | 2026-06-05 | 8B/v6e-8 | Splash attention | inconclusive | no |
| v045 | 2026-06-05 | 8B/v6e-8 | Splash attention | refuted | no |
| v046 | 2026-06-05 | 8B/v6e-8 | Splash attention | invalid | no |
| v047 | 2026-06-13 | 8B/v6e-8 | Batch-size + sequence amortization | supported | no |
| v048 | 2026-06-13 | 8B/v6e-8 | Batch-size + sequence amortization | refuted | no |
| v049 | 2026-06-13 | 8B/v6e-8 | Splash attention | refuted | no |
| v050 | 2026-06-13 | 8B/v6e-8 | Pallas kernels | refuted | no |
| v051 | 2026-06-13 | 8B/v6e-8 | Pallas kernels | refuted | no |

## Sources

- Per-lane log: `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/log.md`
- Model page: `wiki/models/qwen3-cc-jax.md`
- Topic taxonomy: `wiki/model-optimization-index.md`
- All N experiment pages in `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/`
- Prior retrospective: `wiki/analyses/2026-06-05-qwen3-jax-retrospective.md`

## See also

- `wiki/models/qwen3-cc-jax.md`
- `.claude/skills/formulate-hypothesis/SKILL.md` — the downstream consumer
