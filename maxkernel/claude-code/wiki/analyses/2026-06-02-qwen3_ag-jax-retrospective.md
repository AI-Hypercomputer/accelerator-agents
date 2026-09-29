---
title: "Retrospective — qwen3_ag / jax as of 2026-06-02"
type: analysis
tags: [retrospective, qwen3_ag, jax]
model: qwen3_ag
lane: jax
created: 2026-06-02
updated: 2026-06-02
---

# qwen3_ag / jax retrospective — 2026-06-02

**Scope**: 8B/v6e-8 variant
**Period**: 2026-06-02 → 2026-06-02
**Experiments**: 5 total, 2 supported, 2 refuted, 1 invalid, 0 backfilled
**Frontier**: 20.0% MFU @ seq 2048 bs 1 (v000-repro)

## Search-strategy assessment

⚠️ **Single-axis-probe focus**: 3 memory/configuration experiments tried vs 0 custom kernel ports on this lane.
⚠️ **Frontier stuck**: Frontier has not moved past the batch size 1 baseline reproduction (`v000-repro` at 20.0% MFU). Batch size scaling (bs=2) remains completely blocked by compilation-time HBM Out-of-Memory.
✅ **Topic closure**: Pure batch size scaling without memory optimization is definitively closed (refuted by `v001` with a huge 42.99 GB footprint).
🔵 **Under-explored topics**: Custom Pallas kernels, Splash attention, and fused layers have 0% coverage on this lane.

**Inference**: The lane is currently "memory-stuck" trying to scale batch size. Traditional single-axis batch-size scaling is completely unfeasible. However, the selective rematerialization work in `v003` proved extremely promising, slashing activation memory by 9.38 GB (from 42.99 GB to 33.61 GB). This indicates we are only 2.36 GB of HBM away from unlocking `bs=2`. A combination of selective rematerialization and a memory-saving attention kernel (like Splash Attention) or TP=2 sharding is highly likely to unblock the frontier.

## Mechanism tree

- **Batch-size + sequence amortization** (2 experiments)
  - Pure batch size scaling (bs=2) — ❌ [v001](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v001-batch-scaling.md) refuted (HBM OOM during compile: used 42.99 GB).
  - 🔵 batch size 3+ or sequence length 8192 scaling — UNEXPLORED (completely blocked by HBM).
- **FSDP / collective optimization** (1 experiment)
  - Selective SparseCore offload (`reduce-scatter` only) — 💥 [v002](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v002-selective-sc-offload.md) invalid (crashed on startup with unknown XLA flag).
- **Activation checkpointing** (1 experiment)
  - Selective rematerialization (`checkpoint_dots_with_no_batch_dims`) — ❌ [v003](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v003-selective-remat.md) refuted (HBM OOM during compile: used 33.61 GB, saving 9.38 GB but still 2.36 GB over the limit).
- **Splash attention** (0 experiments)
  - 🔵 GQA-native Splash attention kernel port — UNEXPLORED (catalog: jax-ml/jax).
- **torch.compile + scan** (0 experiments)
  - 🔵 scan-over-layers layer stacking — UNEXPLORED.

## Coverage gaps — what's NOT been explored

| Topic | Catalog options | Tried | Unexplored |
|---|---|---|---|
| Splash attention | 2 | 0 | Splash Attention integration |
| torch.compile + scan | 2 | 0 | Scan-over-layers implementation |
| Tensor Parallelism | 2 | 0 | TP=2 Sharding |
| CE / softmax loss | 2 | 0 | Fused Cross-Entropy kernel |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | Splash Attention + Selective Remat | Splash | +5-10pp (unblocks bs=2) | high | M | 2.33 | local-gap: [Pallas catalog](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/analyses/2026-04-23-pallas-kernel-directory.md#attention--dense-flash--splash-training) |
| 2 | Scan-over-layers + Selective Remat | torch.compile+scan | Unblocks bs=2 | medium | M | 1.63 | local-gap: [Llama3 precedent](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/index.md#models-4) |
| 3 | Tensor Parallelism (TP=2) | TP | Unblocks bs=2 (splits memory) | high | M | 2.33 | local-gap: [model-optimization-index](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/model-optimization-index.md#tp-over-dcn-across-pods) |
| 4 | Fused Cross-Entropy (tokamax-CE) | CE | Unblocks bs=2 | medium | M | 1.63 | local-gap: [Llama3 precedent](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/index.md#models-4) |

## Anti-recommendations

- **Pure single-axis batch scaling at TP=1** without any memory-saving optimizations is structurally closed. Do not attempt to run `bs=2` without adding memory-saving levers (Splash, Scan, TP=2, or Fused CE).
- **Collective offloading flags** that are not supported by the local libtpu version (such as `--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter`) must not be used.

## Cross-lane brief

### Sibling lane summaries (from model pages)

- **qwen3-cc-torchax**: 8B/v6e-8 frontier at baseline = 19.2% MFU @ seq 2048 bs 8 (66% TC idle).
  - Universal levers transferable: Splash attention (reduces attention memory overhead), tokamax CE (streamed cross-entropy to bypass large intermediate logit buffer).

## Experiment ledger (machine-readable)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| v6e8-baseline | 2026-06-02 | 8B/v6e-8 | baseline | baseline | yes (+0%) |
| v000-repro | 2026-06-02 | 8B/v6e-8 | baseline | supported | no |
| v001 | 2026-06-02 | 8B/v6e-8 | batch-size-sequence-amortization | refuted | no |
| v002 | 2026-06-02 | 8B/v6e-8 | fsdp-collective-optimization | invalid | no |
| v003 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing | refuted | no |

## Sources

- Per-lane log: `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/log.md`
- Model page: `wiki/models/qwen3-cc-jax.md`
- Topic taxonomy: `wiki/model-optimization-index.md`
- Pallas catalog: `wiki/analyses/2026-04-23-pallas-kernel-directory.md`
- All 5 experiment pages in `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/`

## See also

- `wiki/models/qwen3-cc-jax.md`
- `.claude/skills/formulate-hypothesis/SKILL.md`
