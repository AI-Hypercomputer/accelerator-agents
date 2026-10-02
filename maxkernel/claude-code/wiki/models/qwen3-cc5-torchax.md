---
title: "Qwen3 8B — torchax"
type: model
architecture: qwen3-cc5
lane: torchax
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-cc5, torchax]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 8B — torchax

Qwen3 8B (`Qwen/Qwen3-8B`) running under **torchax (PyTorch-on-JAX)** on **TPU
v6e-8**, fine-tuning `wikitext-2-raw-v1` at fixed seq_len. Primary lane for the
qwen3_cc5 program. A minimal baseline trainer is in place; optimization is the work.

## Target metrics

- Primary: MFU at seq_len=8192 (v6e bf16 peak ≈ 918 TFLOPS/chip).
- Secondary: tokens/sec (and tok/s/chip), step time, peak HBM.
- Measured by `train.py`'s post-warmup throughput + the train-step-TFLOPs MFU
  estimate it prints in the summary block.

## How to run

```bash
conda activate py312
cd wiki/experiments/qwen3_cc5_autoresearch_optimization/torchax
python -m train --steps 20 --batch_size <B> --seqlen <L> \
    --tp_parallelism 1 --weights_dtype bf16 \
    --profile_dir $PROFILE_DIR --profile_step 10
```
- `<B>` = per-fsdp-shard batch (global batch = B × fsdp; fsdp = chips ÷ tp).
- `<L>` = sequence length (target 8192).
- `$PROFILE_DIR` = `raw/profiles/<YYYY-MM-DD>-<slug>/` (repo-root anchored).

## Variant matrix

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
|------|----------|--------|-----------------------------|----------------------------------|-----------|--------------|
| 8B | v6e-8 | live | 519 ms / 29,795 TPS / 19.2% MFU @ seq2048 bs8 | = baseline | 3 | [2026-06-02 baseline](../experiments/qwen3_cc5_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md) |

*Baseline captured at seq 2048 (global batch 8) — the minimal trainer OOMs at the
seq-8192 target without splash/CE. Profile shows **66% TC idle** (under-occupancy,
not compute-bound): large headroom. See the [baseline experiment](../experiments/qwen3_cc5_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md).*

## Cross-variant open hypotheses

Ranked after the [2026-06-02 baseline](../experiments/qwen3_cc5_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md)
(66% TC idle → win occupancy/memory first):

1. [Per-chip batch scaling](../hypotheses/qwen3-torchax-batch-scaling.md) — raise batch to fill the 66% idle. Effort S, confidence high — **cheapest first move**.
2. [Splash attention](../hypotheses/qwen3-torchax-splash-attention.md) — GQA-native TPU kernel; avoids `[B,H,S,S]` materialization; prerequisite for seq 8192. Effort M.
3. [tokamax streamed cross-entropy](../hypotheses/qwen3-torchax-tokamax-ce.md) — drop `[B,L,V]` logit materialization at the lm_head; unlocks seq 8192. Effort M.

Not yet filed (file under `wiki/hypotheses/` when promoted):
- **scan-over-layers** — stack the 36 decoder layers into one scan body to cut compile-time HBM peak.
- **Fused QK-norm + RoPE Pallas kernel** — Qwen3-specific: `q_norm`/`k_norm` RMSNorm + RoPE in a single pass over `[B, S, H, D]`.

## Variant-specific open hypotheses

(none yet — only one variant row)

## Retired hypotheses

(none yet)

## Knobs translation matrix

| Technique | 8B/v6e-8 | Notes |
|-----------|----------|-------|
(populated after first cross-variant validation experiments)

## Iteration ladder

1. **Capture baseline** — 8B/v6e-8, seq=8192, bs tuned to fit HBM. The minimal
   `train.py` as-is.
2. Memory wins first (splash, tokamax CE, scan) to unlock batch/seq headroom.
3. Throughput wins (XLA flag stack, kernel tuning) once memory has slack.
4. (Later) scale-out / cross-lane comparison vs the jax lane.

## See also

- [Qwen3 8B — jax](qwen3-cc5-jax.md) — sibling lane (reference ceiling, not yet built).
- Program: [`../experiments/qwen3_cc5_autoresearch_optimization/program.md`](../experiments/qwen3_cc5_autoresearch_optimization/program.md)
- Lane README: [`../experiments/qwen3_cc5_autoresearch_optimization/torchax/README.md`](../experiments/qwen3_cc5_autoresearch_optimization/torchax/README.md)

## Sources

- `wiki/experiments/qwen3_cc5_autoresearch_optimization/torchax/` (trainer code).
