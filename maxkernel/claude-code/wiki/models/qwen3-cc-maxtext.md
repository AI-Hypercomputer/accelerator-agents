---
title: "Qwen3 8B — maxtext (reference lane)"
type: model
architecture: qwen3-cc
lane: maxtext
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-cc, maxtext, reference]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 8B — maxtext (reference lane)

Upstream **MaxText** running `Qwen/Qwen3-8B` on **TPU v6e-8**, used as the **reference ceiling** for the
qwen3_cc program (mirrors the llama3 program's torchax→jax→MaxText structure). We run MaxText verbatim —
no model code in this lane; the architecture is MaxText's own `configs/models/qwen3-8b.yml`, which matches
our trainer exactly (4096 hidden, 32 q / 8 kv heads, head_dim 128, 36 layers, mlp 12288, QK-norm, SwiGLU,
vocab 151936, rope θ 1e6 — see [observation](../observations/qwen3-maxtext-config-exists.md)).

**Purpose**: establish the MaxText MFU for this exact architecture so the jax (35.8% @ seq2048 / 30.4% @
seq8192) and torchax (19.2%) lanes have a true apples-to-apples ceiling, the way llama3-8b-jax (43.3%) is
measured against the MaxText Llama-3.1-8B reference (44.6%).

## Target metrics

- Primary: MFU at seq_len=8192 (v6e bf16 peak ≈ 918 TFLOPS/chip), reported as TFLOP/s/device → MFU.
- Secondary: tokens/sec/device, step time, peak HBM.

## How to run

Run from the MaxText worktree (`/mnt/disks/persist/maxtext-tpu-recipes-v0.1.4`, tag `tpu-recipes-v0.1.4`,
commit `9f1820b47`) in the maxtext venv (`/mnt/disks/persist/venv-maxtext-v0.1.4`), base image
`maxtext_base_image:latest` (built via `docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack
BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.6.1-rev1`):

```bash
export PROJECT=<your-project> ZONE=<your-zone> CLUSTER_NAME=<your-cluster>
export OUTPUT_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/maxtext/<run>
python3 -m benchmarks.benchmark_runner xpk \
  --project=$PROJECT --zone=$ZONE --device_type=v6e-8 --num_slices=1 \
  --cluster_name=$CLUSTER_NAME --base_output_directory=$OUTPUT_DIR \
  --model_name=qwen3_8b_8192_ref \        # or qwen3_8b_2048_ref for the seq2048 shape
  --base_docker_image=maxtext_base_image
```

The benchmark configs `qwen3_8b_8192_ref` (seq8192, bs3) and `qwen3_8b_2048_ref` (seq2048, bs4) were added
to the worktree's `benchmarks/maxtext_trillium_model_configs.py`, mirroring the proven
`llama3_1-8b-8192-no-collective-matmul` recipe tuning (remat custom + decoder/proj offloads, attention
flash/splash with `sa_block_*=2048` + `sa_use_fused_bwd_kernel`, synthetic data, the HOST_OFFLOAD +
SparseCore-offload + DISABLE_COLLECTIVE_MATMUL XLA flag stack).

## Variant matrix

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best | Open hyps | Frontier exp |
|------|----------|--------|------------------------------|--------------|-----------|--------------|
| 8B | v6e-8 | live | **seq8192 bs3: 45.3% MFU / 6,942 tok/s/chip** · **seq2048 bs4: 38.0% MFU / 7,116 tok/s/chip** | same (reference baselines) | 1 | [seq8192 ref](../experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192.md) · [seq2048 ref](../experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq2048.md) |

***Reference ceiling measured 2026-06-02**: MaxText Qwen3-8B @ seq8192 bs3 = **45.3% MFU / 6,942 tok/s/chip** —
on par with the llama3-8b MaxText reference (44.6%), confirming the recipe transfers (incl. QK-norm). This
is the achievable ceiling. **Cross-stack at seq8192**: MaxText 6,942 vs our jax 5,305 (v009) = **+31%
tok/s/chip** — the jax lane is ~15 pp below ceiling, the gap being MaxText's offload-enabled bs3
(`decoder_layer_input` + 4 proj offloads) that the jax lane never tried. **seq2048 reference** (bs4) =
**38.0% MFU / 7,116 tok/s/chip** — the jax v018 frontier (35.8% / 6,964) is at **97.9%** of it, i.e. near
parity → the jax seq2048 optimization is externally confirmed complete. Net: the jax actionable headroom is
seq8192-only (offload-enabled bs3), not seq2048.*

> [!warning] MaxText MFU above is **non-causal** — causal-adjusted ceiling is **39.8% @ seq8192 / 36.6% @ seq2048** (updated 2026-06-16)
> Both reference numbers came from `tpu-recipes-v0.1.4` (`9f1820b`), which **predates MaxText's causal-mask `/2` fix** (commit `6288c233`, 2026-04-11) and so counts attention **non-causally** (full attention). Verified numerically against the qwen3-8b config: `45.3%` = non-causal at 6,942 tok/s/chip *exactly* (causal = **39.8%**); `38.0%` = non-causal at 7,116 (causal = **36.6%**). **All jax/torchax lanes count attention causally (`÷2`)**, so the lane MFU and this reference MFU are **not on the same basis** — MFU-vs-MaxText must use the causal-adjusted figures, and **TPS (convention-free) is the comparison to trust**. On the causal basis the jax frontier (cx 43.2% / cc5 39.9% @8k) **meets-or-beats** this ceiling — consistent with TPS (cx 7,543 / cc5 6,959 vs MaxText 6,942). The measured 45.3/38.0% on the reference *experiment* pages are left as-recorded (immutable); this is the adjustment. Same finding + mechanism as [llama3-8b](../experiments/llama3_8B_autoresearch_optimization/README.md#maxtext-mfu-is-reported-on-a-stale-non-causal-basis).

## Cross-variant open hypotheses

(none yet — the reference baseline must land first)

## Variant-specific open hypotheses

(none yet)

## Retired hypotheses

(none yet)

## Knobs translation matrix

| Technique | 8B/v6e-8 | Notes |
|-----------|----------|-------|
(populated after the reference baseline + first ablations)

## Iteration ladder

1. **Reference baseline** — `qwen3_8b_8192_ref` (seq8192, recipe tuning) on v6e-8; measure MFU/TFLOP-s.
2. `qwen3_8b_2048_ref` (seq2048) — direct shape-match to the jax v018 frontier (35.8%).
3. Cross-lane analysis: MaxText vs jax (35.8%/30.4%) vs torchax (19.2%) — quantify each stack's gap to the
   MaxText ceiling, as the llama3 program does.

## See also

- [Qwen3 8B — jax](qwen3-cc-jax.md) — hand-tuned native-JAX lane (frontier 35.8% MFU).
- [Qwen3 8B — torchax](qwen3-cc-torchax.md) — torchax lane (19.2%).
- [observation: MaxText qwen3-8b config exists](../observations/qwen3-maxtext-config-exists.md)
- [maxtext codebase](../codebases/maxtext.md); MaxText Llama-3.1-8B reference: [llama3 maxtext baseline](../experiments/llama3_8B_autoresearch_optimization/maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md)

## Sources

- `raw/code/maxtext/` @ `532c8b3d8` (`configs/models/qwen3-8b.yml`); worktree `tpu-recipes-v0.1.4` (`9f1820b47`).
