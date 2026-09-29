---
title: "Qwen3-8B torchax v6e-8 baseline"
type: experiment
hypothesis: baseline
model: qwen3-cc-torchax
variant: "8B/v6e-8"
commit: minimal-trainer-2026-06-02
verdict: baseline
tags: [qwen3-cc, torchax, baseline, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B torchax v6e-8 baseline

First measured baseline for the **minimal** Qwen3-8B torchax trainer on **TPU
v6e-8** (2 hosts × 4 chips, fsdp=8). Establishes the reference number the
optimization experiments climb from. No optimization knobs are enabled
(no splash attention, no tokamax CE, no scan, no AMP master weights) — plain
XLA-through-HF forward + `torch.nn.functional.cross_entropy` + AdamW.

## Hypothesis under test

None — this is the reference baseline, not a hypothesis test. It pins the
starting `step / TPS / MFU` for the `8B/v6e-8` variant row.

## Setup

- **Hardware**: TPU v6e-8 — 2 hosts × 4 chips = 8 chips, fsdp=8, tp=1. Cluster
  `<your-cluster>` (project `<your-project>`, zone `<your-zone>`),
  1 slice (2×4 topology).
- **Dispatch**: GKE via XPK (gke-cluster-runner agent). Workload
  `alekseyv-qwen3-torchax-v000-base`.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-torchax:latest`
  (digest `sha256:1ade963e…`). Base `jax-ai-image/tpu:jax0.9.0-rev1` + torchax
  (from `raw/code/torchax`) + HF transformers 5.3 + the minimal trainer.
- **Env**: `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (required — torch_tpu vs torchax
  privateuse1 conflict), `JAX_COMPILATION_CACHE_DIR=gs://…/qwen3_cc/jax_cache`,
  `XLA_FLAGS=--xla_dump_to=…/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto`.
- **Data**: `use_real_data=False` — synthetic random tokens (perf baseline;
  weights are random meta-init, so loss is not a convergence signal either way).
- **Command** (from the model page "How to run", diffed for the baseline):
  ```bash
  python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
    --batch_size=1 --seqlen=2048 --tp_parallelism=1 \
    --train_steps=20 --weights_dtype=bf16 \
    --profile_dir=gs://…/qwen3_cc/2026-06-02-qwen3-torchax-v6e8-baseline \
    --profile_start_step=12 --profile_steps=3
  ```
  global_batch = batch_size × fsdp = 1 × 8 = 8; tokens/step = 8 × 2048 = 16,384.

> **Why seq 2048, not the 8192 target?** The minimal trainer has no splash
> attention or tokamax CE, so the default XLA-through-HF path materializes the
> `[B, n_heads, S, S]` attention scores and the `[B, L, V]` logits — both OOM at
> seq 8192. 2048 is the honest baseline for the minimal trainer; reaching the
> seq-8192 target is exactly what the splash / CE experiments unlock.

## Baseline comparison

This run *is* the baseline — no prior to compare against. It becomes the
reference for every subsequent `8B/v6e-8` experiment.

## Results

| Metric | This run (baseline) | Notes |
|--------|--------------------|-------|
| Steady step time | **519 ms** (trainer) / 531 ms (xprof avg) | std 4.8 ms — very stable |
| Throughput | **29,795 tok/s** (3,724 / chip) | 18 steps measured (post-warmup) |
| MFU | **19.2%** (trainer est) / MXU util 19.4% (xprof) | v6e bf16 peak 918 TFLOPS/chip |
| TC idle | **66%** (352 / 531 ms) | device idle ~⅔ of the step |
| Cold compile | 112.5 s | cache now warm at `…/qwen3_cc/jax_cache` |
| HBM BW util | ~41% (program) / 30% (conv-fusion) | not memory-bandwidth-bound |
| Exit code | 0 | clean 20-step run |

**Read**: at this tiny shape (per-chip batch 1, seq 2048) the device is **idle
two-thirds of the time** — the matmul (`convolution fusion`) is only 36.9% of the
step and the rest is FSDP-collective stalls + framework dispatch. This is a
*low-occupancy* baseline with large headroom, not a compute-bound ceiling.

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-02-qwen3-torchax-v6e8-baseline`
  (run name `2026-06-02-qwen3-torchax-v6e8-baseline/2026_06_02_04_20_28`).
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-torchax-v6e8-baseline/plugins/profile/2026_06_02_04_20_28/`
  (two hosts: `gke-tpu-46dd3e54-g30s.xplane.pb`, `…-rkr2.xplane.pb` + `.trace.json.gz`).
- **Local pointer**: [`raw/profiles/2026-06-02-qwen3-torchax-v6e8-baseline/`](../../../../../raw/profiles/2026-06-02-qwen3-torchax-v6e8-baseline/GCS_LOCATION.txt)
  (GKE runs are GCS-resident; this pointer file records the gs:// location).
- **Profiled steps**: 12–14 (`profile_start_step=12 profile_steps=3`).
- **Contents**: multi-host xprof trace (2 hosts × 4 chips). Analyzed via
  xprof-mcp: `get_overview` (MXU 19.4%, tc_idle 352 ms), `get_op_profile`
  (`convolution fusion` 36.9%, HBM 30%).

### HLO Dump

- **GCS**: `gs://…/qwen3_cc/2026-06-02-qwen3-torchax-v6e8-baseline/hlo/` — **1274
  files** (`--xla_dump_hlo_as_text --xla_dump_hlo_as_proto`, dumped direct to GCS
  during compile). The post-optimization module is the main train-step jit.
- Use for fusion verification before any kernel-replacement hypothesis (per the
  SCHEMA HLO pre-filter).

## Observations

- **66% TC idle at per-chip batch 1** — the dominant signal. Throughput is gated
  by under-occupancy, not by matmul efficiency. Levers: raise per-chip batch,
  overlap FSDP collectives, splash attention to cut the N² attention stall.
- **Qwen3 RoPE buffer `model.rotary_emb.original_inv_freq` was zero-initialized**
  (`_materialize_buffers_replicated` only special-cases `inv_freq`). Harmless for
  this perf baseline (RoPE values don't affect step time; the live `inv_freq` is
  correctly regenerated), but a **semantics bug to fix before any real-data /
  convergence run**. Filed as a correctness follow-up.
- Direct-to-GCS XLA HLO dump **works** on the `jax-ai-image` base (1274 files
  landed) — no `/tmp`+upload fallback needed.

## Verdict + reasoning

**baseline** — reference established. `EXIT_CODE=0`, profile + HLO + compile
cache all persisted to GCS, MFU/throughput captured. Not a supported/refuted
hypothesis test; it sets the `8B/v6e-8` matrix row's `Baseline` cell at
**519 ms / 29,795 TPS / 19.2% MFU @ (seq 2048, global batch 8)**.

## Next hypotheses

- [Splash attention (TPU)](../../../../hypotheses/qwen3-torchax-splash-attention.md) — replace XLA-through-HF SDPA with the GQA-native splash kernel; cuts the `[B,H,S,S]` materialization, the prerequisite for reaching seq 8192 and a lever on the 66% idle.
- [Per-chip batch scaling](../../../../hypotheses/qwen3-torchax-batch-scaling.md) — raise `batch_size` (per-chip batch >1) to fill the 66% TC idle; cheapest first move, no code change.
- [tokamax streamed cross-entropy](../../../../hypotheses/qwen3-torchax-tokamax-ce.md) — stream logsumexp over V to drop the `[B,L,V]` logit materialization; unlocks seq 8192 at the lm_head.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-torchax-v6e8-baseline/`
- Local pointer: `raw/profiles/2026-06-02-qwen3-torchax-v6e8-baseline/GCS_LOCATION.txt`
- Trainer: `wiki/experiments/qwen3_cc_autoresearch_optimization/torchax/` (train.py, model/, profiling.py).
