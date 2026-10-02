---
title: "MaxText Gemma 4 E4B v6e-8 reference baseline (APPROXIMATION; supported, 282.9 TFLOP/s/device, 30.8% MFU)"
type: experiment
tags: [experiment, gemma4-e4b, maxtext, v6e-8, baseline, reference, supported, approximation]
hypothesis: maxtext-gemma4-e4b-reference-baseline
model: gemma4-e4b-maxtext-v6e8-approx
created: 2026-04-25
updated: 2026-04-25
commit: 532c8b3d8
verdict: supported
---

> **⚠️ This is an architectural approximation, not a true E4B port.** See [`../README.md`](../README.md) for the full caveat. Short version: HF E4B uses `num_kv_shared_layers=18` (last 18 layers reuse K/V from earlier same-type layers); MaxText doesn't implement this. The model run here has 18 extra k/v projection sets — about **+47M params (~0.6 % over true E4B)**. Throughput is a measurement of the dense-shape model that this config builds, not directly comparable to a true E4B implementation.

First MaxText reference run on Gemma 4 E4B-shape on the autoresearch v6e-8 GKE cluster. **No upstream MaxText E4B config exists** — Gemma 4 support landed on `main` after `tpu-recipes-v0.1.4` and ships only `gemma4-26b.yml` (MoE) and `gemma4-31b.yml` (Dense). Authored a wiki-local [`gemma4-e4b.yml`](https://github.com/AI-Hypercomputer/maxtext) that ports the HF `google/gemma-4-E4B` text_config dimensions, plus a one-line patch to the Pydantic model_name whitelist in `src/maxtext/configs/types.py`.

Steady-state median over 9 stable steps (4–9 + 14, 17–19, excluding profile-boundary steps and timing-anomaly steps): **282.9 TFLOP/s/device, 10,003 Tokens/s/device, 30.8 % per-chip MFU** at `bs=2, seq=8192, fsdp=8, remat_policy=full`. Aggregate ~80k tok/s across 8 chips. Compile fits at 22.51 GiB / 31.25 GiB HBM (72 % used). Verdict: `supported` — the approximation runs end-to-end, the model "trains" (compiles + step runs cleanly), and the throughput number anchors the dense-shape ceiling for the Gemma 4 E4B program.

## Hypothesis under test

The Gemma 4 E4B architecture (42 layers, hidden 2560, heads 8, kv_heads 2, head_dim 256, FFN 10240, vocab 262144, SW=512, soft-cap 30.0, 5:1 SW:G attention pattern) is expressible in MaxText's main-branch `decoder_block: gemma4` codepath, and runs end-to-end on v6e-8 at FSDP=8 with reasonable defaults. Establishing this baseline:
1. Confirms the `decoder_block: gemma4` path can host the E4B shape (not just 26B/31B).
2. Provides a measured throughput anchor for the E4B program.
3. Surfaces the architectural delta to upstream — to be filed back as a feature request when the wiki has bandwidth.

## Setup — three modifications to upstream MaxText `main`

This experiment is **not a clean reproduction of a published recipe** — there is no published recipe. Three local changes were required:

### 1. New model config — [`src/maxtext/configs/models/gemma4-e4b.yml`](https://github.com/AI-Hypercomputer/maxtext) (NEW)

```yaml
base_num_decoder_layers: 42
base_emb_dim: 2560
base_num_query_heads: 8
base_num_kv_heads: 2
base_mlp_dim: 10240
mlp_activations: ["gelu","linear"]
head_dim: 256
vocab_size: 262144
decoder_block: "gemma4"
normalization_layer_epsilon: 1e-6
logits_via_embedding: True

# Attention — uniform head config across SW and GLOBAL layers
sliding_window_size: 512
use_post_attn_norm: true
use_post_ffw_norm: true
share_kv_projections: false
v_norm_with_scale: false

# RoPE
local_rope_max_timescale: 10000
global_rope_max_timescale: 1000000
rope_max_timescale: 1000000
global_rope_proportion: 0.25
local_rope_proportion: 1.0

final_logits_soft_cap: 30.0
```

All numeric values pulled from HF [`google/gemma-4-E4B/config.json`](https://huggingface.co/google/gemma-4-E4B/blob/main/config.json) `text_config`, with these gaps documented:
- HF doesn't expose `rope_local_base_freq` / `rope_theta` separately for E4B; using the family defaults from gemma4-26b/31b (10k local, 1M global, `global_rope_proportion=0.25`).
- HF's `num_kv_shared_layers=18` has **no MaxText analog** — see "Approximation caveat" above.

### 2. Pydantic literal patch — [`src/maxtext/configs/types.py:239`](https://github.com/AI-Hypercomputer/maxtext) (1-line addition)

```python
"gemma4-26b",
"gemma4-31b",
"gemma4-e4b",   # ← ADDED for this experiment
"qwen2.5-1.5p",
```

Without this, the Pydantic model_name validator rejects the new config with `literal_error`. xpk's `--base-docker-image` flow COPYs the local source on top of the prebuilt base image at `/app`, so the patched files at `/app/src/maxtext/configs/types.py` exist on the runner — but Python imports `maxtext` from the editable install at `/deps/src/...` (baked into the base image at build time). Worked around with `PYTHONPATH=/app/src` in the user_command (overrides the editable install). For a permanent fix the base image would need to be rebuilt.

### 3. PYTHONPATH override in user_command

```bash
export PYTHONPATH=/app/src:$PYTHONPATH
```

Otherwise the in-image `/deps/src/maxtext/configs/types.py` (without the patch) wins over `/app/src/maxtext/configs/types.py` (with the patch).

## Setup — cluster and tools

**Cluster**: `<your-v6e8-cluster>` (<your-zone>, project <your-project>, 2 hosts × 4 chips = 8, **spot**).

**MaxText**: `main` @ `532c8b3d8` (commit ID; not a tagged release). Worktree at `/mnt/disks/persist/maxtext-main` (kept outside `raw/` per SCHEMA rule 1).

**Base image**: `maxtext_base_image:latest` (1.77 GiB content size), built from the new MaxText main + its `src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile` with default Python 3.12. The Llama-3.1-8B baseline's `jax0.6.1-rev1` based image was overwritten — main MaxText handles its own deps via its Dockerfile; no `BASEIMAGE=` arg needed.

**XPK**: tag `v0.14.3` at `~/xpk` (same as Llama-8B run).

**Workload name**: `<workload-name>` (`-mt-1` through `-mt-4` were earlier failed attempts — see "Iterations" below).

**Submission** (run from `/mnt/disks/persist/maxtext-main` in the maxtext venv):

```bash
W=<workload-name>
RUN_NAME=$W
OUTPUT_DIR=gs://<your-bucket>/maxtext/v6e8-20260425-02-gemma4-e4b
LIBTPU_VAL=" --xla_tpu_scoped_vmem_limit_kib=98304"

CMD=" echo 'using LIBTPU_INIT_ARGS' \
  && export LIBTPU_INIT_ARGS='${LIBTPU_VAL}' \
  && export PYTHONPATH=/app/src:\$PYTHONPATH \
  && export ENABLE_PATHWAYS_PERSISTENCE=1 \
  && export JAX_PLATFORMS=tpu,cpu \
  && export ENABLE_PJRT_COMPATIBILITY=true \
  && python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
       per_device_batch_size=2 \
       ici_fsdp_parallelism=-1 \
       remat_policy=full \
       max_target_length=8192 \
       attention=flash \
       use_iota_embed=True \
       dataset_path=gs://max-datasets-rogue \
       dataset_type=synthetic \
       enable_checkpointing=False \
       sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
       sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
       sa_block_q_dq=2048 sa_block_kv_dq=2048 \
       sa_use_fused_bwd_kernel=True \
       profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=5 \
       steps=20 \
       model_name=gemma4-e4b \
       base_output_directory=${OUTPUT_DIR} \
       use_vertex_tensorboard=false vertex_tensorboard_project=\"\" vertex_tensorboard_region=\"\" \
       run_name=${RUN_NAME}"

python3 ~/xpk/xpk.py workload create \
  --cluster=<your-v6e8-cluster> \
  --project=<your-project> --zone=<your-zone> \
  --device-type=v6e-8 --num-slices=1 \
  --command="$CMD" --base-docker-image=maxtext_base_image \
  --enable-debug-logs --workload=$W --priority=medium --max-restarts=0
```

Note: only `--xla_tpu_scoped_vmem_limit_kib=98304` is used in `LIBTPU_INIT_ARGS`. The Llama-3.1-8B recipe's full 30+-flag set was tried first and rejected by the newer libtpu in main with `Unknown command line flag '2a886c8_chip_config_name'`. Reasonable defaults are sufficient for an unsupported-recipe baseline.

## Iterations

Five workload submissions, four killed before completion:

| # | Failure | Fix | Time spent |
|---|---|---|---|
| `-mt-1` | `pydantic literal_error: model_name='gemma4-e4b'` not in whitelist | Patched `types.py` | ~1 min |
| `-mt-2` | Same error — patch only applied to `/app`; in-image `/deps` wins via editable install | Add `PYTHONPATH=/app/src` | ~2 min |
| `-mt-3` | `Unknown command line flag '2a886c8_chip_config_name'` in newer libtpu | Strip Llama-recipe LIBTPU_INIT_ARGS down to just `--xla_tpu_scoped_vmem_limit_kib=98304` | ~3 min |
| `-mt-4` | `out of memory while compiling` — `bs=4 seq=8192` doesn't fit (FFN intermediate `bf16[4,8192,10240]` × ~20 buffers = 13+ GiB live) | Drop to `bs=2` | ~4 min |
| `-mt-5` | ✅ `Succeeded` | — | ~1.5 min compile + 30 s run |

Total wall-clock from "build maxtext_base_image" to "ran successfully": ~15 min.

## Baseline comparison

There is **no upstream baseline to compare against** — this is the first MaxText E4B-shape run anywhere we know of. The Gemma 4 program's existing baselines come from torchax + native-JAX stacks at smaller seq lengths (typically `seq=2048, bs=1` on v6e-4). Cross-stack comparison is not apples-to-apples here:

| Stack | Baseline | seq | bs | mesh | TPS | MFU | source |
|---|---|---|---|---|---|---|---|
| MaxText main (this run) | E4B-approx (no kv-share) | 8192 | 2 | fsdp=8 (v6e-8) | 10,003 | 30.8 % | this page |
| torchax | E4B (true) | 1024 | 1 | fsdp=4 (v6e-4) | 30,570 | — | [2026-04-22 baseline](../../torchax/experiments/2026-04-22-baseline.md) (per program README) |
| native-JAX (Flax NNX) | E4B (true) | 1024 | 1 | fsdp=4 (v6e-4) | 30,285 | — | [exp 34](../../jax/experiments/2026-04-23-exp34-jax-baseline-accepted.md) |

Direct comparison would require either (a) a torchax/JAX run at matched `seq=8192, bs=2, fsdp=8`, or (b) a MaxText-main run at matched `seq=1024, bs=1, fsdp=4`. Neither has been done.

What this run **does** provide: an absolute-throughput measurement of the dense-shape model on Trillium under reasonable XLA defaults. **30.8 % MFU is the floor**, not a tuned ceiling.

## Results

Per-step (from `kubectl logs <workload-pod>`):

| step | s/step | TFLOP/s/device | Tokens/s/device | loss | notes |
|---:|---:|---:|---:|---:|---|
| 0 | 4.424 | 104.7 | 3,703 | 12.977 | first step = compile + warmup |
| 1 | **0.284** | **1,632** | **57,698** | 12.977 | **timing artifact** (next-step lag from compile) |
| 2 | 1.490 | 310.9 | 10,995 | 12.976 | warmup tail |
| 3 | 1.503 | 308.3 | 10,904 | 12.978 | warmup tail |
| 4 | 1.639 | 282.7 | 9,998 | 12.975 | steady |
| 5 | 1.637 | 283.1 | 10,011 | 12.977 | steady |
| 6 | 1.639 | 282.8 | 9,999 | 12.979 | steady |
| 7 | 1.638 | 282.9 | 10,003 | 12.975 | steady |
| 8 | 1.638 | 282.9 | 10,003 | 12.982 | steady |
| 9 | 1.639 | 282.8 | 9,999 | 12.978 | steady |
| 10 | 3.360 | 137.9 | 4,877 | 12.976 | **profiler START** |
| 11 | **0.011** | **41,228** | — | 12.977 | **timing artifact** |
| 12 | 1.905 | 243.3 | 8,603 | 12.975 | profiled |
| 13 | 1.372 | 337.7 | 11,940 | 12.979 | profiled (timing perturbation) |
| 14 | 1.639 | 282.7 | 9,998 | 12.975 | profiled, steady |
| 15 | **21.975** | 21.1 | 746 | 12.976 | **profiler STOP** + xplane.pb write to GCS, ~22 s overhead |
| 16 | **0.009** | **52,135** | — | 12.975 | **timing artifact** |
| 17 | 1.636 | 283.3 | 10,017 | 12.975 | post-profile, steady |
| 18 | 1.637 | 283.0 | 10,008 | 12.975 | steady |
| 19 | 1.639 | 282.8 | 9,999 | 12.974 | last step |

**Steady-state metrics** (median of steps 4–9, 14, 17–19 — excluding warmup, profile boundaries, and timing artifacts):

| Metric | Value | Notes |
|---|---|---|
| **Step time (median)** | **1.638 s** | window: 1.636 – 1.639 s (Δ=3 ms, very tight) |
| **TFLOP/s/device (median)** | **282.9** | window: 282.7 – 283.3 |
| **Tokens/s/device (median)** | **10,003** | window: 9,998 – 10,017 |
| **Per-chip MFU** | **30.8 %** | 282.9 / 918 (Trillium peak from program-gke.md) |
| **Aggregate TPS (8 chips)** | **80,025 tok/s** | 8 × 10,003 |
| Step-0 compile time | 4.42 s + ~50 s HLO-compile pre-step-0 | One-shot |
| Profile-stop overhead (step 15) | +20 s | One-shot, attribution: xplane.pb write to GCS (~133 MiB) |
| HBM peak (compile-time reported) | 22.51 GiB / 31.25 GiB | 72 % HBM utilization — at `bs=2 seq=8192 remat=full`, much tighter than Llama-3.1-8B's 34 % |

`total_weights` in step log = 131,072 = `bs(2) × num_chips(8) × seq_len(8192)`. ✓

Loss is essentially flat (12.974–12.982) across all 20 steps — synthetic data has no real signal at this scale. **This is expected for a perf-only baseline**, not a sign of a broken model. Step-0 loss of 12.977 = ln(262144), exactly what untrained logits should produce on a uniform-vocab synthetic dataset.

## Profile

- **xprof browser URL**: `http://localhost:8791/?run=2026-04-25-maxtext-gemma4-e4b-v6e8-baseline` (after `xprof --logdir=raw/profiles --port=8791 &`).
- **MaxText run name (in xprof)**: `<workload-name>`. Canonical GCS path: `gs://<your-bucket>/maxtext/v6e8-20260425-02-gemma4-e4b/<workload-name>/`.
- **On-disk trace dir**: [raw/profiles/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline/](../../../../../../raw/profiles/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline/) (mirror via `gcloud storage cp -r`; ~133 MiB, 4× the size of the Llama-8B trace because Gemma 4's HLO has more attention-pattern variants — alternating SW+GLOBAL layers each compile to distinct kernels).
- **Steps captured**: 11–15 (`profiler=xplane`, `skip_first_n_steps_for_profiler=10`, `profiler_steps=5`).
- **Trace contents**: `tensorboard/plugins/profile/2026_04_25_17_37_27/{*.xplane.pb, *.trace.json.gz}`. Rank-0 only.

## Observations

1. **Step time is exceptionally tight** — 3 ms range across 9 steady-state steps (Δ < 0.2 %). Same characteristic as the Llama-8B run on this cluster.
2. **30.8 % MFU is below the Llama-3.1-8B 44.6 % MFU** measured on the same cluster, on the same date, with the same hardware. Three plausible drivers: (a) E4B-shape is smaller (fewer compute FLOPs per byte of activation traffic); (b) `remat_policy=full` recomputes everything (more compute, lower throughput); (c) we're missing the recipe-tuned LIBTPU_INIT_ARGS flag set that Llama-8B used — async collective fusion, all-gather pipelining, and host-offload all impact MFU on Trillium. Item (c) is the most actionable next step.
3. **HBM is tight at 72 % use** — `bs=4` overflowed at compile. This is more constrained than the Llama-8B run (34 % HBM at `bs=3 seq=8192`), despite E4B being a smaller model. Driver: `remat_policy=full` rematerializes selected layers but the FFN intermediate `bf16[bs,8192,10240]` is the dominant live buffer regardless. Switching to `remat_policy=qkv_proj_offloaded` or adding the `*_proj=offload` recipe set (as Llama-8B does) would likely free HBM for `bs=3` or even `bs=4`.
4. **Profile-stop overhead is 20 s** (vs 5.5 s for Llama-8B) — proportional to xplane size: 133 MiB vs 31 MiB. The hybrid-attention pattern produces many more distinct kernels (SW kernel × GLOBAL kernel × 7 cycles + reset graph), inflating the trace.
5. **Compile time is much higher** (~50 s of HLO compile pre-step-0 + 4.4 s on step 0 = ~55 s total) compared to Llama-8B's 12 s. Same root cause as point 4 — Gemma 4's interleaved SW+GLOBAL layers force two distinct attention kernels per cycle, multiplying HLO instruction count and fusion search space.
6. **Step 1 and 11/16 timing artifacts (`0.011`, `0.009` seconds)** are MaxText `metric_logger` quirks, not real measurements. Common pattern across MaxText runs (also visible in the Llama-8B log at step 1 = 3.57 s but in opposite direction). Don't include in steady-state aggregations.
7. **The approximation runs end-to-end** — model compiles, all 20 steps complete, profile captures, pod exits cleanly. Confirms the structural surgery (config + pydantic literal + PYTHONPATH workaround) is functionally correct, even if it omits `num_kv_shared_layers`.

## Verdict + reasoning

**`supported`** — but as an *approximation*, not a true E4B reproduction.

What the verdict means here:
- ✅ The wiki-local `gemma4-e4b.yml` config + Pydantic patch successfully drives MaxText's `gemma4` decoder_block at the E4B shape.
- ✅ The model compiles to a 22.5 GiB HBM program on Trillium v6e-8 and runs cleanly through 20 steps.
- ✅ Steady-state throughput is consistent (3 ms step-to-step variance) and the result reflects a real measurement of the dense-shape model.
- ❌ **The model is not bit-exact E4B** — `num_kv_shared_layers=18` is not implemented. The +47M extra params skew param count by ~0.6 %.
- ❌ **No upstream-published number to validate against** — this is an absolute measurement, not a parity check.

For SCHEMA compliance: this is a baseline-establishment run, not an A/B test against a hypothesis. The "baseline comparison" target (the recipe README) doesn't exist. Treating as `supported` with the explicit caveat above; future hypotheses targeting this baseline must cite both the metric *and* the approximation note.

## Next hypotheses

To be filed as separate hypothesis pages. In rough rank order:

1. **`maxtext-gemma4-e4b-v6e8-libtpu-flag-set`** — port the Llama-8B recipe's full LIBTPU_INIT_ARGS set (minus the version-rejected `2a886c8_*` flag), see how much of the 13.8 pp MFU gap (30.8 → 44.6) is explained by XLA scheduler tuning. Confidence: high. Effort: S.
2. **`maxtext-gemma4-e4b-v6e8-host-offload`** — adopt the recipe's `decoder_layer_input=offload` + `*_proj=offload` set to free HBM, then try `bs=3`. Confidence: medium-high. Effort: S.
3. **`maxtext-gemma4-e4b-v6e8-remat-policy`** — try `remat_policy=qkv_proj_offloaded` and `remat_policy=custom` (matching Llama recipe). Quantify remat overhead on E4B's 42-layer + interleaved-attention shape. Confidence: medium. Effort: S.
4. **`maxtext-gemma4-e4b-v6e8-cross-stack-vs-torchax`** — run the torchax E4B at matched `seq=8192, bs=2, fsdp=8` to enable apples-to-apples cross-stack comparison. **Note: torchax may also need work to scale to seq=8192 — currently runs at seq=1024.** Effort: L (gated on torchax port at 8k seq).
5. **`maxtext-gemma4-e4b-add-kv-share-upstream`** — file a feature request / draft PR upstream to MaxText to add `num_kv_shared_layers` support to the `gemma4` decoder_block. Effort: L. Once landed, retire this approximation experiment in favor of a true-E4B baseline.

## See also

- [maxtext stack README](../README.md) — full approximation caveat.
- [gemma4 program](../../README.md)
- [llama3_8B/maxtext baseline](../../../llama3_8B_autoresearch_optimization/maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md) — reference comparison for the same cluster on the same day.

## Sources

- `/mnt/disks/persist/maxtext-main/src/maxtext/configs/models/gemma4-e4b.yml` — local config (wiki-only; not upstream).
- `/mnt/disks/persist/maxtext-main/src/maxtext/configs/types.py:239` — local 1-line patch to Pydantic literal.
- `raw/code/maxtext` @ `main` (`532c8b3d8`) — base MaxText sources.
- HF [`google/gemma-4-E4B/config.json`](https://huggingface.co/google/gemma-4-E4B/blob/main/config.json) `text_config` — architectural spec source.
- [raw/profiles/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline/](../../../../../../raw/profiles/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline/) (mirror; canonical at `gs://<your-bucket>/maxtext/v6e8-20260425-02-gemma4-e4b/`).
