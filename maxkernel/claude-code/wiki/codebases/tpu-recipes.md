---
title: "tpu-recipes (AI-Hypercomputer/tpu-recipes)"
type: codebase
tags: [tpu-recipes, maxtext, recipes, trillium, ironwood, v6e, v7x, llama, gemma, mixtral, deepseek, qwen, microbenchmark]
repo: AI-Hypercomputer/tpu-recipes
commit: e284e3613721882ce3e15c533a76c691443ea60f
created: 2026-04-25
updated: 2026-04-25
---

> **Grounded front door → [tpu-recipes overview](tpu-recipes/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Google AI-Hypercomputer's curated **per-(model, hardware, topology) reproduction recipes** for training and inference on Cloud TPU. Each recipe is a small directory with a launch script, a `README.md` that pins the exact MaxText commit/tag, the JAX `jax-stable-stack` Docker image version, and the tuned MaxText `tuning_params` block used to hit the published throughput target. **Crown jewel for this wiki: a tested-on-Google-hardware table of per-model (Llama 3.1, Gemma 3/4, Mixtral, DeepSeek 3, Qwen 3, GPT-OSS, Wan 2.1, GPT-3 175B) MaxText config knobs across v6e (Trillium) and v7x (Ironwood) topologies — directly comparable references for any optimization hypothesis on similar models.**

## Overview

Unlike a kernel library or a trainer codebase, tpu-recipes is a **knowledge artifact**: the per-leaf-directory `README.md` + shell launch script + (sometimes) a `k8s_manifest.yaml` together describe a full reproducible workload. The MaxText `tuning_params` blocks embedded in the READMEs are the most direct public source of "what flags Google's perf team picked for this combination." For the autoresearch loop here, those blocks are reference priors: any hypothesis touching `remat_policy`, FSDP sharding, `decoder_layer_input` offload, or per-projection offload should first compare against the recipe most similar to the model under optimization.

The repo also includes a small **microbenchmark** harness (HBM bandwidth, matmul peak TFLOPS) and a profile-conversion utility (`utils/profile_convert.py` — `xplane_pb2.py` for trace post-processing).

## Architecture

```
training/
  MAXTEXT_README.md                ← prerequisite: install MaxText, build docker image, set GCS bucket
  XPK_README.md                    ← prerequisite: GKE cluster setup via XPK
  archive/                         ← retired recipes (older generations)
  ironwood/                        ← v7x recipes (4x4x4 = 64 chips, 4x8x8 = 256 chips)
    deepseek3-671b/                ← bf16, fp8 variants
    gemma4-26b/<seq-precision-topology>/{k8s,xpk}/
    gemma4-31b/
    gpt-oss-120b/
    llama3.1-405b/
    llama3.1-70b/<seq-precision-topology>/{k8s,xpk}/
    qwen3-235b-a22b/
    wan2.1-14b/                   ← video-generation diffusion model
  trillium/                        ← v6e recipes (v6e-32, v6e-64, v6e-128, v6e-256)
    GPT3-175B-MaxText/
    Gemma3-12B-MaxText/
    Llama2-70B-MaxText/
    Llama3.1-8B-MaxText/v6e-{32,64,128,256}/
    Llama3.1-70B-MaxText/v6e-{32,64,128,256}/
    Llama3.1-405B-MaxText/
    Mistral-7B-MaxText/
    Mixtral-8x7B-MaxText/
    Mixtral-8x22B-MaxText/
  v5p/                             ← legacy — DLRM-V2 (TF), Llama 2/3.1, Llama 4 Maverick/Scout, DeepSeek 3, Mixtral 8x7B, SDXL/Diffusion-2 (MaxDiffusion)
inference/
  ironwood/vLLM/                   ← vLLM-on-TPU recipes for v7x
  trillium/                        ← v6e inference recipes
  v5e/                             ← v5e inference recipes
microbenchmarks/
  benchmark_matmul.py              ← measures TFLOPS for [B,K]×[K,N] matmul (bf16/fp8/int8); host-time and trace-time modes
  benchmark_hbm.py                 ← measures HBM bandwidth (GB/s) via copy kernel
  benchmark_utils.py
  trillium/                        ← per-generation knob settings
  requirements.txt
utils/
  profile_convert.py               ← xplane.pb → human-readable
  xplane_pb2.py                    ← generated protobuf bindings
  ironwood/                        ← Ironwood-specific helpers
```

Each leaf training-recipe directory has the same shape:

```
training/<gen>/<model>/<seq-precision-topology>/
  k8s/
    README.md                      ← step-by-step k8s deploy (kubectl apply)
    k8s_manifest.yaml              ← full Kubernetes Job manifest (image, command, resource requests)
  xpk/
    README.md                      ← step-by-step XPK deploy (`xpk workload create`)
    run_recipe.sh                  ← single shell entry point
```

Recipe `README.md` files contain:
1. **Pinned MaxText tag**: e.g. `tpu-recipes-v0.1.4` — the recipe is reproducible only against that exact MaxText commit.
2. **Pinned JAX-stable-stack image**: e.g. `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.6.1-rev1`.
3. **The launch command**: `python3 -m benchmarks.benchmark_runner xpk --model_name=<...> --device_type=<...>`.
4. **Expected step time / TFLOP/s/device / Tokens/s/device** — the reproduction target.
5. **The full MaxText `tuning_params` block** — the set of config knobs the perf team chose.

## Key abstractions

- **MaxText `MaxTextModel(model_name=..., tuning_params={...})` blocks** — the actual data carried by tpu-recipes. Each model has a constellation of these for different (seq-len, precision, topology) cells.
- **`benchmark_runner xpk`** — MaxText's recipe-driven launcher; the same script consumes the `model_name` and topology and produces the same workload across recipes.
- **GKE / XPK launchers** — two parallel paths for the same workload: vanilla Kubernetes manifests vs. XPK (`xpk workload create`). Same docker image, same MaxText invocation, just packaged differently.

## Entry points

- For **training perf**: pick the closest matching recipe under `training/<gen>/<model>/<seq-precision-topology>/`; read its `xpk/README.md` for the launch command and `tuning_params`.
- For **microbenchmarks**: `microbenchmarks/benchmark_matmul.py` and `benchmark_hbm.py`. Each runs on a v6e-1 TPU VM with `pip install -r requirements.txt`. Outputs ms-and-TFLOPS / ms-and-GB/s lines that establish the hardware roofline for that chip.
- For **profile post-processing**: `utils/profile_convert.py` converts an `.xplane.pb` trace to text — useful when running outside the xprof browser.

## Dependencies

- **MaxText** (`AI-Hypercomputer/maxtext`) — the trainer; pinned per-recipe by tag (e.g. `tpu-recipes-v0.1.4`).
- **MaxDiffusion** (`AI-Hypercomputer/maxdiffusion`) — for `*-MaxDiffusion` recipes (SDXL, Diffusion-2).
- **XPK** (`AI-Hypercomputer/xpk`) — Google's GKE TPU job launcher; required for the `xpk/` recipe variant.
- **`jax-ai-image/tpu` Docker images** — pinned per-recipe; they bundle the JAX, libtpu, and Python runtime versions that the recipe was tuned for.
- **vLLM** (for `inference/.../vLLM/` recipes).

## Notable files

- `training/MAXTEXT_README.md` — step-by-step MaxText install + Docker build (Python 3.12, `uv venv`, `bash docker_build_dependency_image.sh`).
- `training/XPK_README.md` — GKE cluster + node pool creation via XPK.
- `training/trillium/Llama3.1-70B-MaxText/v6e-32/README.md` — full worked v6e example: `model_name="llama3_1_70b_8192_bs2_bfloat16_no_collective_matmul"`, `per_device_batch_size=2`, `ici_fsdp_parallelism=-1`, `remat_policy="custom"`, `decoder_layer_input: offload`, `query_proj/key_proj/value_proj: offload`, `max_target_length=8192`. Reports `~443 TFLOP/s/device, ~921 Tokens/s/device, ~17.8 s for 34 steps`.
- `training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x4x4/xpk/README.md` — Gemma 4 26B Ironwood recipe (seq=4k, bf16, 64 chips).
- `microbenchmarks/benchmark_matmul.py` — example output `(8192, 8192, 8192) bf16: 1.33 ms, 827 TFLOPS` (with `--xla_tpu_scoped_vmem_limit_kib=65536`).
- `microbenchmarks/benchmark_hbm.py` — example output `33 MiB copy: 0.049 ms, 1359 GB/s`.
- `utils/profile_convert.py` + `utils/xplane_pb2.py` — generated protobuf bindings + a converter script.

## Performance-relevant surfaces

This is the page's main payload — these are the surfaces optimization hypotheses on this wiki should consult against.

| Surface | What it provides | When to consult |
|---|---|---|
| `training/trillium/Llama3.1-70B-MaxText/v6e-32/README.md` `tuning_params` block | Reference Llama 3.1 70B tuning on v6e-32: `remat_policy: custom`, `decoder_layer_input: offload`, `q/k/v_proj: offload`, `per_device_batch_size: 2`, `max_target_length: 8192`. | Any 70B-class FSDP tuning hypothesis on v6e. |
| `training/trillium/Llama3.1-8B-MaxText/v6e-32/README.md` and similar | Llama 3.1 8B per-topology settings (v6e-32/64/128/256). | Smaller-model FSDP scaling hypotheses; closer scale to gemma4 E4B. |
| `training/trillium/Gemma3-12B-MaxText/` recipes | Gemma-3 12B per-topology MaxText configs. | Direct precedent for any Gemma 4 hypothesis (closest available family-and-scale match). |
| `training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x4x4/` | Gemma 4 26B Ironwood 4k bf16 on 64 chips. | Forward reference for v7x; not directly applicable to v6e but shows how the perf team scaled Gemma 4 specifically. |
| `training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x4x4/` and `4x8x8` (256-chip) variants | bf16 and fp8 Llama 3.1 70B on Ironwood. | Comparison: what changed in Gemma's tuning when moving v6e→v7x. |
| `training/trillium/Mixtral-8x7B-MaxText/` and `Mixtral-8x22B-MaxText/` | MoE recipes — megablox + expert-parallelism settings. | Any MoE hypothesis. |
| `training/trillium/GPT3-175B-MaxText/` | 175B-class FSDP + checkpoint settings on v6e-256. | Largest-scale precedent for FSDP+remat tuning. |
| `microbenchmarks/benchmark_matmul.py` | Per-shape, per-precision matmul TFLOPS on a single chip. Use to establish the chip-level roofline before reading op-profile. | Any roofline-classification dispute (is this op MXU-bound?). |
| `microbenchmarks/benchmark_hbm.py` | Per-tensor-size HBM bandwidth on a single chip. | Any HBM-bound classification check; calibrating the memory-bound roof. |
| `inference/ironwood/vLLM/` | vLLM-on-TPU launch recipes. | Cross-reference with [tpu-inference](tpu-inference.md) when asking inference questions. |
| `utils/profile_convert.py` | Convert `.xplane.pb` to text outside the xprof UI. | When debugging an xprof trace where the browser UI isn't enough. |

**The `tuning_params` block format itself** — every leaf README contains it copied from the corresponding `MaxText@<tag>` benchmark file. This means the **set of MaxText config knobs Google tunes** is implicitly cataloged here: `remat_policy`, `decoder_layer_input`, `query_proj`/`key_proj`/`value_proj` offload, `ici_fsdp_parallelism`, `ici_tensor_parallelism`, `per_device_batch_size`, `max_target_length`, `dtype`, `weight_dtype`, `gradient_accumulation_steps`, plus per-topology adjustments. Any hypothesis page on this wiki touching one of those knobs should cite the closest tpu-recipes recipe as the prior baseline.

**Hardware coverage**: v5p (legacy), v6e (Trillium), v7x (Ironwood). v6e is the most populated and the most relevant to this wiki's gemma4 program (v6e-4 / v6e-32 are within a constant of the Trillium recipes here).

## Connections

- **[maxtext](maxtext.md)** — every `*-MaxText` recipe pins a specific `MaxText@<tag>` and uses `benchmarks.benchmark_runner` from MaxText to launch. The recipe is meaningless without the trainer.
- **[maxdiffusion](maxdiffusion.md)** — `SDXL-MaxDiffusion`, `Diffusion-2-MaxDiffusion` recipes use MaxDiffusion as the trainer.
- **[tpu-inference](tpu-inference.md)** — `inference/ironwood/vLLM/` builds against vLLM's TPU backend.
- **[xprof](xprof.md)** — `utils/profile_convert.py` consumes `.xplane.pb` (xprof's protobuf trace format).
- **gemma4 program** ([wiki/experiments/gemma4_autoresearch_optimization/](../experiments/gemma4_autoresearch_optimization/)) — the `Gemma3-12B-MaxText/v6e-*` recipes are the closest external precedent for the gemma4 E4B optimization here. Hypotheses involving `remat_policy: custom` or per-projection offload should reference these as prior baselines (see exp 36 for the comparable per-projection-offload pattern this wiki landed on independently).

## See also

- [maxtext](maxtext.md) — the trainer all `*-MaxText` recipes invoke; `tuning_params` blocks live in MaxText's `benchmarks/maxtext_v6e_model_configs.py` (and Ironwood/Trillium/v5p siblings).
- [maxdiffusion](maxdiffusion.md) — diffusion-recipe trainer.
- [xprof](xprof.md) — `utils/profile_convert.py` consumer.

## Sources

- `raw/code/tpu-recipes/README.md` — repo overview.
- `raw/code/tpu-recipes/training/MAXTEXT_README.md` — MaxText install + Docker build prerequisite.
- `raw/code/tpu-recipes/training/trillium/Llama3.1-70B-MaxText/v6e-32/README.md` — worked example of the recipe format.
- `raw/code/tpu-recipes/training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x4x4/xpk/README.md` — sample Ironwood recipe.
- `raw/code/tpu-recipes/microbenchmarks/README.md` — microbenchmark harness usage.
