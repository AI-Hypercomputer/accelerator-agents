---
title: "Topic: MaxDiffusion recipes (SDXL + Wan2.x, training + inference)"
type: topic
updated: 2026-07-03
---
# MaxDiffusion recipes (SDXL + Wan2.x, training + inference)

## Overview

MaxDiffusion is this repo's diffusion-model (image/video generation) counterpart to the MaxText
LLM-training family, covering Stable Diffusion XL (SDXL) and the Wan2.x video-generation model
family (Wan2.1-T/I2V-14B, Wan2.2-T/I2V-A14B, and the Wan2.2-T2V-27B text-to-video variant
specifically) across both training (2 recipes, v5p only) and inference (5 recipes, spanning
Trillium, v5e, and Ironwood).

## Training recipes (v5p)

| Model | Recipe |
|---|---|
| Stable Diffusion 2 | (PyTorch/XLA, archived — see [legacy recipes](pytorch-legacy-training-recipes.md)) |
| Stable Diffusion 2 (MaxDiffusion) | [recipe](src:training/v5p/Diffusion-2-MaxDiffusion/README.md#instructions-for-training-stable-diffusion-2-on-tpu-v5p) |
| Stable Diffusion XL (MaxDiffusion) | [recipe](src:training/v5p/SDXL-MaxDiffusion/README.md#instructions-for-training-stable-diffusion-xl-on-tpu-v5p) |

Notably, Stable Diffusion 2 has *both* an archived PyTorch/XLA recipe and an actively-supported
MaxDiffusion (JAX) recipe on the same v5p hardware — the same
PyTorch-legacy-to-JAX-based transition this repo made for LLM training (MaxText replacing the
PyTorch/XLA Llama/Mixtral recipes), applied here to diffusion models.

## Inference recipes

| Model | Hardware | Recipe |
|---|---|---|
| SDXL | Trillium | [recipe](src:inference/trillium/MaxDiffusion/SDXL/README.md#step-2-running-the-inference-benchmark) |
| SDXL | v5e | [recipe](src:inference/v5e/MaxDiffusion/SDXL/README.md#step-2-running-the-inference-benchmark) |
| Wan2.1-T/I2V-14B, Wan2.2-T/I2V-A14B | Trillium | [recipe](src:inference/trillium/MaxDiffusion/Wan2.x/README.md#check-results) |
| Wan2.2-T2V-27B | Trillium | [recipe](src:inference/trillium/MaxDiffusion/Wan2.x/Wan2.2-T2V/README.md#check-results) |
| Wan2.2-T2V-27B | Ironwood | [recipe](src:inference/ironwood/MaxDiffusion/Wan2.x/Wan2.2-T2V/README.md#check-results) |

The Wan2.2-T2V-27B recipe is the only MaxDiffusion inference recipe published on **both** Trillium
and Ironwood with an otherwise-identical structure (XPK cluster creation → Docker image →
prompt-driven test workload → `check-results`), making it this topic's clearest cross-hardware
comparison point, analogous to how the LLM-serving topic pairs Gemma4 across Trillium and Ironwood.

The two SDXL inference recipes (Trillium, v5e) share an unusually generic top-level heading ("Setup")
rather than a descriptive title — both are minimal two-step recipes (install dependencies, run the
inference benchmark) rather than the fuller XPK-cluster-lifecycle recipes used elsewhere.

## See also
- [MaxText training recipe pattern](maxtext-training-recipes.md) — the LLM-training sibling family;
  MaxDiffusion is architecturally analogous but for diffusion/video models.
- [vLLM inference serving recipes](vllm-inference-recipes.md) — the LLM-serving sibling family;
  together these two topics cover all of this repo's inference recipes.

## Sources
- `training/v5p/{Diffusion-2,SDXL}-MaxDiffusion/README.md` (2 recipes)
- `inference/{trillium,v5e,ironwood}/MaxDiffusion/**/README.md` (5 recipes, cited inline above)
