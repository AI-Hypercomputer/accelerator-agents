---
title: "Topic: vLLM inference serving recipes (Trillium + Ironwood)"
type: topic
updated: 2026-07-03
---
# vLLM inference serving recipes (Trillium + Ironwood)

## Overview

11 recipes cover serving open models with vLLM on TPU: 5 on Trillium (GCE-based single-VM serving)
and 6 on Ironwood (GKE-based, since Ironwood recipes are consistently GKE/k8s-first across this
repo). The [Trillium hub](../sources/inference-trillium-vLLM-README.md) frames these as
stress-tested stopgap recipes for models the still-maturing vLLM TPU unified backend
(`tpu-inference`) doesn't yet serve out-of-the-box with full performance — the recipes exist because
a few core components of that unified backend were still being finished at time of writing.

## Trillium (GCE, single-VM) recipes

All five follow the same ten-ish-step shape: install gcloud → create a v6e TPU VM → SSH in → pull
the latest vLLM TPU Docker image → run the container → set env vars → serve the model → prepare a
test image/environment → test the server → run the benchmark.

| Model | Recipe |
|---|---|
| Gemma4 IT | [recipe](src:inference/trillium/vLLM/Gemma4/README.md#serve-gemma-4-it-with-vllm-on-trillium-tpu-vms) |
| Llama3.x (8B/70B) | [recipe](src:inference/trillium/vLLM/Llama3.x/README.md#serve-llama3x-with-vllm-on-tpu-vms) |
| Qwen2.5-32B | [recipe](src:inference/trillium/vLLM/Qwen2.5-32B/README.md#serve-qwen25-32b-with-vllm-on-tpu-vms) |
| Qwen2.5-VL-7B (multimodal) | [recipe](src:inference/trillium/vLLM/Qwen2.5-VL/README.md#serve-qwen25-vl-with-vllm-on-tpu-vms) |
| Qwen3 (4B/32B) | [recipe](src:inference/trillium/vLLM/Qwen3/README.md#serve-qwen3-with-vllm-on-tpu-vms) |

The Gemma4 recipe alone benchmarks both a text-only and a multimodal (image) path (Option A vs.
Option B), and notes "known limitations with current TPU verification" — the only Trillium vLLM
recipe to flag caveats explicitly.

## Ironwood (GKE) recipes

All six are GKE-cluster-based: define parameters → create/update the TPU node pool → deploy the vLLM
server workload onto GKE → optionally benchmark via a Kubernetes Service.

| Model | Notable variant | Recipe |
|---|---|---|
| Gemma4 IT | — | [recipe](src:inference/ironwood/vLLM/Gemma4/README.md#serve-gemma-4-it-with-vllm-on-ironwood-tpu) |
| GPT-OSS-120B | storage: GCS | [recipe](src:inference/ironwood/vLLM/GPT-OSS/README-gcs.md#serve-gpt-oss-120b-with-vllm-on-ironwood-tpu-with-google-cloud-storage) |
| GPT-OSS-120B | storage: Managed Lustre | [recipe](src:inference/ironwood/vLLM/GPT-OSS/README-lustre.md#serve-gpt-oss-120b-with-vllm-on-ironwood-tpu-with-google-cloud-managed-lustre) |
| Qwen3-32B | smart-routing end-to-end perf | [recipe](src:inference/ironwood/vLLM/Qwen3-32B/README.md#serve-qwen3-32b-with-vllm-on-ironwood-tpu) |
| Qwen3-Coder-480B-A35B-Instruct-FP8 | single-host nodepool, FP8 | [recipe](src:inference/ironwood/vLLM/Qwen3-Coder-480B-A35B/README.md#serve-qwen3-coder-480b-a35b-instruct-fp8-with-vllm-on-ironwood-tpu) |
| Qwen3.5-397B | 1k/8k in/out workload pair | [recipe](src:inference/ironwood/vLLM/Qwen3.5-397B/README.md#serve-qwen35-397b-with-vllm-on-ironwood-tpu) |

GPT-OSS-120B is the only model with a **storage-backend** axis (GCS vs. Managed Lustre) for its
checkpoint — the same axis this repo applies to DeepSeek-V3's *training* recipe on Ironwood (see
[Ironwood training recipe matrix](ironwood-training-recipes.md#storage-backend-as-a-recipe-axis-deepseek-v3-only)),
suggesting checkpoint storage backend choice matters enough at large-model scale to be tuned on both
the training and serving side. Qwen3.5-397B's recipe is also the only one to publish a paired
"workload A (1k in / 8k out)" vs. "workload B (8k in / 1k out)" benchmark comparison — a direct
prefill-heavy vs. decode-heavy serving-shape comparison.

## See also
- [Ironwood (TPU7x) training recipe matrix](ironwood-training-recipes.md) — the training-side
  recipes for the same model families (Gemma4, GPT-OSS, Qwen3-family) on the same hardware.
- [MaxDiffusion recipes](maxdiffusion-recipes.md) — the non-LLM (diffusion) inference recipe family,
  serving a different workload class than the LLM serving covered here.

## Sources
- [inference/trillium/vLLM/README.md](../sources/inference-trillium-vLLM-README.md)
- `inference/{trillium,ironwood}/vLLM/*/README*.md` (11 recipes, cited inline above)
