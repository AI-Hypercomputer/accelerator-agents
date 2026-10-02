---
title: "Source: inference/trillium/vLLM/README.md"
type: source-page
doc: inference/trillium/vLLM/README.md
updated: 2026-07-03
---
# inference/trillium/vLLM/README.md

The hub for Trillium (v6e) vLLM serving recipes, framed around a gap: vLLM TPU's new unified backend
(the `vllm-project/tpu-inference` JAX backend) makes out-of-the-box serving possible for any
vLLM-supported model, but a few core components are still being finished — so this repo publishes
stress-tested GCE-based recipes for a specific model set in the meantime: Llama3.x (8B/70B),
Qwen2.5-32B, Qwen2.5-VL-7B, Qwen3 (4B/32B)
([serve-vllm-on-trillium-tpus-v6e](src:inference/trillium/vLLM/README.md#serve-vllm-on-trillium-tpus-v6e)).
GKE-based deployment is pointed at the separate public "Serve an LLM using TPU Trillium on GKE with
vLLM" tutorial rather than duplicated here.

> [!inferred] The "new unified backend" referenced here is the same `tpu_inference` JAX backend
> covered in this wiki's separate `tpu-inference` codebase catalog — the two repos describe the same
> serving stack from the operator (recipe) side and the implementation (kernel/layer) side
> respectively.

## Feeds
- [vLLM inference serving recipes](../topics/vllm-inference-recipes.md) — the four per-model
  recipes this hub links to are exactly that topic's Trillium rows.
