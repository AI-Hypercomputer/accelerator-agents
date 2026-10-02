---
title: 'Module: tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.ops.scaled_dot_product_attention`/
symbols:
  scaled_dot_product_attention: scaled_dot_product_attention().
  vllm_vit_sdpa: vllm_vit_sdpa().
---
# Module: [`tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py)

## Functions
- `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)` — [`L26`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py#L26) — The same args as torch.nn.functional.scaled_dot_product_attention to use flash attention.
- `vllm_vit_sdpa(query, key, value, scale=None, cu_seqlens=None, enable_gqa=False)` — [`L95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py#L95) — Custom JAX implementation of ViT SDPA as an alternative of [upstream vLLM SDPA](https://github.com/vllm-project/vllm/blob/bcc2306cefa4179c548d3e638e7a22a88d281733/vllm/v1/attention/ops/vit_attn_wrappers.py#L211-L239) implementation.

