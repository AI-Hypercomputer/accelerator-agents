---
title: 'Module: python/sgl_jax/srt/lora/backend/base_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/backend/base_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.backend.base_backend`/BaseLoRABackend#
symbols:
  BaseLoRABackend: ''
  BaseLoRABackend.prepare_lora_batch: prepare_lora_batch().
  BaseLoRABackend.run_lora_a_gemm: run_lora_a_gemm().
  BaseLoRABackend.run_lora_b_gemm: run_lora_b_gemm().
  BaseLoRABackend.run_qkv_lora: run_qkv_lora().
  BaseLoRABackend.run_gate_up_lora: run_gate_up_lora().
  BaseLoRABackend.__init__: __init__().
  BaseLoRABackend.max_loras_per_batch: max_loras_per_batch.
---
# Module: [`python/sgl_jax/srt/lora/backend/base_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py)

## Classes
### `BaseLoRABackend`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/lora/backend/base_backend.py:8`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L8)
- doc: Base class for different Lora backends.
- signature: `class BaseLoRABackend(nnx.Module):`
- members:
  - `prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch, weight_indices: list[int], lora_ranks: list[int], scalings: list[float])` — [`L134`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L134) — Prepare the lora weights and batch info for current forward batch.
  - `run_gate_up_lora(self, x: jax.Array, gate_up_lora_a: jax.Array, gate_up_lora_b: jax.Array | tuple[jax.Array], base_output: jax.Array, lora_a_output_sharding: NamedSharding, lora_b_output_sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L103) — Run the lora pass for gate_up_proj.
  - `run_lora_a_gemm(self, x: jax.Array, weights: jax.Array, sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L22`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L22) — Run gemm of lora a modules with current backend.
  - `run_lora_b_gemm(self, x: jax.Array, weights: jax.Array, sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L46) — Run gemm of lora b modules with current backend.
  - `run_qkv_lora(self, x: jax.Array, qkv_lora_a: jax.Array, qkv_lora_b: jax.Array | tuple[jax.Array], output_slices: tuple, lora_a_output_sharding: NamedSharding, lora_b_output_sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L69) — Run the lora pass for QKV Layer.
  - `max_loras_per_batch` — [`L20`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L20)
- protocol/private: `__init__`[`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/base_backend.py#L18)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, [`prepare_lora_batch`](bgmv_backend.md#BgmvLoRABackend.prepare_lora_batch), [`run_gate_up_lora`](bgmv_backend.md#BgmvLoRABackend.run_gate_up_lora), [`run_qkv_lora`](bgmv_backend.md#BgmvLoRABackend.run_qkv_lora), [`BgmvLoRABackend`](bgmv_backend.md#BgmvLoRABackend), [`run_lora_b_gemm`](bgmv_backend.md#BgmvLoRABackend.run_lora_b_gemm), [`run_lora_a_gemm`](bgmv_backend.md#BgmvLoRABackend.run_lora_a_gemm)
- used by: [`apply_lora`](../layers.md#LoRALinear.apply_lora), [`__init__`](../layers.md#LoRAEmbedding.__init__), [`__init__`](../layers.md#LoRALinear.__init__), [`lora_backend`](../layers.md#BaseLayerWithLoRA.lora_backend), [`__init__`](../lora.md#LoRAAdapter.__init__), [`BgmvLoRABackend`](bgmv_backend.md#BgmvLoRABackend), [`__init__`](../layers.md#BaseLayerWithLoRA.__init__), [`__init__`](bgmv_backend.md#BgmvLoRABackend.__init__), [`lora_backend`](../lora.md#LoRAAdapter.lora_backend)

