---
title: 'Module: python/sgl_jax/srt/lora/backend/bgmv_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/backend/bgmv_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.backend.bgmv_backend`/
symbols:
  BgmvLoRABackend.prepare_lora_batch: BgmvLoRABackend#prepare_lora_batch().
  BgmvLoRABackend.run_qkv_lora: BgmvLoRABackend#run_qkv_lora().
  BgmvLoRABackend.run_gate_up_lora: BgmvLoRABackend#run_gate_up_lora().
  BgmvLoRABackend: BgmvLoRABackend#
  expand: expand().
  bgmv_shrink: bgmv_shrink().
  BgmvLoRABackend.run_lora_b_gemm: BgmvLoRABackend#run_lora_b_gemm().
  shrink: shrink().
  bgmv_expand_slice: bgmv_expand_slice().
  BgmvLoRABackend.max_lora_rank: BgmvLoRABackend#max_lora_rank.
  BgmvLoRABackend.__init__: BgmvLoRABackend#__init__().
  BgmvLoRABackend.run_lora_a_gemm: BgmvLoRABackend#run_lora_a_gemm().
  bgmv_jax: bgmv_jax().
  MIN_CHUNK_SIZE: MIN_CHUNK_SIZE.
  BgmvLoRABackend.name: BgmvLoRABackend#name.
---
# Module: [`python/sgl_jax/srt/lora/backend/bgmv_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py)

## Classes
### `BgmvLoRABackend`  ·  implements/extends BaseLoRABackend
- def: [`python/sgl_jax/srt/lora/backend/bgmv_backend.py:30`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L30)
- doc: Bgmv LoRA backend using batched grouped matrix-vector multiplication.
- signature: `class BgmvLoRABackend(BaseLoRABackend):`
- members:
  - `prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch, weight_indices: list[int], lora_ranks: list[int], scalings: list[float])` — [`L214`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L214)
  - `run_gate_up_lora(self, x: jax.Array, gate_up_lora_a: jax.Array, gate_up_lora_b: jax.Array | tuple[jax.Array], base_output: jax.Array, lora_a_output_sharding: NamedSharding, lora_b_output_sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L163`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L163) — Run the lora pass for gate_up_proj.
  - `run_lora_a_gemm(self, x: jax.Array, weights: jax.Array, sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L45`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L45) — Run gemm of lora a modules with current backend.
  - `run_lora_b_gemm(self, x: jax.Array, weights: jax.Array, base_output: jax.Array, sharding: NamedSharding, token_indices: jax.Array, *args, **kwargs)` — [`L69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L69) — Run gemm of lora b modules with current backend.
  - `run_qkv_lora(self, x: jax.Array, qkv_lora_a: jax.Array, qkv_lora_b: jax.Array | tuple[jax.Array], output_slices: tuple, base_output: jax.Array, lora_a_output_sharding: NamedSharding, lora_b_output_sharding: NamedSharding, scalings: jax.Array, token_indices: jax.Array, *args, **kwargs)` — [`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L103) — Run the lora pass for QKV Layer.
  - `max_lora_rank` — [`L43`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L43)
  - `name` — [`L35`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L35)
- protocol/private: `__init__`[`L37`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L37)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `seq_lens`, `forward_mode`, [`EXTEND`](../../model_executor/forward_batch_info.md#ForwardMode.EXTEND), `extend_seq_lens`, [`DECODE`](../../model_executor/forward_batch_info.md#ForwardMode.DECODE), `input_ids`, [`BaseLoRABackend`](base_backend.md#BaseLoRABackend), [`bgmv_shrink`](bgmv_backend.md#bgmv_shrink), [`expand`](bgmv_backend.md#expand), `lora_scalings`, [`shrink`](bgmv_backend.md#shrink), `lora_ranks`, `lora_token_indices`, [`__init__`](base_backend.md#BaseLoRABackend.__init__)
- used by: [`load_lora_weights`](../lora_manager.md#LoRAManager.load_lora_weights), [`prepare_dynamic_lora_batch`](../lora_manager.md#LoRAManager.prepare_dynamic_lora_batch), [`lora_backend`](../lora_manager.md#LoRAManager.lora_backend), [`prepare_static_lora_batch`](../lora_manager.md#LoRAManager.prepare_static_lora_batch), [`BaseLoRABackend`](base_backend.md#BaseLoRABackend), [`_replace_with_lora`](../lora_manager.md#LoRAManager._replace_with_lora), [`prepare_lora_batch`](base_backend.md#BaseLoRABackend.prepare_lora_batch), [`run_lora_a_gemm`](base_backend.md#BaseLoRABackend.run_lora_a_gemm), [`run_lora_b_gemm`](base_backend.md#BaseLoRABackend.run_lora_b_gemm), [`run_gate_up_lora`](base_backend.md#BaseLoRABackend.run_gate_up_lora), [`run_qkv_lora`](base_backend.md#BaseLoRABackend.run_qkv_lora)

## Functions
- `bgmv_expand_slice(inputs, lora_weights, output_array, lora_indices, slice_offset: int, slice_size: int, sharding: NamedSharding)` — [`L343`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L343) — Expand operation: maps from low-rank space to output space.
- `bgmv_jax(inputs, loras, idxs, sharding)` — [`L374`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L374) — Batched grouped matrix-vector multiplication.
- `bgmv_shrink(inputs, lora_weights, lora_indices, sharding: NamedSharding, scaling: float = 1)` — [`L321`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L321) — Shrink operation: maps input to low-rank space.
- `expand(x: jax.Array, lora_b_stacked: jax.Array, token_lora_indices: jax.Array, output_slices: tuple, max_lora_rank: int, sharding: NamedSharding)` — [`L281`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L281) — Optimized: Loop with slicing.
- `shrink(x: jax.Array, lora_a_stacked: jax.Array, token_lora_indices: jax.Array, scalings: jax.Array, sharding: NamedSharding)` — [`L271`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L271)

## Module values
- `MIN_CHUNK_SIZE` — [`L27`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/backend/bgmv_backend.py#L27)

