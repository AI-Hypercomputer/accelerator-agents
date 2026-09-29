---
title: 'Module: python/sgl_jax/srt/lora/utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.utils`/
symbols:
  LoRABatchInfo.tree_flatten: LoRABatchInfo#tree_flatten().
  get_target_module_name: get_target_module_name().
  get_lora_a_sharding: get_lora_a_sharding().
  get_lora_b_sharding: get_lora_b_sharding().
  get_lora_a_output_sharding: get_lora_a_output_sharding().
  get_lora_b_output_sharding: get_lora_b_output_sharding().
  get_normalized_target_modules: get_normalized_target_modules().
  LoRABatchInfo.scalings: LoRABatchInfo#scalings.
  LoRABatchInfo.token_lora_indices: LoRABatchInfo#token_lora_indices.
  LoRABatchInfo.lora_ranks: LoRABatchInfo#lora_ranks.
  LoRABatchInfo: LoRABatchInfo#
  LoRABatchInfo.tree_unflatten: LoRABatchInfo#tree_unflatten().
  LoRAType: LoRAType#
  LoRAType.LORA_A: LoRAType#LORA_A.
  LoRAType.LORA_B: LoRAType#LORA_B.
---
# Module: [`python/sgl_jax/srt/lora/utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py)

## Classes
### `LoRABatchInfo`
- def: [`python/sgl_jax/srt/lora/utils.py:13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L13)
- signature: `class LoRABatchInfo:`
- members:
  - `tree_flatten(self)` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L23)
  - `tree_unflatten(cls, aux_data, children)` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L27)
  - `lora_ranks` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L21)
  - `scalings` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L15)
  - `token_lora_indices` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L18)

### `LoRAType`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/lora/utils.py:31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L31)
- signature: `class LoRAType(Enum):`
- members:
  - `LORA_A` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L32)
  - `LORA_B` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L33)

## Functions
- `get_lora_a_output_sharding(module_name: str, mesh: Mesh)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L81)
- `get_lora_a_sharding(module_name: str, mesh: Mesh)` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L49) — Get sharding spec for LoRA A matrix.
- `get_lora_b_output_sharding(module_name: str, mesh: Mesh)` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L85) — Get sharding spec for LoRA B matrix.
- `get_lora_b_sharding(module_name: str, mesh: Mesh)` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L61) — Get sharding spec for LoRA B matrix.
- `get_normalized_target_modules(target_modules: Iterable[str])` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L103) — Mapping a list of target module name to names of the normalized LoRA weights.
- `get_target_module_name(full_module_name: str, target_modules: set[str])` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/utils.py#L36) — Get the target module name in target_modules that can match full_module_name.

