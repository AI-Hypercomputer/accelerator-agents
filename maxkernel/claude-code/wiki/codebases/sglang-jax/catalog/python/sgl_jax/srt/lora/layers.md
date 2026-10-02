---
title: 'Module: python/sgl_jax/srt/lora/layers.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/layers.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.layers`/
symbols:
  LoRALinear.apply_lora: LoRALinear#apply_lora().
  LoRALinear.__call__: LoRALinear#__call__().
  LoRALinear.__init__: LoRALinear#__init__().
  LoRAEmbedding.__init__: LoRAEmbedding#__init__().
  BaseLayerWithLoRA: BaseLayerWithLoRA#
  LoRALinear.lora_b_output_sharding: LoRALinear#lora_b_output_sharding.
  BaseLayerWithLoRA.lora_backend: BaseLayerWithLoRA#lora_backend.
  BaseLayerWithLoRA.__call__: BaseLayerWithLoRA#__call__().
  BaseLayerWithLoRA.__init__: BaseLayerWithLoRA#__init__().
  LoRALinear: LoRALinear#
  BaseLayerWithLoRA.base_layer: BaseLayerWithLoRA#base_layer.
  LoRALinear.A_buffer: LoRALinear#A_buffer.
  BaseLayerWithLoRA.set_lora_info: BaseLayerWithLoRA#set_lora_info().
  LoRALinear.lora_a_output_sharding: LoRALinear#lora_a_output_sharding.
  BaseLayerWithLoRA.weight: BaseLayerWithLoRA#weight.
  LoRAEmbedding: LoRAEmbedding#
  LoRALinear.B_buffer: LoRALinear#B_buffer.
  LoRALinear.set_lora_info: LoRALinear#set_lora_info().
  LoRAEmbedding.weight: LoRAEmbedding#weight.
---
# Module: [`python/sgl_jax/srt/lora/layers.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py)

## Classes
### `BaseLayerWithLoRA`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/lora/layers.py:39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L39)
- signature: `class BaseLayerWithLoRA(nnx.Module):`
- members:
  - `set_lora_info(self, *args)` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L54)
  - `base_layer` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L46)
  - `lora_backend` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L47)
  - `weight` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L49)
- protocol/private: `__call__`[`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L51), `__init__`[`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L40)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), [`BaseLoRABackend`](backend/base_backend.md#BaseLoRABackend), [`LoRALinear`](layers.md#LoRALinear), [`LoRAEmbedding`](layers.md#LoRAEmbedding), [`set_lora_info`](layers.md#LoRALinear.set_lora_info)
- used by: [`lora_modules`](lora_manager.md#LoRAManager.lora_modules), [`update_lora_info`](lora_manager.md#LoRAManager.update_lora_info), [`apply_lora`](layers.md#LoRALinear.apply_lora), [`__call__`](layers.md#LoRALinear.__call__), [`__init__`](layers.md#LoRAEmbedding.__init__), [`__init__`](layers.md#LoRALinear.__init__), [`LoRALinear`](layers.md#LoRALinear), [`LoRAEmbedding`](layers.md#LoRAEmbedding)

### `LoRAEmbedding`  ·  implements/extends BaseLayerWithLoRA
- def: [`python/sgl_jax/srt/lora/layers.py:154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L154)
- doc: LoRA wrapper for Embedding layers.
- signature: `class LoRAEmbedding(BaseLayerWithLoRA):`
- members:
  - `__init__(self, base_layer: LinearBase | None = None, lora_backend: BaseLoRABackend | None = None)` — [`L162`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L162) — Initialize LoRA Embedding layer.
  - `weight` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L175)
- uses (calls/refs, reference-scoped): [`LinearBase`](../layers/linear.md#LinearBase), [`BaseLoRABackend`](backend/base_backend.md#BaseLoRABackend), [`BaseLayerWithLoRA`](layers.md#BaseLayerWithLoRA), [`weight`](../layers/linear.md#LinearBase.weight), [`__init__`](layers.md#BaseLayerWithLoRA.__init__)
- used by: [`BaseLayerWithLoRA`](layers.md#BaseLayerWithLoRA)

### `LoRALinear`  ·  implements/extends BaseLayerWithLoRA
- def: [`python/sgl_jax/srt/lora/layers.py:58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L58)
- doc: LoRA wrapper for Linear layers using Flax NNX.
- signature: `class LoRALinear(BaseLayerWithLoRA):`
- members:
  - `__call__(self, x: jax.Array, *, out_sharding: jax.sharding.Sharding | None = None)` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L124) — Forward pass with optional LoRA computation using backend.
  - `__init__(self, base_layer: LinearBase | None = None, lora_backend: BaseLoRABackend | None = None)` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L69) — Initialize LoRA Linear layer.
  - `apply_lora(self, base_output, x, scalings, token_indices)` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L107)
  - `set_lora_info(self, A_buffer: jax.Array, B_buffer: jax.Array, module_name: str, mesh: Mesh)` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L84)
  - `A_buffer` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L104)
  - `B_buffer` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L105)
  - `lora_a_output_sharding` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L91)
  - `lora_b_output_sharding` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/layers.py#L92)
- uses (calls/refs, reference-scoped): [`LinearBase`](../layers/linear.md#LinearBase), [`BaseLoRABackend`](backend/base_backend.md#BaseLoRABackend), [`BaseLayerWithLoRA`](layers.md#BaseLayerWithLoRA), [`lora_backend`](layers.md#BaseLayerWithLoRA.lora_backend), [`get_batch`](context_manager.md#LoraBatchContext.get_batch), [`__init__`](layers.md#BaseLayerWithLoRA.__init__), [`LoraBatchContext`](context_manager.md#LoraBatchContext), [`base_layer`](layers.md#BaseLayerWithLoRA.base_layer), [`lora_scalings`](../model_executor/forward_batch_info.md#ForwardBatch.lora_scalings), [`lora_token_indices`](../model_executor/forward_batch_info.md#ForwardBatch.lora_token_indices), [`run_lora_a_gemm`](backend/base_backend.md#BaseLoRABackend.run_lora_a_gemm), [`run_lora_b_gemm`](backend/base_backend.md#BaseLoRABackend.run_lora_b_gemm), [`get_lora_a_output_sharding`](utils.md#get_lora_a_output_sharding), [`get_lora_b_output_sharding`](utils.md#get_lora_b_output_sharding)
- used by: [`lora_backend`](lora_manager.md#LoRAManager.lora_backend), [`_replace_with_lora`](lora_manager.md#LoRAManager._replace_with_lora), [`BaseLayerWithLoRA`](layers.md#BaseLayerWithLoRA), [`set_lora_info`](layers.md#BaseLayerWithLoRA.set_lora_info)

