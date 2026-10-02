---
title: 'Module: python/sgl_jax/srt/lora/context_manager.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/context_manager.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.context_manager`/
symbols:
  LoraBatchContext.set_batch: LoraBatchContext#set_batch().
  LoraBatchContext.get_batch: LoraBatchContext#get_batch().
  LoraBatchContext: LoraBatchContext#
  _lora_context: _lora_context.
---
# Module: [`python/sgl_jax/srt/lora/context_manager.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/context_manager.py)

## Classes
### `LoraBatchContext`
- def: [`python/sgl_jax/srt/lora/context_manager.py:9`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/context_manager.py#L9)
- signature: `class LoraBatchContext:`
- members:
  - `get_batch()` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/context_manager.py#L20)
  - `set_batch(forward_batch: ForwardBatch)` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/context_manager.py#L12)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), [`_lora_context`](context_manager.md#_lora_context)
- used by: [`__call__`](layers.md#LoRALinear.__call__), [`jitted_run_model`](../model_executor/model_runner.md#ModelRunner.jitted_run_model)

## Module values
- `_lora_context` — [`L6`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/context_manager.py#L6)

