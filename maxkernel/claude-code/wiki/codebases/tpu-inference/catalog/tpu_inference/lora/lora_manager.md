---
title: 'Module: tpu_inference/lora/lora_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/lora/lora_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.lora.lora_manager`/
symbols:
  parse_lora_module_path_env: parse_lora_module_path_env().
  MAXTEXT_TO_HF_LORA_MAPPING: MAXTEXT_TO_HF_LORA_MAPPING.
  TPULRUCacheWorkerLoRAManager: TPULRUCacheWorkerLoRAManager#
  LORA_GROUP_PATTERN: LORA_GROUP_PATTERN.
  LORA_SPLIT_PATTERN: LORA_SPLIT_PATTERN.
  TPULRUCacheWorkerLoRAManager.add_dummy_lora: TPULRUCacheWorkerLoRAManager#add_dummy_lora().
---
# Module: [`tpu_inference/lora/lora_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py)

## Classes
### `TPULRUCacheWorkerLoRAManager`
- def: [`tpu_inference/lora/lora_manager.py:43`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py#L43)
- doc: TPU-specific wrapper to ensure dummy LoRA creation happens
- signature: `class TPULRUCacheWorkerLoRAManager(LRUCacheWorkerLoRAManager):`
- members:
  - `add_dummy_lora(self, lora_request, rank: int)` — [`L49`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py#L49)
- used by: [`load_lora_model`](../models/vllm/vllm_model_wrapper.md#load_lora_model)

## Functions
- `parse_lora_module_path_env()` — [`L54`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py#L54) — Parses LORA_MODULE_PATH env var into vLLM canonical target_modules list.

## Module values
- `LORA_GROUP_PATTERN` — [`L39`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py#L39)
- `LORA_SPLIT_PATTERN` — [`L40`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py#L40)
- `MAXTEXT_TO_HF_LORA_MAPPING` — [`L27`](../../../../../../raw/code/tpu-inference/tpu_inference/lora/lora_manager.py#L27)

