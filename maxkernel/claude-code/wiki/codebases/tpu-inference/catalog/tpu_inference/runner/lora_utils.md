---
title: 'Module: tpu_inference/runner/lora_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/lora_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.lora_utils`/
symbols:
  LoraUtils.set_active_loras: LoraUtils#set_active_loras().
  LoraUtils.extract_lora_metadata: LoraUtils#extract_lora_metadata().
  LoraUtils.runner: LoraUtils#runner.
  replace_lora_metadata: replace_lora_metadata().
  LoraUtils: LoraUtils#
  LoraUtils.__init__: LoraUtils#__init__().
---
# Module: [`tpu_inference/runner/lora_utils.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py)

## Classes
### `LoraUtils`
- def: [`tpu_inference/runner/lora_utils.py:33`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py#L33)
- signature: `class LoraUtils:`
- members:
  - `extract_lora_metadata(self)` — [`L74`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py#L74)
  - `set_active_loras(self, num_scheduled_tokens_per_req, total_num_scheduled_tokens, padded_total_num_scheduled_tokens)` — [`L38`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py#L38)
  - `runner` — [`L36`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py#L36)
- protocol/private: `__init__`[`L35`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py#L35)
- uses (calls/refs, reference-scoped): `input_batch`, `state`, `state_leaves`, `TPUModelRunner`, [`make_lora_inputs`](input_batch.md#InputBatch.make_lora_inputs), `model`, `lora_config`, [`update_lora`](../layers/vllm/process_weights/cleanup_sharding.md#update_lora)
- used by: `_execute_model`, [`_precompile_continue_decode`](compilation_manager.md#CompilationManager._precompile_continue_decode), `_prepare_inputs`, `_execute_continue_decode`, [`_precompile_backbone_helper`](compilation_manager.md#CompilationManager._precompile_backbone_helper), [`_precompile_compute_logits`](compilation_manager.md#CompilationManager._precompile_compute_logits), `lora_utils`

## Functions
- `replace_lora_metadata(model, metadata: dict, lora_config)` — [`L99`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/lora_utils.py#L99)

