---
title: 'Module: tpu_inference/layers/vllm/process_weights/cleanup_sharding.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/process_weights/cleanup_sharding.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.process_weights.cleanup_sharding`/
symbols:
  MODULE_TYPE_TO_SHARDING_FUNC: MODULE_TYPE_TO_SHARDING_FUNC.
  shard_model_to_tpu: shard_model_to_tpu().
  _shard_column_linear_lora: _shard_column_linear_lora().
  _shard_tensor_to_tpu_replicated: _shard_tensor_to_tpu_replicated().
  _convert_to_torchax_and_shard: _convert_to_torchax_and_shard().
  _shard_module_to_tpu: _shard_module_to_tpu().
  update_lora: update_lora().
  _shard_base_linear_lora_replicated: _shard_base_linear_lora_replicated().
  _tensor_is_in_cpu: _tensor_is_in_cpu().
  logger: logger.
  _shard_qkv_linear_lora: _shard_qkv_linear_lora().
  _shard_merged_column_parallel_linear_lora: _shard_merged_column_parallel_linear_lora().
  _shard_merged_qkv_parallel_linear_lora: _shard_merged_qkv_parallel_linear_lora().
  _shard_row_parallel_linear_lora: _shard_row_parallel_linear_lora().
  _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias: _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias().
  P: P.
  _extract_all_params_buffers: _extract_all_params_buffers().
---
# Module: [`tpu_inference/layers/vllm/process_weights/cleanup_sharding.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py)

## Functions
- `_convert_to_torchax_and_shard(tensor: torch.Tensor, sharding: NamedSharding)` — [`L95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L95)
- `_extract_all_params_buffers(model: torch.nn.Module)` — [`L83`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L83)
- `_replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(module: torch.nn.Module, mesh: Mesh)` — [`L182`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L182)
- `_shard_base_linear_lora_replicated(layer: BaseLinearLayerWithLoRA, mesh: Mesh)` — [`L124`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L124)
- `_shard_column_linear_lora(layer: ColumnParallelLinearWithLoRA, mesh: Mesh)` — [`L140`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L140)
- `_shard_merged_column_parallel_linear_lora(layer: MergedColumnParallelLinearWithLoRA, mesh: Mesh)` — [`L167`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L167)
- `_shard_merged_qkv_parallel_linear_lora(layer: MergedQKVParallelLinearWithLoRA, mesh: Mesh)` — [`L172`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L172)
- `_shard_module_to_tpu(model: torch.nn.Module, mesh: Mesh)` — [`L212`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L212)
- `_shard_qkv_linear_lora(layer: ColumnParallelLinearWithLoRA, mesh: Mesh)` — [`L162`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L162)
- `_shard_row_parallel_linear_lora(layer: RowParallelLinearWithLoRA, mesh: Mesh)` — [`L177`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L177)
- `_shard_tensor_to_tpu_replicated(tensor: torch.Tensor, mesh: Mesh)` — [`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L119)
- `_tensor_is_in_cpu(tensor: torch.tensor)` — [`L87`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L87)
- `shard_model_to_tpu(model: torch.nn.Module, mesh: Mesh)` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L42) — Shard the model weights and move them to TPU.
- `update_lora(model: torch.nn.Module, initial_params_buffers)` — [`L71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L71)

## Module values
- `MODULE_TYPE_TO_SHARDING_FUNC` — [`L200`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L200)
- `P` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L37)
- `logger` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/process_weights/cleanup_sharding.py#L39)

