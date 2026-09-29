---
title: 'Module: tpu_inference/distributed/jax_parallel_state.py'
type: catalog
provenance: extracted
module: tpu_inference/distributed/jax_parallel_state.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.distributed.jax_parallel_state`/
symbols:
  get_pp_group: get_pp_group().
  GroupCoordinator.is_first_rank: GroupCoordinator#is_first_rank().
  GroupCoordinator.is_last_rank: GroupCoordinator#is_last_rank().
  GroupCoordinator.__init__: GroupCoordinator#__init__().
  init_pp_distributed_environment: init_pp_distributed_environment().
  connect: connect().
  GroupCoordinator.rank_in_group: GroupCoordinator#rank_in_group.
  GroupCoordinator.world_size: GroupCoordinator#world_size.
  GroupCoordinator.transfer_server: GroupCoordinator#transfer_server.
  GroupCoordinator.connection: GroupCoordinator#connection.
  GroupCoordinator.send_tensor_dict: GroupCoordinator#send_tensor_dict().
  GroupCoordinator.recv_tensor_dict: GroupCoordinator#recv_tensor_dict().
  BASE_JAX_PORT: BASE_JAX_PORT.
  GroupCoordinator: GroupCoordinator#
---
# Module: [`tpu_inference/distributed/jax_parallel_state.py`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py)

## Classes
### `GroupCoordinator`
- def: [`tpu_inference/distributed/jax_parallel_state.py:23`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L23)
- doc: Jax ProcessGroup wrapper for a group of Pipeline Parallel processes.
- signature: `class GroupCoordinator:`
- members:
  - `is_first_rank(self)` — [`L52`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L52)
  - `is_last_rank(self)` — [`L56`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L56)
  - `recv_tensor_dict(self, uuid: int, tensor_spec: dict[str, jax.ShapeDtypeStruct])` — [`L47`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L47)
  - `send_tensor_dict(self, uuid: int, tensor_dict: dict[str, jax.Array])` — [`L44`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L44)
  - `connection` — [`L36`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L36)
  - `rank_in_group` — [`L33`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L33)
  - `transfer_server` — [`L35`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L35)
  - `world_size` — [`L34`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L34)
- protocol/private: `__init__`[`L38`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L38)
- used by: [`end_layer`](../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`end_layer`](../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`get_pp_group`](jax_parallel_state.md#get_pp_group), [`make_layers`](../layers/jax/pp_utils.md#make_layers), [`is_last_rank`](../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.is_last_rank), [`is_last_rank`](../models/jax/llama3.md#LlamaModel.is_last_rank), [`is_last_rank`](../models/jax/qwen3_moe.md#Qwen3MoeModel.is_last_rank), [`is_last_rank`](../models/jax/llama4.md#Llama4ForCausalLM.is_last_rank), [`is_last_rank`](../models/jax/qwen3.md#Qwen3Model.is_last_rank), [`is_last_rank`](../models/jax/qwen2.md#Qwen2Model.is_last_rank), [`is_last_rank`](../models/jax/deepseek_v3.md#DeepSeekV3.is_last_rank), [`is_first_rank`](../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.is_first_rank), [`is_last_rank`](../models/jax/gemma4.md#Gemma4Model.is_last_rank), [`is_last_rank`](../models/jax/qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.is_last_rank), [`connect`](jax_parallel_state.md#connect), [`init_pp_distributed_environment`](jax_parallel_state.md#init_pp_distributed_environment), [`is_first_rank`](../models/jax/gemma4.md#Gemma4Model.is_first_rank), [`is_first_rank`](../models/jax/llama3.md#LlamaModel.is_first_rank), [`is_first_rank`](../models/jax/llama4.md#Llama4ForCausalLM.is_first_rank), [`is_first_rank`](../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.is_first_rank), [`is_first_rank`](../models/jax/qwen3_moe.md#Qwen3MoeModel.is_first_rank), [`_patched_forward`](../models/vllm/experimental/qwen3_vl_patcher.md#_patched_forward), [`_patched_qwen3_omni_forward`](../models/vllm/experimental/qwen3_omni_patcher.md#_patched_qwen3_omni_forward), [`is_first_rank`](../models/jax/deepseek_v3.md#DeepSeekV3.is_first_rank), [`is_first_rank`](../models/jax/qwen2.md#Qwen2Model.is_first_rank), [`is_first_rank`](../models/jax/qwen3.md#Qwen3Model.is_first_rank)

## Functions
- `connect(prev_ip: str, prev_rank: int)` — [`L73`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L73)
- `get_pp_group()` — [`L79`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L79)
- `init_pp_distributed_environment(ip: str, rank: int, world_size: int, device: Any, need_pp: bool)` — [`L60`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L60)

## Module values
- `BASE_JAX_PORT` — [`L20`](../../../../../../raw/code/tpu-inference/tpu_inference/distributed/jax_parallel_state.py#L20)

