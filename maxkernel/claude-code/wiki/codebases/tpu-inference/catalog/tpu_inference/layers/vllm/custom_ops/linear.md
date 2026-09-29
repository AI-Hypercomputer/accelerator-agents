---
title: 'Module: tpu_inference/layers/vllm/custom_ops/linear.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/linear.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.linear`/Vllm
symbols:
  VllmRowParallelLinear: RowParallelLinear#
  VllmRowParallelLinear.forward: RowParallelLinear#forward().
  VllmColumnParallelLinear: ColumnParallelLinear#
  VllmColumnParallelLinear.forward: ColumnParallelLinear#forward().
  VllmReplicatedLinear: ReplicatedLinear#
  VllmReplicatedLinear.forward: ReplicatedLinear#forward().
  VllmQKVParallelLinear: QKVParallelLinear#
  VllmQKVParallelLinear.__init__: QKVParallelLinear#__init__().
  VllmQKVParallelLinear.weight_loader: QKVParallelLinear#weight_loader().
  VllmQKVParallelLinear.weight_loader_v2: QKVParallelLinear#weight_loader_v2().
  VllmQKVParallelLinear._tile_kv: QKVParallelLinear#_tile_kv().
  VllmQKVParallelLinear.forward: QKVParallelLinear#forward().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/linear.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py)

## Classes
### `VllmColumnParallelLinear`
- def: [`tpu_inference/layers/vllm/custom_ops/linear.py:45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L45)
- signature: `class VllmColumnParallelLinear(ColumnParallelLinear):`
- members:
  - `forward(self, input_)` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L47)

### `VllmQKVParallelLinear`
- def: [`tpu_inference/layers/vllm/custom_ops/linear.py:65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L65)
- doc: Pre-replicate KV heads when TP > total_num_kv_heads.
- signature: `class VllmQKVParallelLinear(QKVParallelLinear):`
- members:
  - `forward(self, x: torch.Tensor)` — [`L151`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L151)
  - `weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None=None)` — [`L117`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L117)
  - `weight_loader_v2(self, param, loaded_weight: torch.Tensor, loaded_shard_id: str | None=None)` — [`L125`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L125)
- protocol/private: `__init__`[`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L85), `_tile_kv`[`L133`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L133)

### `VllmReplicatedLinear`
- def: [`tpu_inference/layers/vllm/custom_ops/linear.py:55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L55)
- signature: `class VllmReplicatedLinear(ReplicatedLinear):`
- members:
  - `forward(self, x: torch.Tensor)` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L57)

### `VllmRowParallelLinear`
- def: [`tpu_inference/layers/vllm/custom_ops/linear.py:35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L35)
- signature: `class VllmRowParallelLinear(RowParallelLinear):`
- members:
  - `forward(self, input_)` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/linear.py#L37)

