---
title: 'Module: spmd_sharding/ffn_2_layer.py'
type: catalog
provenance: extracted
module: spmd_sharding/ffn_2_layer.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `spmd_sharding.ffn_2_layer`/
symbols:
  main.FFN.forward: main().FFN#forward().
  main: main().
  RandomTensorDataset.__iter__: RandomTensorDataset#__iter__().
  main.Model.forward: main().Model#forward().
  Model.m: Model#m.
  RandomTensorDataset: RandomTensorDataset#
  RandomTensorDataset.tensor_shape: RandomTensorDataset#tensor_shape.
  RandomTensorDataset.element_count: RandomTensorDataset#element_count.
  main.FFN: main().FFN#
  FFN.layer1: FFN#layer1.
  FFN.layer2: FFN#layer2.
  FFN.dropout: FFN#dropout.
  main.Model: main().Model#
  Model.output: Model#output.
  log_tensor_sharding: log_tensor_sharding().
  RandomTensorDataset.__init__: RandomTensorDataset#__init__().
  main.FFN.__init__: main().FFN#__init__().
  main.Model.__init__: main().Model#__init__().
---
# Module: [`spmd_sharding/ffn_2_layer.py`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py)

## Classes
### `FFN`  ·  implements/extends Module
- def: [`spmd_sharding/ffn_2_layer.py:67`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L67)
- signature: `class FFN(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L74`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L74)
  - `dropout` — [`L72`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L72)
  - `layer1` — [`L70`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L70)
  - `layer2` — [`L71`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L71)
- protocol/private: `__init__`[`L68`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L68)
- used by: [`m`](ffn_2_layer.md#Model.m)

### `Model`  ·  implements/extends Module
- def: [`spmd_sharding/ffn_2_layer.py:82`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L82)
- signature: `class Model(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L91`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L91)
  - `m` — [`L86`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L86)
  - `output` — [`L89`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L89)
- protocol/private: `__init__`[`L84`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L84)
- uses (calls/refs, reference-scoped): [`FFN`](ffn_2_layer.md#main.FFN)
- used by: [`main`](ffn_2_layer.md#main)

### `RandomTensorDataset`
- def: [`spmd_sharding/ffn_2_layer.py:22`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L22)
- signature: `class RandomTensorDataset:`
- members:
  - `element_count` — [`L25`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L25)
  - `tensor_shape` — [`L24`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L24)
- protocol/private: `__init__`[`L23`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L23), `__iter__`[`L27`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L27)
- used by: [`main`](ffn_2_layer.md#main)

## Functions
- `log_tensor_sharding(t: torch.Tensor, log_prefix: str)` — [`L14`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L14)
- `main(model_axis=2, fsdp_axis=32, num_layers=48, profile_dir='/tmp/profile_dir', use_fsdp_wrapper=False, per_device_batch_size=None)` — [`L32`](../../../../../raw/code/learning-machine/spmd_sharding/ffn_2_layer.py#L32)

