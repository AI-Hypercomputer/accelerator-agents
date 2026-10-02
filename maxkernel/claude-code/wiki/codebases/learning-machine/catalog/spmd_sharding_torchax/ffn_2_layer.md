---
title: 'Module: spmd_sharding_torchax/ffn_2_layer.py'
type: catalog
provenance: extracted
module: spmd_sharding_torchax/ffn_2_layer.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `spmd_sharding_torchax.ffn_2_layer`/
symbols:
  main: main().
  main.FFN.forward: main().FFN#forward().
  sharded_device_put: sharded_device_put().
  make_train_step: make_train_step().
  RandomTensorDataset.__iter__: RandomTensorDataset#__iter__().
  main.Model.forward: main().Model#forward().
  Model.m: Model#m.
  RandomTensorDataset: RandomTensorDataset#
  RandomTensorDataset.tensor_shape: RandomTensorDataset#tensor_shape.
  RandomTensorDataset.element_count: RandomTensorDataset#element_count.
  make_train_step.loss: make_train_step().loss().
  make_train_step.step: make_train_step().step().
  main.FFN: main().FFN#
  FFN.layer1: FFN#layer1.
  FFN.layer2: FFN#layer2.
  FFN.dropout: FFN#dropout.
  main.Model: main().Model#
  Model.output: Model#output.
  main.call_model: main().call_model().
  RandomTensorDataset.__init__: RandomTensorDataset#__init__().
  main.FFN.__init__: main().FFN#__init__().
  main.Model.__init__: main().Model#__init__().
---
# Module: [`spmd_sharding_torchax/ffn_2_layer.py`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py)

## Classes
### `FFN`  ·  implements/extends Module
- def: [`spmd_sharding_torchax/ffn_2_layer.py:108`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L108)
- signature: `class FFN(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L115`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L115)
  - `dropout` — [`L113`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L113)
  - `layer1` — [`L111`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L111)
  - `layer2` — [`L112`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L112)
- protocol/private: `__init__`[`L109`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L109)
- used by: [`m`](ffn_2_layer.md#Model.m)

### `Model`  ·  implements/extends Module
- def: [`spmd_sharding_torchax/ffn_2_layer.py:123`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L123)
- signature: `class Model(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L132`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L132)
  - `m` — [`L127`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L127)
  - `output` — [`L130`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L130)
- protocol/private: `__init__`[`L125`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L125)
- uses (calls/refs, reference-scoped): [`FFN`](ffn_2_layer.md#main.FFN)
- used by: [`main`](ffn_2_layer.md#main)

### `RandomTensorDataset`
- def: [`spmd_sharding_torchax/ffn_2_layer.py:15`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L15)
- signature: `class RandomTensorDataset:`
- members:
  - `element_count` — [`L18`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L18)
  - `tensor_shape` — [`L17`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L17)
- protocol/private: `__init__`[`L16`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L16), `__iter__`[`L20`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L20)
- used by: [`main`](ffn_2_layer.md#main)

## Functions
- `call_model(weights, buffer, args)` — [`L160`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L160)
- `loss(weights, buffers, args, label)` — [`L44`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L44)
- `main(model_axis=4, num_layers=48, profile_dir='/tmp/profile_dir')` — [`L81`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L81)
- `make_train_step(model_fn, loss_fn, optax_optimizer, remat_policy=None)` — [`L24`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L24) — Make a function that do one train step given model and loss.
- `sharded_device_put(tensor, sharding)` — [`L67`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L67)
- `step(weights, buffers, opt_state, args, label)` — [`L53`](../../../../../raw/code/learning-machine/spmd_sharding_torchax/ffn_2_layer.py#L53)

