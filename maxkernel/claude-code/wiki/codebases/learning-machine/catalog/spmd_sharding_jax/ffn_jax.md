---
title: 'Module: spmd_sharding_jax/ffn_jax.py'
type: catalog
provenance: extracted
module: spmd_sharding_jax/ffn_jax.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `spmd_sharding_jax.ffn_jax`/
symbols:
  StackedFFN.layers: StackedFFN#layers.
  main: main().
  RandomTensorDataset.__iter__: RandomTensorDataset#__iter__().
  FeedForwardNetwork.__call__: FeedForwardNetwork#__call__().
  StackedFFN.__call__: StackedFFN#__call__().
  FeedForwardNetwork.dense1: FeedForwardNetwork#dense1.
  FeedForwardNetwork.dense2: FeedForwardNetwork#dense2.
  StackedFFN.out_proj: StackedFFN#out_proj.
  RandomTensorDataset.data: RandomTensorDataset#data.
  RandomTensorDataset.labels: RandomTensorDataset#labels.
  RandomTensorDataset.sharding: RandomTensorDataset#sharding.
  FeedForwardNetwork.mesh: FeedForwardNetwork#mesh.
  FeedForwardNetwork.use_bias: FeedForwardNetwork#use_bias.
  StackedFFN.mesh: StackedFFN#mesh.
  FeedForwardNetwork.hidden_dim: FeedForwardNetwork#hidden_dim.
  FeedForwardNetwork.output_dim: FeedForwardNetwork#output_dim.
  StackedFFN.num_layers: StackedFFN#num_layers.
  StackedFFN.hidden_dim: StackedFFN#hidden_dim.
  StackedFFN.output_dim: StackedFFN#output_dim.
  StackedFFN.out_channels: StackedFFN#out_channels.
  StackedFFN.use_bias: StackedFFN#use_bias.
  RandomTensorDataset: RandomTensorDataset#
  RandomTensorDataset.tensor_shape: RandomTensorDataset#tensor_shape.
  RandomTensorDataset.label_shape: RandomTensorDataset#label_shape.
  FeedForwardNetwork: FeedForwardNetwork#
  StackedFFN: StackedFFN#
  main.loss: main().loss().
  main.train_step: main().train_step().
  RandomTensorDataset.__init__: RandomTensorDataset#__init__().
  RandomTensorDataset.element_count: RandomTensorDataset#element_count.
  FeedForwardNetwork.setup: FeedForwardNetwork#setup().
  StackedFFN.setup: StackedFFN#setup().
  print_hlo: print_hlo().
---
# Module: [`spmd_sharding_jax/ffn_jax.py`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py)

## Classes
### `FeedForwardNetwork`  ·  implements/extends Module
- def: [`spmd_sharding_jax/ffn_jax.py:41`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L41) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
- signature: `class FeedForwardNetwork(nn.Module):`
- members:
  - `setup(self)` — [`L47`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L47)
  - `dense1` — [`L48`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L48)
  - `dense2` — [`L54`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L54)
  - `hidden_dim` — [`L42`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L42) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `mesh` — [`L44`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L44)
  - `output_dim` — [`L43`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L43) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `use_bias` — [`L45`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L45)
- protocol/private: `__call__`[`L60`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L60)
- used by: [`layers`](ffn_jax.md#StackedFFN.layers)

### `RandomTensorDataset`
- def: [`spmd_sharding_jax/ffn_jax.py:17`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L17) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
- signature: `class RandomTensorDataset:`
- members:
  - `data` — [`L28`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L28) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `element_count` — [`L21`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L21)
  - `label_shape` — [`L20`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L20)
  - `labels` — [`L30`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L30) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `sharding` — [`L22`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L22) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `tensor_shape` — [`L19`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L19)
- protocol/private: `__init__`[`L18`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L18), `__iter__`[`L33`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L33)
- used by: [`main`](ffn_jax.md#main)

### `StackedFFN`  ·  implements/extends Module
- def: [`spmd_sharding_jax/ffn_jax.py:67`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L67) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
- signature: `class StackedFFN(nn.Module):`
- members:
  - `setup(self)` — [`L75`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L75)
  - `hidden_dim` — [`L69`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L69)
  - `layers` — [`L76`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L76) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `mesh` — [`L72`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L72)
  - `num_layers` — [`L68`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L68) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `out_channels` — [`L71`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L71)
  - `out_proj` — [`L85`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L85) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
  - `output_dim` — [`L70`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L70)
  - `use_bias` — [`L73`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L73)
- protocol/private: `__call__`[`L91`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L91)
- uses (calls/refs, reference-scoped): [`mesh`](ffn_jax.md#FeedForwardNetwork.mesh), [`use_bias`](ffn_jax.md#FeedForwardNetwork.use_bias), [`hidden_dim`](ffn_jax.md#FeedForwardNetwork.hidden_dim), [`output_dim`](ffn_jax.md#FeedForwardNetwork.output_dim), [`FeedForwardNetwork`](ffn_jax.md#FeedForwardNetwork)
- used by: [`main`](ffn_jax.md#main)

## Functions
- `loss(params, x, y)` — [`L167`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L167) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
- `main(model_axis=4, num_layers=48, profile_path="gs://lsiyuan-multipod-2/lightricks-ffn-profile/local-v5e-8-tmp")` — [`L105`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L105) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)
- `print_hlo(f, args, post_opt=False)` — [`L99`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L99)
- `train_step(params, dummy_x, dummy_y, opt_state)` — [`L182`](../../../../../raw/code/learning-machine/spmd_sharding_jax/ffn_jax.py#L182) — documented in [spmd_sharding_jax-ffn_jax](../../concepts/spmd_sharding_jax-ffn_jax.md)

