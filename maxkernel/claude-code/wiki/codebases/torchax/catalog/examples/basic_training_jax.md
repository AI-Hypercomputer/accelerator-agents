---
title: 'Module: examples/basic_training_jax.py'
type: catalog
provenance: extracted
module: examples/basic_training_jax.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.basic_training_jax`/
symbols:
  avg_loss: avg_loss.
  train_step: train_step.
  opt_state: opt_state.
  model_jittable: model_jittable.
  weights: weights.
  GarmentClassifier.forward: GarmentClassifier#forward().
  model_fn: model_fn.
  train_one_epoch: train_one_epoch().
  classes: classes.
  epoch: epoch.
  training_set: training_set.
  validation_set: validation_set.
  training_loader: training_loader.
  model: model.
  jax_optimizer: jax_optimizer.
  buffers: buffers.
  writer: writer.
  env: env.
  validation_loader: validation_loader.
  GarmentClassifier.__init__: GarmentClassifier#__init__().
  transform: transform.
  GarmentClassifier: GarmentClassifier#
  epoch_number: epoch_number.
  GarmentClassifier.fc1: GarmentClassifier#fc1.
  GarmentClassifier.fc2: GarmentClassifier#fc2.
  GarmentClassifier.fc3: GarmentClassifier#fc3.
  loss_fn: loss_fn.
  timestamp: timestamp.
  EPOCHS: EPOCHS.
  dummy_inputs: dummy_inputs.
  dummy_outputs: dummy_outputs.
  dummy_labels: dummy_labels.
  best_vloss: best_vloss.
---
# Module: [`examples/basic_training_jax.py`](../../../../../raw/code/torchax/examples/basic_training_jax.py)

## Classes
### `GarmentClassifier`  ·  implements/extends Module
- def: [`examples/basic_training_jax.py:80`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L80)
- signature: `class GarmentClassifier(nn.Module):`
- members:
  - `forward(self, x)` — [`L87`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L87)
  - `fc1` — [`L83`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L83)
  - `fc2` — [`L84`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L84)
  - `fc3` — [`L85`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L85)
- protocol/private: `__init__`[`L81`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L81)
- used by: (1 test-only callers)

## Functions
- `train_one_epoch(weights, buffers, opt_state, epoch_index, tb_writer)` — [`L122`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L122)

## Module values
- `EPOCHS` — [`L153`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L153)
- `avg_loss` — [`L159`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L159)
- `best_vloss` — [`L154`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L154)
- `buffers` — [`L103`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L103)
- `classes` — [`L57`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L57)
- `dummy_inputs` — [`L114`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L114)
- `dummy_labels` — [`L117`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L117)
- `dummy_outputs` — [`L115`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L115)
- `env` — [`L36`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L36)
- `epoch` — [`L156`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L156)
- `epoch_number` — [`L152`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L152)
- `jax_optimizer` — [`L98`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L98)
- `loss_fn` — [`L96`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L96)
- `model` — [`L95`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L95)
- `model_fn` — [`L106`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L106)
- `model_jittable` — [`L101`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L101)
- `opt_state` — [`L105`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L105)
- `timestamp` — [`L150`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L150)
- `train_step` — [`L108`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L108)
- `training_loader` — [`L51`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L51)
- `training_set` — [`L43`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L43)
- `transform` — [`L38`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L38)
- `validation_loader` — [`L52`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L52)
- `validation_set` — [`L46`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L46)
- `weights` — [`L102`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L102)
- `writer` — [`L151`](../../../../../raw/code/torchax/examples/basic_training_jax.py#L151)

