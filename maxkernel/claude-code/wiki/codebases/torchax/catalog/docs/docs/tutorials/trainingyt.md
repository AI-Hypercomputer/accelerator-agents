---
title: 'Module: docs/docs/tutorials/trainingyt.py'
type: catalog
provenance: extracted
module: docs/docs/tutorials/trainingyt.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `docs.docs.tutorials.trainingyt`/
symbols:
  one_step: one_step.
  loss: loss.
  model: model.
  train_one_epoch: train_one_epoch().
  avg_vloss: avg_vloss.
  grad_fn_jax: grad_fn_jax.
  vloss: vloss.
  vlabels: vlabels.
  GarmentClassifier.forward: GarmentClassifier#forward().
  weights: weights.
  img_grid: img_grid.
  opt_state: opt_state.
  dummy_labels: dummy_labels.
  model_path: model_path.
  avg_loss: avg_loss.
  optimizer: optimizer.
  cpu_state_dict: cpu_state_dict.
  run_model_and_loss: run_model_and_loss().
  training_loader: training_loader.
  epoch: epoch.
  classes: classes.
  grad_fn: grad_fn.
  voutputs: voutputs.
  images: images.
  writer: writer.
  training_set: training_set.
  validation_set: validation_set.
  running_vloss: running_vloss.
  loss_fn: loss_fn.
  epoch_number: epoch_number.
  validation_loader: validation_loader.
  dataiter: dataiter.
  model_fn: model_fn().
  labels: labels.
  vdata: vdata.
  vinputs: vinputs.
  GarmentClassifier.__init__: GarmentClassifier#__init__().
  transform: transform.
  GarmentClassifier: GarmentClassifier#
  GarmentClassifier.pool: GarmentClassifier#pool.
  dummy_outputs: dummy_outputs.
  ckpt_dir: ckpt_dir.
  i: i.
  matplotlib_imshow: matplotlib_imshow().
  GarmentClassifier.conv1: GarmentClassifier#conv1.
  GarmentClassifier.conv2: GarmentClassifier#conv2.
  GarmentClassifier.fc1: GarmentClassifier#fc1.
  GarmentClassifier.fc2: GarmentClassifier#fc2.
  GarmentClassifier.fc3: GarmentClassifier#fc3.
  start_learning_rate: start_learning_rate.
  timestamp: timestamp.
  EPOCHS: EPOCHS.
  best_vloss: best_vloss.
---
# Module: [`docs/docs/tutorials/trainingyt.py`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py)

## Classes
### `GarmentClassifier`  ·  implements/extends Module
- def: [`docs/docs/tutorials/trainingyt.py:137`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L137) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- signature: `class GarmentClassifier(nn.Module):`
- members:
  - `forward(self, x)` — [`L147`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L147)
  - `conv1` — [`L140`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L140)
  - `conv2` — [`L142`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L142)
  - `fc1` — [`L143`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L143)
  - `fc2` — [`L144`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L144)
  - `fc3` — [`L145`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L145)
  - `pool` — [`L141`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L141)
- protocol/private: `__init__`[`L138`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L138)
- used by: [`model`](trainingyt.md#model)

## Functions
- `matplotlib_imshow(img, one_channel=False)` — [`L106`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L106)
- `model_fn(weights, buffers, data)` — [`L367`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L367)
- `run_model_and_loss(weights, inputs, labels)` — [`L249`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L249) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `train_one_epoch(epoch_index, tb_writer)` — [`L308`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L308) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)

## Module values
- `EPOCHS` — [`L441`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L441)
- `avg_loss` — [`L450`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L450)
- `avg_vloss` — [`L472`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L472)
- `best_vloss` — [`L443`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L443)
- `ckpt_dir` — [`L500`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L500)
- `classes` — [`L90`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L90)
- `cpu_state_dict` — [`L516`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L516)
- `dataiter` — [`L116`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L116)
- `dummy_labels` — [`L178`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L178) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `dummy_outputs` — [`L176`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L176)
- `epoch` — [`L445`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L445)
- `epoch_number` — [`L439`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L439)
- `grad_fn` — [`L289`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L289) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `grad_fn_jax` — [`L275`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L275) — documented in [torchax-interop](../../../../concepts/torchax-interop.md)
- `i` — [`L460`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L460)
- `images` — [`L117`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L117) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `img_grid` — [`L120`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L120)
- `labels` — [`L117`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L117)
- `loss` — [`L183`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L183)
- `loss_fn` — [`L172`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L172)
- `model` — [`L157`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L157) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `model_path` — [`L501`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L501)
- `one_step` — [`L371`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L371) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `opt_state` — [`L305`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L305) — documented in [torchax-interop](../../../../concepts/torchax-interop.md)
- `optimizer` — [`L208`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L208)
- `running_vloss` — [`L453`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L453)
- `start_learning_rate` — [`L207`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L207)
- `timestamp` — [`L437`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L437)
- `training_loader` — [`L86`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L86) — documented in [docs-docs-tutorials-trainingyt](../../../../concepts/docs-docs-tutorials-trainingyt.md)
- `training_set` — [`L82`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L82)
- `transform` — [`L77`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L77)
- `validation_loader` — [`L87`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L87)
- `validation_set` — [`L83`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L83)
- `vdata` — [`L460`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L460)
- `vinputs` — [`L461`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L461)
- `vlabels` — [`L461`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L461)
- `vloss` — [`L466`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L466)
- `voutputs` — [`L465`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L465)
- `weights` — [`L247`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L247)
- `writer` — [`L438`](../../../../../../../raw/code/torchax/docs/docs/tutorials/trainingyt.py#L438)

