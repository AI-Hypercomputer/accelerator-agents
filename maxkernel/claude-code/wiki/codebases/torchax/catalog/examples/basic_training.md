---
title: 'Module: examples/basic_training.py'
type: catalog
provenance: extracted
module: examples/basic_training.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.basic_training`/
symbols:
  img_grid: img_grid.
  vloss: vloss.
  avg_vloss: avg_vloss.
  loss: loss.
  train_one_epoch: train_one_epoch().
  vlabels: vlabels.
  GarmentClassifier.forward: GarmentClassifier#forward().
  model: model.
  epoch: epoch.
  dummy_labels: dummy_labels.
  classes: classes.
  avg_loss: avg_loss.
  voutputs: voutputs.
  training_loader: training_loader.
  training_set: training_set.
  validation_set: validation_set.
  optimizer: optimizer.
  running_vloss: running_vloss.
  validation_loader: validation_loader.
  dataiter: dataiter.
  labels: labels.
  vdata: vdata.
  loss_fn: loss_fn.
  epoch_number: epoch_number.
  vinputs: vinputs.
  GarmentClassifier.__init__: GarmentClassifier#__init__().
  transform: transform.
  GarmentClassifier: GarmentClassifier#
  dummy_outputs: dummy_outputs.
  matplotlib_imshow: matplotlib_imshow().
  GarmentClassifier.fc1: GarmentClassifier#fc1.
  GarmentClassifier.fc2: GarmentClassifier#fc2.
  GarmentClassifier.fc3: GarmentClassifier#fc3.
  EPOCHS: EPOCHS.
  images: images.
  i: i.
  best_vloss: best_vloss.
---
# Module: [`examples/basic_training.py`](../../../../../raw/code/torchax/examples/basic_training.py)

## Classes
### `GarmentClassifier`  ·  implements/extends Module
- def: [`examples/basic_training.py:103`](../../../../../raw/code/torchax/examples/basic_training.py#L103)
- signature: `class GarmentClassifier(nn.Module):`
- members:
  - `forward(self, x)` — [`L110`](../../../../../raw/code/torchax/examples/basic_training.py#L110)
  - `fc1` — [`L106`](../../../../../raw/code/torchax/examples/basic_training.py#L106)
  - `fc2` — [`L107`](../../../../../raw/code/torchax/examples/basic_training.py#L107)
  - `fc3` — [`L108`](../../../../../raw/code/torchax/examples/basic_training.py#L108)
- protocol/private: `__init__`[`L104`](../../../../../raw/code/torchax/examples/basic_training.py#L104)
- used by: (1 test-only callers)

## Functions
- `matplotlib_imshow(img, one_channel=False)` — [`L77`](../../../../../raw/code/torchax/examples/basic_training.py#L77)
- `train_one_epoch(epoch_index, tb_writer=None)` — [`L138`](../../../../../raw/code/torchax/examples/basic_training.py#L138)

## Module values
- `EPOCHS` — [`L182`](../../../../../raw/code/torchax/examples/basic_training.py#L182)
- `avg_loss` — [`L191`](../../../../../raw/code/torchax/examples/basic_training.py#L191)
- `avg_vloss` — [`L208`](../../../../../raw/code/torchax/examples/basic_training.py#L208)
- `best_vloss` — [`L183`](../../../../../raw/code/torchax/examples/basic_training.py#L183)
- `classes` — [`L55`](../../../../../raw/code/torchax/examples/basic_training.py#L55)
- `dataiter` — [`L90`](../../../../../raw/code/torchax/examples/basic_training.py#L90)
- `dummy_labels` — [`L126`](../../../../../raw/code/torchax/examples/basic_training.py#L126)
- `dummy_outputs` — [`L124`](../../../../../raw/code/torchax/examples/basic_training.py#L124)
- `epoch` — [`L185`](../../../../../raw/code/torchax/examples/basic_training.py#L185)
- `epoch_number` — [`L181`](../../../../../raw/code/torchax/examples/basic_training.py#L181)
- `i` — [`L200`](../../../../../raw/code/torchax/examples/basic_training.py#L200)
- `images` — [`L91`](../../../../../raw/code/torchax/examples/basic_training.py#L91)
- `img_grid` — [`L94`](../../../../../raw/code/torchax/examples/basic_training.py#L94)
- `labels` — [`L91`](../../../../../raw/code/torchax/examples/basic_training.py#L91)
- `loss` — [`L131`](../../../../../raw/code/torchax/examples/basic_training.py#L131)
- `loss_fn` — [`L120`](../../../../../raw/code/torchax/examples/basic_training.py#L120)
- `model` — [`L118`](../../../../../raw/code/torchax/examples/basic_training.py#L118)
- `optimizer` — [`L135`](../../../../../raw/code/torchax/examples/basic_training.py#L135)
- `running_vloss` — [`L193`](../../../../../raw/code/torchax/examples/basic_training.py#L193)
- `training_loader` — [`L49`](../../../../../raw/code/torchax/examples/basic_training.py#L49)
- `training_set` — [`L41`](../../../../../raw/code/torchax/examples/basic_training.py#L41)
- `transform` — [`L36`](../../../../../raw/code/torchax/examples/basic_training.py#L36)
- `validation_loader` — [`L50`](../../../../../raw/code/torchax/examples/basic_training.py#L50)
- `validation_set` — [`L44`](../../../../../raw/code/torchax/examples/basic_training.py#L44)
- `vdata` — [`L200`](../../../../../raw/code/torchax/examples/basic_training.py#L200)
- `vinputs` — [`L201`](../../../../../raw/code/torchax/examples/basic_training.py#L201)
- `vlabels` — [`L201`](../../../../../raw/code/torchax/examples/basic_training.py#L201)
- `vloss` — [`L205`](../../../../../raw/code/torchax/examples/basic_training.py#L205)
- `voutputs` — [`L204`](../../../../../raw/code/torchax/examples/basic_training.py#L204)

