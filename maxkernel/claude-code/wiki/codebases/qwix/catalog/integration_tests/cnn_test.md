---
title: 'Module: integration_tests/cnn_test.py'
type: catalog
provenance: extracted
module: integration_tests/cnn_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `integration_tests.cnn_test`/
symbols:
  CnnTest.test_srq: CnnTest#test_srq().
  CnnTest.test_drq: CnnTest#test_drq().
  train_and_evaluate: train_and_evaluate().
  evaluate: evaluate().
  apply_model: apply_model().
  create_train_state: create_train_state().
  CNN: CNN#
  TrainStateWithQuantStats: TrainStateWithQuantStats#
  apply_model.loss_fn: apply_model().loss_fn().
  train_epoch: train_epoch().
  TrainStateWithQuantStats.quant_stats: TrainStateWithQuantStats#quant_stats.
  get_datasets: get_datasets().
  CNN.__call__: CNN#__call__().
  CnnTest: CnnTest#
---
# Module: [`integration_tests/cnn_test.py`](../../../../../raw/code/qwix/integration_tests/cnn_test.py)

## Classes
### `CNN`  ·  implements/extends Module
- def: [`integration_tests/cnn_test.py:44`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L44)
- doc: A simple CNN model.
- signature: `class CNN(nn.Module):`
- protocol/private: `__call__`[`L48`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L48)
- used by: [`test_srq`](cnn_test.md#CnnTest.test_srq), [`test_drq`](cnn_test.md#CnnTest.test_drq), [`train_and_evaluate`](cnn_test.md#train_and_evaluate), [`evaluate`](cnn_test.md#evaluate), [`create_train_state`](cnn_test.md#create_train_state)

### `CnnTest`  ·  implements/extends TestCase
- def: [`integration_tests/cnn_test.py:167`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L167)
- signature: `class CnnTest(parameterized.TestCase):`
- members:
  - `test_drq(self)` — [`L169`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L169)
  - `test_srq(self)` — [`L214`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L214)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_params`](../qwix/_src/providers/ptq.md#quantize_params), [`act_static_scale`](../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`QtRule`](../qwix/_src/providers/qt.md#QtRule), [`QtProvider`](../qwix/_src/providers/qt.md#QtProvider), [`bwd_qtype`](../qwix/_src/providers/qt.md#QtRule.bwd_qtype), [`train_and_evaluate`](cnn_test.md#train_and_evaluate), [`evaluate`](cnn_test.md#evaluate), [`CNN`](cnn_test.md#CNN), [`quant_stats`](cnn_test.md#TrainStateWithQuantStats.quant_stats)

### `TrainStateWithQuantStats`  ·  implements/extends TrainState
- def: [`integration_tests/cnn_test.py:40`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L40)
- signature: `class TrainStateWithQuantStats(train_state.TrainState):`
- members:
  - `quant_stats` — [`L41`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L41)
- used by: [`test_srq`](cnn_test.md#CnnTest.test_srq), [`train_and_evaluate`](cnn_test.md#train_and_evaluate), [`apply_model`](cnn_test.md#apply_model), [`create_train_state`](cnn_test.md#create_train_state), [`loss_fn`](cnn_test.md#apply_model.loss_fn)

## Functions
- `apply_model(state: TrainStateWithQuantStats, images, labels)` — [`L63`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L63) — Computes gradients, loss and accuracy for a single batch.
- `create_train_state(cnn: CNN, rng, config)` — [`L118`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L118) — Creates initial `TrainState`.
- `evaluate(cnn: CNN, variables: Mapping[str, Any])` — [`L158`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L158) — Only evaluate the model.
- `get_datasets()` — [`L107`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L107) — Load MNIST train and test datasets into memory.
- `loss_fn(params)` — [`L66`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L66)
- `train_and_evaluate(cnn: CNN, config: ml_collections.ConfigDict)` — [`L125`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L125) — Execute model training and evaluation loop.
- `train_epoch(state, train_ds, batch_size, rng)` — [`L83`](../../../../../raw/code/qwix/integration_tests/cnn_test.py#L83) — Train for a single epoch.

