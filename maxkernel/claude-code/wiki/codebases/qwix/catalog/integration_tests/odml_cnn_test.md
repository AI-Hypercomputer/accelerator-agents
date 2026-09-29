---
title: 'Module: integration_tests/odml_cnn_test.py'
type: catalog
provenance: extracted
module: integration_tests/odml_cnn_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `integration_tests.odml_cnn_test`/
symbols:
  OdmlCnnTest.test_cnn_srq: OdmlCnnTest#test_cnn_srq().
  train_and_evaluate: train_and_evaluate().
  apply_model: apply_model().
  evaluate: evaluate().
  create_train_state: create_train_state().
  CNN: CNN#
  TrainStateWithQuantStats: TrainStateWithQuantStats#
  apply_model.loss_fn: apply_model().loss_fn().
  train_epoch: train_epoch().
  TrainStateWithQuantStats.quant_stats: TrainStateWithQuantStats#quant_stats.
  get_datasets: get_datasets().
  CNN.__call__: CNN#__call__().
  OdmlCnnTest: OdmlCnnTest#
---
# Module: [`integration_tests/odml_cnn_test.py`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py)

## Classes
### `CNN`  ·  implements/extends Module
- def: [`integration_tests/odml_cnn_test.py:41`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L41)
- doc: A simple CNN model.
- signature: `class CNN(nn.Module):`
- protocol/private: `__call__`[`L45`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L45)
- used by: [`test_cnn_srq`](odml_cnn_test.md#OdmlCnnTest.test_cnn_srq), [`train_and_evaluate`](odml_cnn_test.md#train_and_evaluate), [`evaluate`](odml_cnn_test.md#evaluate), [`create_train_state`](odml_cnn_test.md#create_train_state)

### `OdmlCnnTest`  ·  implements/extends TestCase
- def: [`integration_tests/odml_cnn_test.py:168`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L168)
- signature: `class OdmlCnnTest(absltest.TestCase):`
- members:
  - `test_cnn_srq(self)` — [`L170`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L170)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../qwix/_src/qconfig.md#QuantizationRule.module_path), [`OdmlQatProvider`](../qwix/_src/providers/odml.md#OdmlQatProvider), [`OdmlConversionProvider`](../qwix/_src/providers/odml.md#OdmlConversionProvider), [`train_and_evaluate`](odml_cnn_test.md#train_and_evaluate), [`evaluate`](odml_cnn_test.md#evaluate), [`CNN`](odml_cnn_test.md#CNN), [`get_datasets`](odml_cnn_test.md#get_datasets), [`quant_stats`](odml_cnn_test.md#TrainStateWithQuantStats.quant_stats)

### `TrainStateWithQuantStats`  ·  implements/extends TrainState
- def: [`integration_tests/odml_cnn_test.py:37`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L37)
- signature: `class TrainStateWithQuantStats(train_state.TrainState):`
- members:
  - `quant_stats` — [`L38`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L38)
- used by: [`test_cnn_srq`](odml_cnn_test.md#OdmlCnnTest.test_cnn_srq), [`train_and_evaluate`](odml_cnn_test.md#train_and_evaluate), [`apply_model`](odml_cnn_test.md#apply_model), [`create_train_state`](odml_cnn_test.md#create_train_state), [`loss_fn`](odml_cnn_test.md#apply_model.loss_fn)

## Functions
- `apply_model(state: TrainStateWithQuantStats, images, labels)` — [`L60`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L60) — Computes gradients, loss and accuracy for a single batch.
- `create_train_state(cnn: CNN, rng, config)` — [`L115`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L115) — Creates initial `TrainState`.
- `evaluate(cnn: CNN, variables: Mapping[str, Any])` — [`L159`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L159) — Only evaluate the model.
- `get_datasets()` — [`L104`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L104) — Load MNIST train and test datasets into memory.
- `loss_fn(params)` — [`L63`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L63)
- `train_and_evaluate(cnn: CNN, qat_cnn: CNN, config: ml_collections.ConfigDict)` — [`L122`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L122) — Execute model training and evaluation loop.
- `train_epoch(state, train_ds, batch_size, rng)` — [`L80`](../../../../../raw/code/qwix/integration_tests/odml_cnn_test.py#L80) — Train for a single epoch.

