---
title: 'Module: integration_tests/vae_qat_test.py'
type: catalog
provenance: extracted
module: integration_tests/vae_qat_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `integration_tests.vae_qat_test`/
symbols:
  VaeQatTest.test_srq: VaeQatTest#test_srq().
  VaeQatTest.test_drq: VaeQatTest#test_drq().
  train_and_evaluate: train_and_evaluate().
  evaluate: evaluate().
  Encoder.__call__: Encoder#__call__().
  VAE.__call__: VAE#__call__().
  VAE: VAE#
  loss_fn: loss_fn().
  eval_step: eval_step().
  train_step: train_step().
  Encoder.kl_loss: Encoder#kl_loss.
  Decoder.__call__: Decoder#__call__().
  VAE.generate: VAE#generate().
  VAE.decoder: VAE#decoder.
  VAE.encoder: VAE#encoder.
  Loss: Loss#
  VAE.output_shape: VAE#output_shape.
  get_datasets: get_datasets().
  Encoder: Encoder#
  Encoder.linear1: Encoder#linear1.
  Encoder.linear_mean: Encoder#linear_mean.
  Encoder.linear_std: Encoder#linear_std.
  Encoder.rngs: Encoder#rngs.
  Decoder: Decoder#
  Decoder.linear1: Decoder#linear1.
  Decoder.linear2: Decoder#linear2.
  Encoder.__init__: Encoder#__init__().
  Decoder.__init__: Decoder#__init__().
  VAE.__init__: VAE#__init__().
  VaeQatTest: VaeQatTest#
---
# Module: [`integration_tests/vae_qat_test.py`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py)

## Classes
### `Decoder`  ·  implements/extends Module
- def: [`integration_tests/vae_qat_test.py:63`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L63)
- signature: `class Decoder(nnx.Module):`
- members:
  - `linear1` — [`L66`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L66)
  - `linear2` — [`L67`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L67)
- protocol/private: `__call__`[`L69`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L69), `__init__`[`L65`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L65)
- used by: [`decoder`](vae_qat_test.md#VAE.decoder)

### `Encoder`  ·  implements/extends Module
- def: [`integration_tests/vae_qat_test.py:37`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L37)
- signature: `class Encoder(nnx.Module):`
- members:
  - `kl_loss` — [`L53`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L53)
  - `linear1` — [`L40`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L40)
  - `linear_mean` — [`L41`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L41)
  - `linear_std` — [`L42`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L42)
  - `rngs` — [`L43`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L43)
- protocol/private: `__call__`[`L45`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L45), `__init__`[`L39`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L39)
- uses (calls/refs, reference-scoped): [`Loss`](vae_qat_test.md#Loss)
- used by: [`encoder`](vae_qat_test.md#VAE.encoder)

### `Loss`  ·  implements/extends Variable
- def: [`integration_tests/vae_qat_test.py:33`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L33)
- signature: `class Loss(nnx.Variable):`
- used by: [`loss_fn`](vae_qat_test.md#loss_fn), [`kl_loss`](vae_qat_test.md#Encoder.kl_loss)

### `VAE`  ·  implements/extends Module
- def: [`integration_tests/vae_qat_test.py:76`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L76)
- signature: `class VAE(nnx.Module):`
- members:
  - `generate(self, z)` — [`L99`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L99)
  - `decoder` — [`L89`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L89)
  - `encoder` — [`L88`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L88)
  - `output_shape` — [`L87`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L87)
- protocol/private: `__call__`[`L93`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L93), `__init__`[`L78`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L78)
- uses (calls/refs, reference-scoped): [`Decoder`](vae_qat_test.md#Decoder), [`Encoder`](vae_qat_test.md#Encoder)
- used by: [`test_srq`](vae_qat_test.md#VaeQatTest.test_srq), [`test_drq`](vae_qat_test.md#VaeQatTest.test_drq), [`train_and_evaluate`](vae_qat_test.md#train_and_evaluate), [`evaluate`](vae_qat_test.md#evaluate), [`eval_step`](vae_qat_test.md#eval_step), [`loss_fn`](vae_qat_test.md#loss_fn), [`train_step`](vae_qat_test.md#train_step)

### `VaeQatTest`  ·  implements/extends TestCase
- def: [`integration_tests/vae_qat_test.py:196`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L196)
- signature: `class VaeQatTest(absltest.TestCase):`
- members:
  - `test_drq(self)` — [`L198`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L198)
  - `test_srq(self)` — [`L254`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L254)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../qwix/_src/providers/ptq.md#PtqProvider), [`act_static_scale`](../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`QtProvider`](../qwix/_src/providers/qt.md#QtProvider), [`QuantStat`](../qwix/_src/utils/flax_util.md#QuantStat), [`train_and_evaluate`](vae_qat_test.md#train_and_evaluate), [`evaluate`](vae_qat_test.md#evaluate), [`VAE`](vae_qat_test.md#VAE)

## Functions
- `eval_step(model: VAE, x: jax.Array)` — [`L123`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L123)
- `evaluate(model: VAE)` — [`L184`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L184)
- `get_datasets()` — [`L129`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L129) — Load MNIST train and test datasets into memory.
- `loss_fn(model: VAE, x: jax.Array)` — [`L105`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L105)
- `train_and_evaluate(model: VAE, *, epochs: int, batch_size: int, rng: jax.Array)` — [`L140`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L140)
- `train_step(model: VAE, optimizer: nnx.ModelAndOptimizer, x: jax.Array)` — [`L115`](../../../../../raw/code/qwix/integration_tests/vae_qat_test.py#L115)

