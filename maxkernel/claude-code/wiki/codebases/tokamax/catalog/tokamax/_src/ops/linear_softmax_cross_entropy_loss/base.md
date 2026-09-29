---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.base`/
symbols:
  LinearSoftmaxCrossEntropyLoss: LinearSoftmaxCrossEntropyLoss#
  LinearSoftmaxCrossEntropyLoss._fwd: LinearSoftmaxCrossEntropyLoss#_fwd().
  LinearSoftmaxCrossEntropyLossVjp._fwd: LinearSoftmaxCrossEntropyLossVjp#_fwd().
  LinearSoftmaxCrossEntropyLossVjp: LinearSoftmaxCrossEntropyLossVjp#
  Residuals: Residuals.
  LinearSoftmaxCrossEntropyLoss.__post_init__: LinearSoftmaxCrossEntropyLoss#__post_init__().
  _Config: _Config.
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py)

## Classes
### `LinearSoftmaxCrossEntropyLoss`  ·  implements/extends Op
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py:31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L31)
- doc: Linear Softmax Cross-Entropy Loss tokamax Op API using reference impl.
- signature: `class LinearSoftmaxCrossEntropyLoss(op.Op[Any, jax.Array, Residuals, _Config, None]):`
- protocol/private: `__post_init__`[`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L36), `_fwd`[`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L41)
- uses (calls/refs, reference-scoped): [`Op`](../op.md#Op), [`config`](../op.md#Op.config), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._fwd), [`_fwd`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss._fwd), [`LinearSoftmaxCrossEntropyLossVjp`](base.md#LinearSoftmaxCrossEntropyLossVjp), [`PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss), [`Residuals`](base.md#Residuals), [`ChunkedXlaLinearSoftmaxCrossEntropyLoss`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss), [`linear_softmax_cross_entropy_loss_fwd_reference`](reference.md#linear_softmax_cross_entropy_loss_fwd_reference), [`_Config`](base.md#_Config)
- used by: [`Op`](../op.md#Op), [`_fwd`](../op.md#Op._fwd), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS), [`test_reference_running_correctly`](base_test.md#LinearSoftmaxCrossEntropyLossBaseTest.test_reference_running_correctly), [`PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss), [`test_op_wrapper_fwd_bwd`](pallas_mosaic_tpu_test.md#FlashLcePallasMosaicTpuTest.test_op_wrapper_fwd_bwd), [`ChunkedXlaLinearSoftmaxCrossEntropyLoss`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss)

### `LinearSoftmaxCrossEntropyLossVjp`
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py:59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L59)
- members:
  - `_fwd(self, residuals: Residuals, out: jax.Array, dout: Real[Array, ""], x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, reduction: Literal["sum", "mean"] = "sum", config: _Config, return_residuals: bool)` — [`L62`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L62) — Computes Linear Softmax Cross-Entropy Loss VJP `(dx, dlabels, dw)`.
- uses (calls/refs, reference-scoped): [`Op`](../op.md#Op), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._fwd), [`_fwd`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp._fwd), [`Residuals`](base.md#Residuals), [`linear_softmax_cross_entropy_loss_bwd_reference`](reference.md#linear_softmax_cross_entropy_loss_bwd_reference), [`_Config`](base.md#_Config)
- used by: [`_fwd`](../op.md#Op._fwd), [`ChunkedXlaLinearSoftmaxCrossEntropyLossVjp`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp), [`PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp), [`__post_init__`](base.md#LinearSoftmaxCrossEntropyLoss.__post_init__)

## Module values
- `Residuals` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L27)
- `_Config` — [`L26`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py#L26)

