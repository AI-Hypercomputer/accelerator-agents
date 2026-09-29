---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu`/
symbols:
  Config: Config#
  get_tpu_specific_default_config: get_tpu_specific_default_config().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._fwd: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#_fwd().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._get_autotuning_configs: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#_get_autotuning_configs().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._fwd: PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp#_fwd().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._get_autotuning_configs: PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp#_get_autotuning_configs().
  Config.b_block_size: Config#b_block_size.
  Config.h_block_size: Config#h_block_size.
  Config.v_block_size: Config#v_block_size.
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._get_heuristics_config: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#_get_heuristics_config().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._get_heuristics_config: PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp#_get_heuristics_config().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#
  PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp: PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp#
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.__post_init__: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#__post_init__().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.config_cls: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#config_cls.
  PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp.config_cls: PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp#config_cls.
  PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.supported_on: PallasMosaicTpuLinearSoftmaxCrossEntropyLoss#supported_on().
  PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp.supported_on: PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp#supported_on().
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py:33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L33)
- doc: The configuration specific for the Pallas Mosaic TPU kernel.
- signature: `class Config:`
- members:
  - `b_block_size` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L42)
  - `h_block_size` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L43)
  - `v_block_size` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L44)
- used by: [`get_tpu_specific_default_config`](pallas_mosaic_tpu.md#get_tpu_specific_default_config), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._fwd), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._fwd), [`_get_autotuning_configs`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._get_autotuning_configs), [`_get_autotuning_configs`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._get_autotuning_configs), [`test_kernel_bwd_matches_reference`](pallas_mosaic_tpu_kernel_test.md#FlashLcePallasMosaicTpuKernelTest.test_kernel_bwd_matches_reference), [`test_kernel_forward_matches_reference`](pallas_mosaic_tpu_kernel_test.md#FlashLcePallasMosaicTpuKernelTest.test_kernel_forward_matches_reference), [`test_validation_errors`](pallas_mosaic_tpu_kernel_test.md#FlashLcePallasMosaicTpuKernelTest.test_validation_errors), [`_get_heuristics_config`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss._get_heuristics_config), [`_get_heuristics_config`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp._get_heuristics_config), [`PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss), [`PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp), [`config_cls`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.config_cls), [`config_cls`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp.config_cls)

### `PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`  ·  implements/extends LinearSoftmaxCrossEntropyLoss
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py:65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L65)
- doc: Wrapper for the tokamax Op API for Pallas Mosaic TPU kernel.
- signature: `class PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(base.LinearSoftmaxCrossEntropyLoss[Config]):`
- members:
  - `supported_on(self, device: jax.Device)` — [`L143`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L143)
  - `config_cls` — [`L70`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L70) — ---
- protocol/private: `__post_init__`[`L72`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L72), `_fwd`[`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L79), `_get_autotuning_configs`[`L111`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L111), `_get_heuristics_config`[`L101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L101)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`arguments`](../op.md#BoundArguments.arguments), [`Config`](pallas_mosaic_tpu.md#Config), [`LinearSoftmaxCrossEntropyLoss`](base.md#LinearSoftmaxCrossEntropyLoss), [`config`](../op.md#Op.config), [`get_tpu_specific_default_config`](pallas_mosaic_tpu.md#get_tpu_specific_default_config), [`b_block_size`](pallas_mosaic_tpu.md#Config.b_block_size), [`h_block_size`](pallas_mosaic_tpu.md#Config.h_block_size), [`v_block_size`](pallas_mosaic_tpu.md#Config.v_block_size), [`linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu`](pallas_mosaic_tpu_kernel.md#linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu), [`Residuals`](base.md#Residuals), [`PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`LinearSoftmaxCrossEntropyLoss`](base.md#LinearSoftmaxCrossEntropyLoss), [`_fwd`](base.md#LinearSoftmaxCrossEntropyLoss._fwd), [`_DEFAULT_IMPLEMENTATION`](api.md#_DEFAULT_IMPLEMENTATION), [`test_op_wrapper_fwd_bwd`](pallas_mosaic_tpu_test.md#FlashLcePallasMosaicTpuTest.test_op_wrapper_fwd_bwd)

### `PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp`
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py:148`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L148)
- members:
  - `_fwd(self, residuals: base.Residuals, out: Real[Array, ""], dout: Real[Array, ""], x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, reduction: Literal["sum", "mean"] = "sum", config: Config, return_residuals: bool)` — [`L155`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L155) — Computes Linear Softmax Cross-Entropy Loss VJP `(dx, dlabels, dw)`.
  - `supported_on(self, device: jax.Device)` — [`L230`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L230)
  - `config_cls` — [`L153`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L153) — ---
- protocol/private: `_get_autotuning_configs`[`L198`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L198), `_get_heuristics_config`[`L188`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L188)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`arguments`](../op.md#BoundArguments.arguments), [`Config`](pallas_mosaic_tpu.md#Config), [`get_tpu_specific_default_config`](pallas_mosaic_tpu.md#get_tpu_specific_default_config), [`b_block_size`](pallas_mosaic_tpu.md#Config.b_block_size), [`h_block_size`](pallas_mosaic_tpu.md#Config.h_block_size), [`v_block_size`](pallas_mosaic_tpu.md#Config.v_block_size), [`LinearSoftmaxCrossEntropyLossVjp`](base.md#LinearSoftmaxCrossEntropyLossVjp), [`linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu`](pallas_mosaic_tpu_kernel.md#linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu), [`Residuals`](base.md#Residuals)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#LinearSoftmaxCrossEntropyLossVjp._fwd), [`__post_init__`](pallas_mosaic_tpu.md#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.__post_init__)

## Functions
- `get_tpu_specific_default_config(b_dim: int, h_dim: int, v_dim: int)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L47) — Returns the heuristic config for based on TPU version.

