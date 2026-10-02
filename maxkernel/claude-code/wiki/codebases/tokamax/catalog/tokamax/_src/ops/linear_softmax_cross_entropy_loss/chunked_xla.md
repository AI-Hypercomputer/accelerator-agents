---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla`/
symbols:
  ChunkedXlaLinearSoftmaxCrossEntropyLoss._fwd: ChunkedXlaLinearSoftmaxCrossEntropyLoss#_fwd().
  ChunkedXlaLinearSoftmaxCrossEntropyLossVjp._fwd: ChunkedXlaLinearSoftmaxCrossEntropyLossVjp#_fwd().
  Config: Config#
  ChunkedXlaLinearSoftmaxCrossEntropyLoss: ChunkedXlaLinearSoftmaxCrossEntropyLoss#
  ChunkedXlaLinearSoftmaxCrossEntropyLossVjp: ChunkedXlaLinearSoftmaxCrossEntropyLossVjp#
  ChunkedXlaLinearSoftmaxCrossEntropyLoss.__post_init__: ChunkedXlaLinearSoftmaxCrossEntropyLoss#__post_init__().
  ChunkedXlaLinearSoftmaxCrossEntropyLoss._get_heuristics_config: ChunkedXlaLinearSoftmaxCrossEntropyLoss#_get_heuristics_config().
  ChunkedXlaLinearSoftmaxCrossEntropyLossVjp._get_heuristics_config: ChunkedXlaLinearSoftmaxCrossEntropyLossVjp#_get_heuristics_config().
  linear_softmax_cross_entropy_loss_fwd_chunked_xla: linear_softmax_cross_entropy_loss_fwd_chunked_xla().
  linear_softmax_cross_entropy_loss_bwd_chunked_xla: linear_softmax_cross_entropy_loss_bwd_chunked_xla().
  linear_softmax_cross_entropy_loss_fwd_chunked_xla.b_loop_body: linear_softmax_cross_entropy_loss_fwd_chunked_xla().b_loop_body().
  linear_softmax_cross_entropy_loss_bwd_chunked_xla.b_loop_body: linear_softmax_cross_entropy_loss_bwd_chunked_xla().b_loop_body().
  ChunkedXlaLinearSoftmaxCrossEntropyLoss.config_cls: ChunkedXlaLinearSoftmaxCrossEntropyLoss#config_cls.
  ChunkedXlaLinearSoftmaxCrossEntropyLossVjp.config_cls: ChunkedXlaLinearSoftmaxCrossEntropyLossVjp#config_cls.
  Config.b_block_size: Config#b_block_size.
  Config.v_block_size: Config#v_block_size.
  linear_softmax_cross_entropy_loss_fwd_chunked_xla.b_loop_body.v_loop_body: linear_softmax_cross_entropy_loss_fwd_chunked_xla().b_loop_body().v_loop_body().
  linear_softmax_cross_entropy_loss_bwd_chunked_xla.b_loop_body.v_loop_body: linear_softmax_cross_entropy_loss_bwd_chunked_xla().b_loop_body().v_loop_body().
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py)

## Classes
### `ChunkedXlaLinearSoftmaxCrossEntropyLoss`  ·  implements/extends LinearSoftmaxCrossEntropyLoss
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py:281`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L281)
- doc: Linear Softmax Cross-Entropy Loss Op API using chunked XLA backward.
- signature: `class ChunkedXlaLinearSoftmaxCrossEntropyLoss(base.LinearSoftmaxCrossEntropyLoss[Config]):`
- members:
  - `config_cls` — [`L286`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L286) — ---
- protocol/private: `__post_init__`[`L288`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L288), `_fwd`[`L295`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L295), `_get_heuristics_config`[`L315`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L315)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`LinearSoftmaxCrossEntropyLoss`](base.md#LinearSoftmaxCrossEntropyLoss), [`config`](../op.md#Op.config), [`Config`](chunked_xla.md#Config), [`Residuals`](base.md#Residuals), [`ChunkedXlaLinearSoftmaxCrossEntropyLossVjp`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp), [`linear_softmax_cross_entropy_loss_fwd_chunked_xla`](chunked_xla.md#linear_softmax_cross_entropy_loss_fwd_chunked_xla), [`b_block_size`](chunked_xla.md#Config.b_block_size), [`v_block_size`](chunked_xla.md#Config.v_block_size)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`LinearSoftmaxCrossEntropyLoss`](base.md#LinearSoftmaxCrossEntropyLoss), [`_fwd`](base.md#LinearSoftmaxCrossEntropyLoss._fwd), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS)

### `ChunkedXlaLinearSoftmaxCrossEntropyLossVjp`
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py:320`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L320)
- members:
  - `_fwd(self, residuals: base.Residuals, out: jax.Array, dout: Real[Array, ""], x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, reduction: Literal["sum", "mean"] = "sum", config: Config, return_residuals: bool)` — [`L327`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L327) — Computes Linear Softmax Cross-Entropy Loss chunked XLA VJP `(dx, dlabels, dw)`.
  - `config_cls` — [`L325`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L325) — ---
- protocol/private: `_get_heuristics_config`[`L359`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L359)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Config`](chunked_xla.md#Config), [`LinearSoftmaxCrossEntropyLossVjp`](base.md#LinearSoftmaxCrossEntropyLossVjp), [`Residuals`](base.md#Residuals), [`linear_softmax_cross_entropy_loss_bwd_chunked_xla`](chunked_xla.md#linear_softmax_cross_entropy_loss_bwd_chunked_xla), [`b_block_size`](chunked_xla.md#Config.b_block_size), [`v_block_size`](chunked_xla.md#Config.v_block_size)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_fwd`](base.md#LinearSoftmaxCrossEntropyLossVjp._fwd), [`__post_init__`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss.__post_init__)

### `Config`
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py:33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L33)
- doc: The configuration specific for the Chunked XLA kernel.
- signature: `class Config:`
- members:
  - `b_block_size` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L41)
  - `v_block_size` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L42)
- used by: [`_fwd`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss._fwd), [`_fwd`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp._fwd), [`ChunkedXlaLinearSoftmaxCrossEntropyLoss`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss), [`ChunkedXlaLinearSoftmaxCrossEntropyLossVjp`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp), [`_get_heuristics_config`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss._get_heuristics_config), [`_get_heuristics_config`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp._get_heuristics_config), [`config_cls`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLoss.config_cls), [`config_cls`](chunked_xla.md#ChunkedXlaLinearSoftmaxCrossEntropyLossVjp.config_cls)

## Functions
- `b_loop_body(i, carry)` — [`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L104)
- `b_loop_body(i, b_carry)` — [`L228`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L228)
- `linear_softmax_cross_entropy_loss_bwd_chunked_xla(dout: Real[Array, ""], lse: Real[Array, B], x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, b_block_sz: int = 1024, v_block_sz: int = 2048, reduction: Literal["sum", "mean"] = "sum", preferred_element_type: jax.typing.DTypeLike | None = None)` — [`L176`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L176) — The chunked XLA implementation of linear softmax cross-entropy loss backward pass.
- `linear_softmax_cross_entropy_loss_fwd_chunked_xla(x: Real[Array, "B H"], labels: Integer[Array, B], w: Real[Array, "H V"], *, reduction: Literal["sum", "mean"] = "sum", b_block_sz: int = 1024, v_block_sz: int = 1024, unroll_factor: int | bool = False, preferred_element_type: jax.typing.DTypeLike | None = None)` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L55) — Chunked XLA implementation of linear softmax cross-entropy loss forward.
- `v_loop_body(j, v_carry)` — [`L108`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L108)
- `v_loop_body(j, v_carry)` — [`L236`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py#L236)

