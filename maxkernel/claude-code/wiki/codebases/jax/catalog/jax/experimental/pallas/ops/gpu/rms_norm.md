---
title: 'Module: jax/experimental/pallas/ops/gpu/rms_norm.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/rms_norm.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.rms_norm`/rms_norm
symbols:
  rms_norm_backward: _backward().
  rms_norm: ().
  rms_norm_forward: _forward().
  rms_norm_backward_kernel_dw_db: _backward_kernel_dw_db().
  rms_norm_forward_kernel.body: _forward_kernel().body().
  rms_norm_backward_kernel_dx.dx_body: _backward_kernel_dx().dx_body().
  rms_norm_forward_kernel: _forward_kernel().
  rms_norm_forward_kernel.var_body: _forward_kernel().var_body().
  rms_norm_backward_kernel_dw_db.body: _backward_kernel_dw_db().body().
  rms_norm_reference: _reference().
  rms_norm_backward_kernel_dx: _backward_kernel_dx().
  rms_norm_backward_kernel_dx.mean_body: _backward_kernel_dx().mean_body().
---
# Module: [`jax/experimental/pallas/ops/gpu/rms_norm.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py)

## Functions
- `body(i)` — [`L51`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L51)
- `body(i, acc)` — [`L168`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L168)
- `dx_body(i)` — [`L135`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L135)
- `mean_body(i, c1_acc)` — [`L110`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L110)
- `rms_norm(x, weight, bias, num_warps: int | None = None, num_stages: int | None = 3, eps: float = 0.00001, backward_pass_impl: str = 'triton', interpret: bool = False)` — [`L270`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L270)
- `rms_norm_backward(num_warps: int | None, num_stages: int | None, eps: float, backward_pass_impl: str, interpret: bool, res, do)` — [`L195`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L195)
- `rms_norm_backward_kernel_dw_db(x_ref, weight_ref, bias_ref, do_ref, rstd_ref, dw_ref, db_ref, *, eps: float, block_m: int, block_n: int)` — [`L156`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L156)
- `rms_norm_backward_kernel_dx(x_ref, weight_ref, bias_ref, do_ref, rstd_ref, dx_ref, *, eps: float, block_size: int)` — [`L101`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L101)
- `rms_norm_forward(x, weight, bias, num_warps: int | None = None, num_stages: int | None = 3, eps: float = 0.00001, backward_pass_impl: str = 'triton', interpret: bool = False)` — [`L63`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L63)
- `rms_norm_forward_kernel(x_ref, weight_ref, bias_ref, o_ref, rstd_ref=None, *, eps: float, block_size: int)` — [`L27`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L27)
- `rms_norm_reference(x, weight, bias, *, eps: float = 0.00001)` — [`L305`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L305)
- `var_body(i, acc)` — [`L33`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py#L33)

