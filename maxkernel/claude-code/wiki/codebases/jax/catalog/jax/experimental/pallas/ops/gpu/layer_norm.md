---
title: 'Module: jax/experimental/pallas/ops/gpu/layer_norm.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/layer_norm.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.layer_norm`/layer_norm
symbols:
  layer_norm_backward: _backward().
  layer_norm: ().
  layer_norm_forward_kernel: _forward_kernel().
  layer_norm_forward: _forward().
  layer_norm_backward_kernel_dw_db: _backward_kernel_dw_db().
  layer_norm_forward_kernel.body: _forward_kernel().body().
  layer_norm_backward_kernel_dx.dx_body: _backward_kernel_dx().dx_body().
  layer_norm_reference: _reference().
  layer_norm_forward_kernel.var_body: _forward_kernel().var_body().
  layer_norm_backward_kernel_dw_db.body: _backward_kernel_dw_db().body().
  layer_norm_forward_kernel.mean_body: _forward_kernel().mean_body().
  layer_norm_backward_kernel_dx: _backward_kernel_dx().
  layer_norm_backward_kernel_dx.mean_body: _backward_kernel_dx().mean_body().
---
# Module: [`jax/experimental/pallas/ops/gpu/layer_norm.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py)

## Functions
- `body(i)` — [`L73`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L73)
- `body(i, acc)` — [`L195`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L195)
- `dx_body(i)` — [`L162`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L162)
- `layer_norm(x, weight, bias, num_warps: int | None = None, num_stages: int | None = 3, eps: float = 0.00001, backward_pass_impl: str = 'triton', interpret: bool = False)` — [`L303`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L303)
- `layer_norm_backward(num_warps: int | None, num_stages: int | None, eps: float, backward_pass_impl: str, interpret: bool, res, do)` — [`L227`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L227)
- `layer_norm_backward_kernel_dw_db(x_ref, weight_ref, bias_ref, do_ref, mean_ref, rstd_ref, dw_ref, db_ref, *, eps: float, block_m: int, block_n: int)` — [`L183`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L183)
- `layer_norm_backward_kernel_dx(x_ref, weight_ref, bias_ref, do_ref, mean_ref, rstd_ref, dx_ref, *, eps: float, block_size: int)` — [`L124`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L124)
- `layer_norm_forward(x, weight, bias, num_warps: int | None = None, num_stages: int | None = 3, eps: float = 0.00001, backward_pass_impl: str = 'triton', interpret: bool = False)` — [`L85`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L85)
- `layer_norm_forward_kernel(x_ref, weight_ref, bias_ref, o_ref, mean_ref=None, rstd_ref=None, *, eps: float, block_size: int)` — [`L28`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L28)
- `layer_norm_reference(x, weight, bias, *, eps: float = 0.00001)` — [`L337`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L337)
- `mean_body(i, acc)` — [`L34`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L34)
- `mean_body(i, acc)` — [`L133`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L133)
- `var_body(i, acc)` — [`L50`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py#L50)

