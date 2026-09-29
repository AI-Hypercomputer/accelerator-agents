---
title: 'Module: python/sgl_jax/srt/kernels/fused_mlp.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_mlp.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_mlp`/
symbols:
  inner_mlp_kernel: inner_mlp_kernel().
  apply_fused_mlp_sharded: apply_fused_mlp_sharded().
  inner_mlp_kernel._compute: inner_mlp_kernel()._compute().
  inner_mlp_kernel.compute_first_last: inner_mlp_kernel().compute_first_last().
  inner_mlp_kernel.compute_first: inner_mlp_kernel().compute_first().
  inner_mlp_kernel.compute: inner_mlp_kernel().compute().
  inner_mlp_kernel.compute_last: inner_mlp_kernel().compute_last().
  mlp_kernel_main: mlp_kernel_main().
  apply_fused_mlp_sharded.local_fused_mlp: apply_fused_mlp_sharded().local_fused_mlp().
  apply_fused_mlp_with_padding: apply_fused_mlp_with_padding().
---
# Module: [`python/sgl_jax/srt/kernels/fused_mlp.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py)

## Functions
- `_compute(is_first: bool, is_last: bool)` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L33)
- `apply_fused_mlp_sharded(x: jax.Array, w_gu: jax.Array, wd: jax.Array, mesh: jax.sharding.Mesh, b_seq: int = 64, b_inter: int = 128)` — [`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L126)
- `apply_fused_mlp_with_padding(x: jax.Array, w_gu: jax.Array, wd: jax.Array, mesh: jax.sharding.Mesh, b_seq: int = 64, b_inter: int = 128)` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L185) — Pads the input sequence length to be a multiple of b_seq if necessary.
- `compute()` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L68)
- `compute_first()` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L64)
- `compute_first_last()` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L60)
- `compute_last()` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L72)
- `inner_mlp_kernel(x_tile, w_gu_tile, wd_tile, y_tile, y_scratch, *, b_seq: int, b_inter: int, hidden_size: int, num_inter: int)` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L19)
- `local_fused_mlp(x_loc, w_gu_loc, wd_loc)` — [`L148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L148)
- `mlp_kernel_main(x_hbm, w_gu_hbm, wd_hbm, y_hbm, y_scratch, *, b_seq, b_inter, hidden_size)` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L86) — Entry point for Pallas grid. Wires up HBM references to the pipeline.

