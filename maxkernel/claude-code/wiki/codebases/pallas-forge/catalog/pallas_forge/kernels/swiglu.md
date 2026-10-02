---
title: 'Module: pallas_forge/kernels/swiglu.py'
type: catalog
provenance: extracted
module: pallas_forge/kernels/swiglu.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.kernels.swiglu`/
symbols:
  fused_swiglu: fused_swiglu().
  _fused_glu: _fused_glu().
  fused_geglu: fused_geglu().
  _swiglu_kernel: _swiglu_kernel().
---
# Module: [`pallas_forge/kernels/swiglu.py`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py)

## Functions
- `_fused_glu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array, *, block_m: int = 128, block_n: int = 128, use_gelu: bool = False, num_stages: int = 2)` — [`L56`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py#L56) — Internal implementation for fused gated linear units.
- `_swiglu_kernel(x_ref, w_gate_ref, w_up_ref, out_ref, *, use_gelu: bool)` — [`L25`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py#L25) — Fused gated activation kernel.
- `fused_geglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array, *, block_m: int = 128, block_n: int = 128, num_stages: int = 2)` — [`L157`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py#L157) — Fused GeGLU activation: ``gelu(x @ w_gate) * (x @ w_up)``.
- `fused_swiglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array, *, block_m: int = 128, block_n: int = 128, num_stages: int = 2)` — [`L124`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py#L124) — Fused SwiGLU activation: ``silu(x @ w_gate) * (x @ w_up)``.

