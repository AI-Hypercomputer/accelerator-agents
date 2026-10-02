---
title: 'Module: src/maxtext/kernels/attention/ragged_attention.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/attention/ragged_attention.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.attention.ragged_attention`/
symbols:
  ragged_mqa: ragged_mqa().
  ragged_mha: ragged_mha().
  ragged_gqa: ragged_gqa().
  reference_mha: reference_mha().
  get_mha_cost_estimate: get_mha_cost_estimate().
  ragged_mqa.compute_ragged_block_indices: ragged_mqa().compute_ragged_block_indices().
  reference_mqa: reference_mqa().
  ragged_flash_attention_kernel: ragged_flash_attention_kernel().
  reference_gqa: reference_gqa().
  ragged_flash_attention_kernel.init: ragged_flash_attention_kernel().init().
  ragged_flash_attention_kernel.run: ragged_flash_attention_kernel().run().
---
# Module: [`src/maxtext/kernels/attention/ragged_attention.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py)

## Functions
- `compute_ragged_block_indices(b, i, lengths_ref)` — [`L245`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L245)
- `get_mha_cost_estimate(shape_dtype)` — [`L30`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L30) — Get cost estimate for MHA based on static shape information.
- `init()` — [`L178`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L178)
- `ragged_flash_attention_kernel(lengths_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, *, block_size: int, mask_value: float)` — [`L162`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L162) — Pallas kernel for flash attention.
- `ragged_gqa(query: jax.Array, key: jax.Array, value: jax.Array, lengths: jax.Array, *, block_size: int = 256, mask_value: float = DEFAULT_MASK_VALUE, interpret: bool = False)` — [`L355`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L355) — Ragged group query attention.
- `ragged_mha(query: jax.Array, key: jax.Array, value: jax.Array, lengths: jax.Array, *, block_size: int = 256, mask_value: float = DEFAULT_MASK_VALUE, interpret: bool = False)` — [`L294`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L294) — Ragged multi head attention.
- `ragged_mqa(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, *, block_size: int = 256, mask_value: float = DEFAULT_MASK_VALUE, cost_estimate: pl.CostEstimate | None = None, interpret: bool = False)` — [`L213`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L213) — Ragged multi query attention.
- `reference_gqa(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, mask_value: float = DEFAULT_MASK_VALUE)` — [`L120`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L120) — Vanilla attention GQA implementation for reference.
- `reference_mha(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, *, mask_value: float = DEFAULT_MASK_VALUE)` — [`L88`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L88) — Multi head attention reference.
- `reference_mqa(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, *, mask_value: float = DEFAULT_MASK_VALUE)` — [`L52`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L52) — Multi query attention reference.
- `run()` — [`L186`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L186)

