---
title: 'Module: src/maxdiffusion/kernels/custom_splash_attention.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/kernels/custom_splash_attention.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.kernels.custom_splash_attention`/
symbols:
  _BlockSizes: _BlockSizes#
  _splash_attention_forward_ring: _splash_attention_forward_ring().
  _splash_attention_forward: _splash_attention_forward().
  _splash_attention_forward_mhpt: _splash_attention_forward_mhpt().
  NUM_SUBLANES: NUM_SUBLANES.
  tpu_custom_attention: tpu_custom_attention().
  make_custom_splash_sdpa: make_custom_splash_sdpa().
  make_splash_mha: make_splash_mha().
  tpu_custom_attention._attention_on_slices._kernel_3d: tpu_custom_attention()._attention_on_slices()._kernel_3d().
  _flash_attention_kernel.compute_body: _flash_attention_kernel().compute_body().
  _flash_attention_kernel_mhpt.compute_body: _flash_attention_kernel_mhpt().compute_body().
  make_splash_mha._splash_attention: make_splash_mha()._splash_attention().
  make_custom_splash_sdpa._sdpa: make_custom_splash_sdpa()._sdpa().
  _flash_attention_kernel.last_body: _flash_attention_kernel().last_body().
  _flash_attention_kernel_mhpt.last_body: _flash_attention_kernel_mhpt().last_body().
  _flash_attention_kernel: _flash_attention_kernel().
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  _flash_attention_kernel.last_compute_body: _flash_attention_kernel().last_compute_body().
  _flash_attention_kernel_mhpt: _flash_attention_kernel_mhpt().
  _flash_attention_kernel_mhpt.last_compute_body: _flash_attention_kernel_mhpt().last_compute_body().
  tpu_custom_attention._attention_on_slices: tpu_custom_attention()._attention_on_slices().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  DEFAULT_BKVCOMPUTEINSIZE: DEFAULT_BKVCOMPUTEINSIZE.
  _splash_attention_forward_ring.out_index_map: _splash_attention_forward_ring().out_index_map().
  tpu_custom_attention._attention_on_slices._pad_to_multiple: tpu_custom_attention()._attention_on_slices()._pad_to_multiple().
  _BlockSizes.__init__: _BlockSizes#__init__().
  _flash_attention_kernel.body: _flash_attention_kernel().body().
  _flash_attention_kernel_mhpt.body: _flash_attention_kernel_mhpt().body().
  DEFAULT_BQSIZE: DEFAULT_BQSIZE.
  DEFAULT_BKVSIZE: DEFAULT_BKVSIZE.
  DEFAULT_BKVCOMPUTESIZE: DEFAULT_BKVCOMPUTESIZE.
  _splash_attention_forward.q_index_map: _splash_attention_forward().q_index_map().
  _splash_attention_forward.out_index_map: _splash_attention_forward().out_index_map().
  _splash_attention_forward.k_index_map: _splash_attention_forward().k_index_map().
  _splash_attention_forward.v_index_map: _splash_attention_forward().v_index_map().
  _splash_attention_forward_ring.q_index_map: _splash_attention_forward_ring().q_index_map().
  _splash_attention_forward_ring.k_index_map: _splash_attention_forward_ring().k_index_map().
  _splash_attention_forward_ring.v_index_map: _splash_attention_forward_ring().v_index_map().
  _splash_attention_forward_mhpt.q_index_map: _splash_attention_forward_mhpt().q_index_map().
  _splash_attention_forward_mhpt.k_index_map: _splash_attention_forward_mhpt().k_index_map().
  _splash_attention_forward_mhpt.v_index_map: _splash_attention_forward_mhpt().v_index_map().
  _splash_attention_forward_mhpt.out_index_map: _splash_attention_forward_mhpt().out_index_map().
  make_custom_splash_sdpa._simple_attention: make_custom_splash_sdpa()._simple_attention().
  NUM_LANES: NUM_LANES.
  _BlockSizes.__slots__: _BlockSizes#__slots__.
  _flash_attention_kernel.init: _flash_attention_kernel().init().
  _flash_attention_kernel.end: _flash_attention_kernel().end().
  _flash_attention_kernel_mhpt.init: _flash_attention_kernel_mhpt().init().
  _flash_attention_kernel_mhpt.end: _flash_attention_kernel_mhpt().end().
---
# Module: [`src/maxdiffusion/kernels/custom_splash_attention.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py)

## Classes
### `_BlockSizes`
- def: [`src/maxdiffusion/kernels/custom_splash_attention.py:45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L45)
- signature: `class _BlockSizes:`
- protocol/private: `__init__`[`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L48), `__slots__`[`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L46)
- used by: [`_splash_attention_forward_ring`](custom_splash_attention.md#_splash_attention_forward_ring), [`_splash_attention_forward`](custom_splash_attention.md#_splash_attention_forward), [`_splash_attention_forward_mhpt`](custom_splash_attention.md#_splash_attention_forward_mhpt), [`wrap_flash_attention`](../models/attention_flax.md#_tpu_flash_attention.wrap_flash_attention), [`wrap_ulysses_attention`](../models/attention_flax.md#_ulysses_attention.wrap_ulysses_attention), [`_custom_bidirectional_ring_forward`](splash_attention/ring_attention_kernel.md#_custom_bidirectional_ring_forward), [`_custom_ring_attention_forward`](splash_attention/ring_attention_kernel.md#_custom_ring_attention_forward), [`wrap_ulysses_ring_attention`](../models/attention_flax.md#_ulysses_ring_custom_attention.wrap_ulysses_ring_attention), [`make_splash_mha`](custom_splash_attention.md#make_splash_mha), [`_kernel_3d`](custom_splash_attention.md#tpu_custom_attention._attention_on_slices._kernel_3d), [`make_custom_ring_attention`](splash_attention/ring_attention_kernel.md#make_custom_ring_attention)

## Functions
- `_attention_on_slices(q, k, v)` — [`L734`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L734)
- `_flash_attention_kernel(q_ref, k_ref, v_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, l_ring_ref=None, m_ring_ref=None, *, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, bkv_compute_in: int, head_dim_v: int, q_seq_len: int, kv_seq_len: int, use_base2_exp: bool = True, fuse_reciprocal: bool = True)` — [`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L54)
- `_flash_attention_kernel_mhpt(q_ref, k_ref, v_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, *, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, bkv_compute_in: int, head_dim_v: int, q_seq_len: int, kv_seq_len: int, heads_per_tile: int, use_base2_exp: bool = True)` — [`L212`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L212)
- `_kernel_3d(q_3d, k_3d, v_3d)` — [`L750`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L750)
- `_pad_to_multiple(x, multiple, axis)` — [`L741`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L741)
- `_sdpa(query, key, value, attn_mask=None, dropout_p=0, is_causal=False, scale=None, enable_gqa=False)` — [`L850`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L850)
- `_simple_attention(q, k, v, scale=None)` — [`L844`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L844)
- `_splash_attention(q, k, v)` — [`L664`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L664)
- `_splash_attention_forward(q: jax.Array, k: jax.Array, v: jax.Array, block_sizes: _BlockSizes, bkv_compute_in: int, q_seq_len: int | None = None, kv_seq_len: int | None = None, use_base2_exp: bool = True, use_experimental_scheduler: bool = False, vmem_limit_bytes: int | None = None)` — [`L365`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L365)
- `_splash_attention_forward_mhpt(q: jax.Array, k: jax.Array, v: jax.Array, block_sizes: _BlockSizes, bkv_compute_in: int, heads_per_tile: int, q_seq_len: int | None = None, kv_seq_len: int | None = None, use_base2_exp: bool = True, use_experimental_scheduler: bool = False, vmem_limit_bytes: int | None = None)` — [`L563`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L563)
- `_splash_attention_forward_ring(q: jax.Array, k: jax.Array, v: jax.Array, block_sizes: _BlockSizes, bkv_compute_in: int, q_seq_len: int | None = None, kv_seq_len: int | None = None, use_base2_exp: bool = True, use_experimental_scheduler: bool = False, vmem_limit_bytes: int | None = None)` — [`L453`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L453) — Ring-specific forward path that returns pre-reciprocal fp32 accumulators.
- `body()` — [`L178`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L178)
- `body()` — [`L340`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L340)
- `compute_body(kv_compute_index, _)` — [`L91`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L91)
- `compute_body(kv_compute_index, _)` — [`L247`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L247)
- `end()` — [`L196`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L196)
- `end()` — [`L358`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L358)
- `init()` — [`L86`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L86)
- `init()` — [`L242`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L242)
- `k_index_map(h, i, j, *_)` — [`L394`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L394)
- `k_index_map(h, i, j, *_)` — [`L496`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L496)
- `k_index_map(h, i, j, *_)` — [`L591`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L591)
- `last_body()` — [`L182`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L182)
- `last_body()` — [`L344`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L344)
- `last_compute_body(kv_compute_index)` — [`L133`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L133)
- `last_compute_body(kv_compute_index)` — [`L292`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L292)
- `make_custom_splash_sdpa(mesh, env, **kwargs)` — [`L832`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L832)
- `make_splash_mha(block_sizes: _BlockSizes, bkv_compute_in: int = DEFAULT_BKVCOMPUTEINSIZE, orig_q_seq_len: int | None = None, orig_kv_seq_len: int | None = None, heads_per_tile: int = 1, use_base2_exp: bool = True, use_experimental_scheduler: bool = False, vmem_limit_bytes: int | None = None)` — [`L654`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L654)
- `out_index_map(h, i, j, *_)` — [`L391`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L391)
- `out_index_map(h, i, j, *_)` — [`L493`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L493)
- `out_index_map(h, i, j, *_)` — [`L597`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L597)
- `q_index_map(h, i, j, *_)` — [`L388`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L388)
- `q_index_map(h, i, j, *_)` — [`L490`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L490)
- `q_index_map(h, i, j, *_)` — [`L588`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L588)
- `tpu_custom_attention(query, key, value, mesh, *, scale=None, block_q=None, block_kv=None, block_kv_compute=None, block_kv_compute_in=None, heads_per_tile=None, use_base2_exp=True, use_experimental_scheduler=False, vmem_limit_bytes=None, flash_block_sizes=None)` — [`L700`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L700)
- `v_index_map(h, i, j, *_)` — [`L397`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L397)
- `v_index_map(h, i, j, *_)` — [`L499`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L499)
- `v_index_map(h, i, j, *_)` — [`L594`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L594)

## Module values
- `DEFAULT_BKVCOMPUTEINSIZE` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L42)
- `DEFAULT_BKVCOMPUTESIZE` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L40)
- `DEFAULT_BKVSIZE` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L38)
- `DEFAULT_BQSIZE` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L37)
- `DEFAULT_MASK_VALUE` — [`L31`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L31)
- `NT_DIM_NUMBERS` — [`L34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L34)
- `NUM_LANES` — [`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L32)
- `NUM_SUBLANES` — [`L33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/kernels/custom_splash_attention.py#L33)

