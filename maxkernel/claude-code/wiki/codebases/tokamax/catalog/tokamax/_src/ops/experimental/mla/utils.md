---
title: 'Module: tokamax/_src/ops/experimental/mla/utils.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/utils.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.utils`/
symbols:
  generate_mla_inputs: generate_mla_inputs().
  cdiv: cdiv().
  get_dtype_packing: get_dtype_packing().
  generate_mla_inputs.gen_random: generate_mla_inputs().gen_random().
  static_validate_inputs: static_validate_inputs().
  unsigned_align_to: unsigned_align_to().
  align_to: align_to().
  get_dtype_bitwidth: get_dtype_bitwidth().
  unsigned_cdiv: unsigned_cdiv().
---
# Module: [`tokamax/_src/ops/experimental/mla/utils.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py)

## Functions
- `align_to(a, b)` — [`L28`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L28)
- `cdiv(a, b)` — [`L23`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L23)
- `gen_random(shape, dtype)` — [`L82`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L82)
- `generate_mla_inputs(seq_lens, num_heads, lkv_dim, r_dim, page_size, q_dtype, kv_dtype, num_pages, rng=None, with_kv_cache=True)` — [`L41`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L41) — Generates inputs for the MLA kernel.
- `get_dtype_bitwidth(dtype)` — [`L32`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L32)
- `get_dtype_packing(dtype)` — [`L36`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L36)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_blocks: tuple[int, int, int] | None = None, num_queries_per_blocks: tuple[int, int, int] | None = None, vmem_limit_bytes: int | None = None, decode_batch_size: int = 1, transpose_kv_cache: bool = False, debug_mode: bool = False)` — [`L163`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L163) — Validate inputs to the MLA RPA implementation statically.
- `unsigned_align_to(a, b)` — [`L156`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L156)
- `unsigned_cdiv(a, b)` — [`L149`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/utils.py#L149)

