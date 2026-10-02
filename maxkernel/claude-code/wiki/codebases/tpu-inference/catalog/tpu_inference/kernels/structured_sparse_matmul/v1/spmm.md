---
title: 'Module: tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.structured_sparse_matmul.v1.spmm`/
symbols:
  Sparsifier: Sparsifier#
  Sparsifier.__init__: Sparsifier#__init__().
  Sparsifier.encode: Sparsifier#encode().
  Sparsifier.decode: Sparsifier#decode().
  Sparsifier.compress: Sparsifier#compress().
  Sparsifier.decompress: Sparsifier#decompress().
  next_log2: next_log2().
  next_pow2: next_pow2().
  gen_sparse_mask: gen_sparse_mask().
  _get_dot_general_dim_nums: _get_dot_general_dim_nums().
  _decompress_metadata: _decompress_metadata().
  _decompress_nonzeros: _decompress_nonzeros().
  _get_metadata_packing: _get_metadata_packing().
  _get_dim_mapping: _get_dim_mapping().
  _get_in_sepcs: _get_in_sepcs().
  _infer_out_dtype: _infer_out_dtype().
  _verify: _verify().
  _structured_spmm: _structured_spmm().
  structured_spmm: structured_spmm().
---
# Module: [`tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py)

## Classes
### `Sparsifier`
- def: [`tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py:81`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L81)
- doc: The sparsifier for structured sparse matrix.
- signature: `class Sparsifier:`
- members:
  - `compress(self, data, bitwidth)` — [`L175`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L175) — Pack the rows of data based on the element's bitwidth.
  - `decode(self)` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L139) — Decode nonzeros and metadata to sparse matrix and mask.
  - `decompress(self, data, bitwidth)` — [`L191`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L191) — Unpack the rows of data based on the element's bitwidth.
  - `encode(self, data, mask)` — [`L113`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L113) — Encode data to nonzeros and metadata based on the mask.
- protocol/private: `__init__`[`L96`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L96)

## Functions
- `_decompress_metadata(md_tile_ref, packing)` — [`L267`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L267) — Decompress metadata. Expected to use in a Pallas kernel.
- `_decompress_nonzeros(sparsity, nonzeros, nonzeros_idx, sparse_dim, stride, default_value)` — [`L285`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L285) — Decompress nonzeros. Expected to use in a Pallas kernel.
- `_get_dim_mapping(rhs_sparse: bool, rhs_transpose: bool)` — [`L317`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L317) — The mapping to (m, n, k) dimensions.
- `_get_dot_general_dim_nums(transposed: bool)` — [`L262`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L262)
- `_get_in_sepcs(sparsity: tuple[int, int], sparse_dim: int, rhs_sparse: bool, rhs_transpose: bool, block_m: int, block_k: int, block_n: int)` — [`L326`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L326) — Get in_specs of structured SPMM.
- `_get_metadata_packing(sparsity_base)` — [`L312`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L312)
- `_infer_out_dtype(ty1: jnp.dtype, ty2: jnp.dtype)` — [`L363`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L363) — Infer matmul output dtype.
- `_structured_spmm(sparsity: tuple[int, int], nonzeros: jax.Array, metadata: jax.Array, matrix: jax.Array, *, sparse_dim: int, rhs_sparse: bool, rhs_transpose: bool=False, stride: int, block_m: int, block_k: int, block_n: int, default_value: Any=0, out_dtype: Any=None, debug: bool=False)` — [`L508`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L508) — General structured sparse matmul.
- `_verify(sparsity: tuple[int, int], nz: jax.Array, md: jax.Array, mat: jax.Array, *, sparse_dim: int, rhs_sparse: bool, rhs_transpose: bool, stride: int, block_m: int, block_k: int, block_n: int)` — [`L373`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L373) — Validate inputs of structured SPMM and return original dim size m, n, k.
- `gen_sparse_mask(key: Any, shape: jax.Array | tuple[int, int], sparsity: tuple[int, int], *, sparse_dim: int, stride: int=1)` — [`L216`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L216) — Generates a mask with N:M sparsity on a given dim.
- `next_log2(x)` — [`L208`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L208)
- `next_pow2(x)` — [`L212`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L212)
- `structured_spmm(sparsity: tuple[int, int], nonzeros: jax.Array, metadata: jax.Array, matrix: jax.Array, *, rhs_sparse: bool, contract_sparse: bool, rhs_transpose: bool=False, stride: int, block_m: int, block_k: int, block_n: int, default_value: Any=0, out_dtype: Any=None, debug: bool=False)` — [`L657`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L657) — General structured sparse matmul.

