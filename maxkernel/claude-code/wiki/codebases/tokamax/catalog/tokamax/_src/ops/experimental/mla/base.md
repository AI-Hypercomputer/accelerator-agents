---
title: 'Module: tokamax/_src/ops/experimental/mla/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.base`/
symbols:
  MultiHeadLatentAttention: MultiHeadLatentAttention#
  MultiHeadLatentAttention.bind: MultiHeadLatentAttention#bind().
  MultiHeadLatentAttention._fwd: MultiHeadLatentAttention#_fwd().
  _Config: _Config.
---
# Module: [`tokamax/_src/ops/experimental/mla/base.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base.py)

## Classes
### `MultiHeadLatentAttention`  ·  implements/extends Op
- def: [`tokamax/_src/ops/experimental/mla/base.py:29`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base.py#L29)
- doc: Tokamax operator for Multi-Head Latent Attention.
- signature: `class MultiHeadLatentAttention(op.Op[Any, Any, None, _Config, Any]):`
- members:
  - `bind(self, ql_nope: Float[Array, "max_num_tokens actual_num_q_heads actual_lkv_dim"], q_pe: Float[Array, "max_num_tokens actual_num_q_heads actual_r_dim"], new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"], new_k_pe: Float[Array, "max_num_tokens actual_r_dim"], cache_kv: Float[Array, "total_num_pages page_size_per_kv_packing kv_packing lkv_dim",], kv_lens: Int[Array, max_num_seqs], page_indices: Int[Array, num_page_indices], cu_q_lens: Int[Array, max_num_seqs_plus_1], distribution: Int[Array, 3], *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, s_dtype: jax.typing.DTypeLike = jnp.bfloat16, debug_mode: bool = False, return_residuals: bool = False)` — [`L33`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base.py#L33)
- protocol/private: `_fwd`[`L98`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base.py#L98)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../../op.md#BoundArguments), [`Op`](../../op.md#Op), [`bind`](../../op.md#Op.bind), [`jaxtyped`](../../../jaxtyping.md#jaxtyped), [`_fwd`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention._fwd), [`mla_attention`](reference.md#mla_attention), [`PallasTpuMultiHeadLatentAttention`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention), [`_Config`](base.md#_Config)
- used by: [`Op`](../../op.md#Op), [`bind`](../../op.md#Op.bind), [`_fwd`](../../op.md#Op._fwd), [`_API_IMPLEMENTATIONS`](../../../autotuning/api.md#_API_IMPLEMENTATIONS._API_IMPLEMENTATIONS), [`test_reference_running_correctly`](base_test.md#BaselineMlaTest.test_reference_running_correctly), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`test_cu_q_lens_shape_validation`](base_test.md#BaselineMlaTest.test_cu_q_lens_shape_validation), [`test_page_indices_divisibility_validation`](base_test.md#BaselineMlaTest.test_page_indices_divisibility_validation), [`test_shape_mismatch_validation`](base_test.md#BaselineMlaTest.test_shape_mismatch_validation), [`PallasTpuMultiHeadLatentAttention`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention), [`ReferenceMhla`](pallas_mosaic_tpu_test.md#ReferenceMhla)

## Module values
- `_Config` — [`L26`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base.py#L26)

