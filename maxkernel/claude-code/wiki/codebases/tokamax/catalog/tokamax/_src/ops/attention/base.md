---
title: 'Module: tokamax/_src/ops/attention/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.base`/
symbols:
  DotProductAttention: DotProductAttention#
  Mask: Mask#
  DotProductAttention._fwd: DotProductAttention#_fwd().
  DotProductAttentionVjp._fwd: DotProductAttentionVjp#_fwd().
  Mask.as_array: Mask#as_array().
  DotProductAttentionVjp: DotProductAttentionVjp#
  AUTO: AUTO#
  fold_q_sequence_heads: fold_q_sequence_heads().
  DotProductAttention.bind: DotProductAttention#bind().
  QArray: QArray.
  Residuals: Residuals.
  CanonicalPrecision: CanonicalPrecision.
  Mask.__and__: Mask#__and__().
  Mask.is_causal: Mask#is_causal.
  Mask.q_end: Mask#q_end.
  PagingInfo: PagingInfo#
  Mask.k_end: Mask#k_end.
  Mask.__bool__: Mask#__bool__().
  Mask.k_start: Mask#k_start.
  Mask.q_start: Mask#q_start.
  DotProductAttention.__call__: DotProductAttention#__call__().
  vmap_batch_dims: vmap_batch_dims().
  Mask.take: Mask#take().
  DotProductAttentionGrads: DotProductAttentionGrads#
  DotProductAttention.fwd_sharded: DotProductAttention#fwd_sharded().
  Mask.bool_mask: Mask#bool_mask.
  DotProductAttention.fwd_closed: DotProductAttention#fwd_closed().
  CAUSAL_MASK: CAUSAL_MASK.
  vmap_batch_dims.vmap: vmap_batch_dims().vmap().
  DotProductAttentionVjp.attend: DotProductAttentionVjp#attend().
  _softmax: _softmax().
  Mask.combine: Mask#combine().
  combine_partial_results: combine_partial_results().
  _Config: _Config.
  fold_q_sequence_heads.fold: fold_q_sequence_heads().fold().
  _P: _P.
  DotProductAttentionGrads.q: DotProductAttentionGrads#q.
  DotProductAttentionGrads.k: DotProductAttentionGrads#k.
  DotProductAttentionGrads.v: DotProductAttentionGrads#v.
  unfold_q_sequence_heads: unfold_q_sequence_heads().
  DotProductAttentionGrads.bias: DotProductAttentionGrads#bias.
  _softmax_jvp: _softmax_jvp().
  _Key: _Key.
  needs_stable_softmax: needs_stable_softmax().
  _T: _T.
  PagingInfo.num_active_pages: PagingInfo#num_active_pages.
  PagingInfo.active_page_indices: PagingInfo#active_page_indices.
  DotProductAttention.bcast: DotProductAttention#bcast().
  DotProductAttention.spec: DotProductAttention#spec().
  PagingInfo.lengths: PagingInfo#lengths.
---
# Module: [`tokamax/_src/ops/attention/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py)

## Classes
### `AUTO`
- def: [`tokamax/_src/ops/attention/base.py:41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L41)
- signature: `class AUTO:`
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention._fwd), [`_fwd`](pallas_triton.md#PallasTritonFlashAttention._fwd), [`dot_product_attention`](api.md#dot_product_attention), [`bind`](base.md#DotProductAttention.bind), [`__call__`](base.md#DotProductAttention.__call__), [`use_stable_softmax`](pallas_triton.md#PallasTritonFlashAttention.use_stable_softmax), [`use_stable_softmax`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention.use_stable_softmax)  (1 test-only)

### `DotProductAttention`  ·  implements/extends Op
- def: [`tokamax/_src/ops/attention/base.py:219`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L219) — documented in [tokamax-_src-ops-attention-base](../../../../../concepts/tokamax-_src-ops-attention-base.md)
- doc: Dot product attention function.
- signature: `class DotProductAttention(op.Op[Any, Float[Array, "*B T H d"], Residuals, _Config, _Key]):`
- members:
  - `bcast(x, axes)` — [`L463`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L463)
  - `bind(self, q: Float[Array | QArray, "*B T H D"], k: Float[Array | QArray, "*b t h D"], v: Float[Array | QArray, "*b t h d"], *, precision: (jax.lax.PrecisionLike | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]) = jax.lax.Precision.DEFAULT, logits_dtype: DTypeLike | type[AUTO] = AUTO, logits_scale: float | type[AUTO] = AUTO, bias: Float[Array, "*#B #H #T #t"] | None = None, logits_soft_cap: float | None = None, mask: Bool[Array, "*#B #H #T #t"] | Mask | None = None, is_causal: bool = False, dropout_mask: Bool[Array, "*#B #H #T #t"] | None = None, dropout_rate: float = 0, paging_info: PagingInfo | None = None, q_sharding: jax.sharding.NamedSharding | None = None, k_sharding: jax.sharding.NamedSharding | None = None, q_indices: Int[Array, "*#B #H T"] | None = None, k_indices: Int[Array, "*#b #h t"] | None = None, normalize_output: bool = True, return_residuals: bool = False)` — [`L485`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L485) — Binds the arguments for the attention function. — documented in [tokamax-_src-ops-attention-base](../../../../../concepts/tokamax-_src-ops-attention-base.md)
  - `fwd_closed(q, k, v, bias, mask, dropout_mask, paging_info, q_indices, k_indices)` — [`L377`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L377)
  - `fwd_sharded(*args)` — [`L472`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L472)
  - `spec(x, axes)` — [`L466`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L466)
- protocol/private: `__call__`[`L226`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L226), `_fwd`[`L565`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L565)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention._fwd), [`Mask`](base.md#Mask), [`bind`](../op.md#Op.bind), [`_fwd`](pallas_triton.md#PallasTritonFlashAttention._fwd), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`as_array`](base.md#Mask.as_array), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._fwd), [`_fwd`](jax_nn.md#JaxNnDotProductAttention._fwd), [`AUTO`](base.md#AUTO), [`_fwd`](../flex_attention/wrapper.md#WrappedFlexAttention._fwd), [`kwargs`](../op.md#BoundArguments.kwargs), [`QArray`](base.md#QArray), [`Residuals`](base.md#Residuals), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`_fwd`](xla_chunked.md#XlaChunkedDotProductAttention._fwd), [`is_causal`](base.md#Mask.is_causal), [`PagingInfo`](base.md#PagingInfo), [`PallasTritonFlashAttention`](pallas_triton.md#PallasTritonFlashAttention), [`PallasMosaicTpuFlashAttention`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention), [`__call__`](../op.md#Op.__call__), [`PallasMosaicGpuFlashAttention`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention), [`XlaChunkedDotProductAttention`](xla_chunked.md#XlaChunkedDotProductAttention), [`JaxNnDotProductAttention`](jax_nn.md#JaxNnDotProductAttention), [`WrappedFlexAttention`](../flex_attention/wrapper.md#WrappedFlexAttention), [`CAUSAL_MASK`](base.md#CAUSAL_MASK), [`_softmax`](base.md#_softmax), [`_Config`](base.md#_Config), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`test_roundtrip`](../op_test.md#BoundArgumentsTest.test_roundtrip), [`_fwd`](../op.md#Op._fwd), [`_API_IMPLEMENTATIONS`](../../autotuning/api.md#_API_IMPLEMENTATIONS._API_IMPLEMENTATIONS), [`_IMPLS`](bench.md#_IMPLS), [`_IMPLEMENTATIONS`](api.md#_IMPLEMENTATIONS), [`test_autotune_configs`](pallas_mosaic_gpu_test.md#PallasMosaicGpuFlashAttentionTest.test_autotune_configs), [`PallasTritonFlashAttention`](pallas_triton.md#PallasTritonFlashAttention), [`test_vjp_autotune_configs`](pallas_mosaic_gpu_test.md#PallasMosaicGpuFlashAttentionTest.test_vjp_autotune_configs), [`PallasMosaicTpuFlashAttention`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention), [`PallasMosaicGpuFlashAttention`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention), [`XlaChunkedDotProductAttention`](xla_chunked.md#XlaChunkedDotProductAttention), [`test_autotune_configs`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.test_autotune_configs), [`JaxNnDotProductAttention`](jax_nn.md#JaxNnDotProductAttention), [`WrappedFlexAttention`](../flex_attention/wrapper.md#WrappedFlexAttention), [`__init__`](base_test.md#DotProductAttentionTest.__init__), [`attend`](base.md#DotProductAttentionVjp.attend), [`f_base`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.f_base)

### `DotProductAttentionGrads`  ·  implements/extends _TypedDict
- def: [`tokamax/_src/ops/attention/base.py:807`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L807)
- signature: `class DotProductAttentionGrads(TypedDict):`
- members:
  - `bias` — [`L811`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L811)
  - `k` — [`L809`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L809)
  - `q` — [`L808`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L808)
  - `v` — [`L810`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L810)
- used by: [`_fwd`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp._fwd), [`_fwd`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._fwd), [`_fwd`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._fwd), [`_fwd`](base.md#DotProductAttentionVjp._fwd), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp)

### `DotProductAttentionVjp`  ·  implements/extends Op
- def: [`tokamax/_src/ops/attention/base.py:814`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L814)
- doc: Dot product attention VJP.
- signature: `class DotProductAttentionVjp(op.Op[Any, DotProductAttentionGrads, types.NoneType, _Config, _Key]):`
- members:
  - `_fwd(self, residuals: Residuals, out: Float[Array, "*B T H d"], dout: Float[Array, "*B T H d"], q: Float[Array, "*B T H D"], k: Float[Array, "*b t h D"], v: Float[Array, "*b t h d"], *, precision: tuple[CanonicalPrecision, CanonicalPrecision], logits_dtype: jnp.dtype, logits_scale: float, bias: Float[Array, "*#B #H #T #t"] | None, logits_soft_cap: float | None, mask: Mask, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float, paging_info: PagingInfo | None, q_indices: Int[Array, "*#B #H T"] | None, k_indices: Int[Array, "*#b #h t"] | None, normalize_output: bool, return_residuals: bool, config: _Config)` — [`L821`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L821) — Computes attention VJP.
  - `attend(q, k, v, bias)` — [`L855`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L855)
- uses (calls/refs, reference-scoped): [`Op`](../op.md#Op), [`DotProductAttention`](base.md#DotProductAttention), [`Mask`](base.md#Mask), [`_fwd`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp._fwd), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`_fwd`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._fwd), [`_fwd`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._fwd), [`_fwd`](base.md#DotProductAttention._fwd), [`Residuals`](base.md#Residuals), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`PagingInfo`](base.md#PagingInfo), [`DotProductAttentionGrads`](base.md#DotProductAttentionGrads), [`PallasMosaicTpuFlashAttentionVjp`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp), [`PallasMosaicGpuFlashAttentionVjp`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp), [`PallasTritonFlashAttentionVjp`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp), [`_Config`](base.md#_Config), [`k`](base.md#DotProductAttentionGrads.k), [`q`](base.md#DotProductAttentionGrads.q), [`v`](base.md#DotProductAttentionGrads.v), [`_F32PrecisionXlaAttentionVjp`](pallas_triton_test.md#_F32PrecisionXlaAttentionVjp), [`bias`](base.md#DotProductAttentionGrads.bias), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`_fwd`](../op.md#Op._fwd), [`PallasMosaicTpuFlashAttentionVjp`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp), [`PallasMosaicGpuFlashAttentionVjp`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp), [`PallasTritonFlashAttentionVjp`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp), [`__init__`](base_test.md#DotProductAttentionWithExplicitVjpTest.__init__), [`_F32PrecisionXlaAttentionVjp`](pallas_triton_test.md#_F32PrecisionXlaAttentionVjp)

### `Mask`
- def: [`tokamax/_src/ops/attention/base.py:51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L51) — documented in [tokamax-_src-ops-attention-base](../../../../../concepts/tokamax-_src-ops-attention-base.md)
- doc: An attention mask.
- signature: `class Mask:`
- members:
  - `__and__(self, other: Bool[Array, '*#B #H #T #t'] | Mask)` — [`L167`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L167) — Returns the intersection of two masks.
  - `as_array(self, q_len_or_indices: int | Int[Array, "*#B #H T"], k_len_or_indices: int | Int[Array, "*#B #h t"])` — [`L101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L101) — Returns the mask as a boolean array. — documented in [tokamax-_src-ops-attention-base](../../../../../concepts/tokamax-_src-ops-attention-base.md)
  - `combine(fn)` — [`L172`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L172)
  - `take(self, *attrs: str)` — [`L150`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L150) — Returns a mask with attrs removed and the removed attrs.
  - `bool_mask` — [`L93`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L93)
  - `is_causal` — [`L99`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L99)
  - `k_end` — [`L98`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L98)
  - `k_start` — [`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L97)
  - `q_end` — [`L96`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L96)
  - `q_start` — [`L95`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L95)
- protocol/private: `__bool__`[`L157`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L157)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention._fwd), [`_fwd`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp._fwd), [`_fwd`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._fwd), [`_fwd`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._fwd), [`_fwd`](base.md#DotProductAttention._fwd), [`_fwd`](base.md#DotProductAttentionVjp._fwd), [`dot_product_attention`](api.md#dot_product_attention), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._fwd), [`build_splash_kernel`](pallas_mosaic_tpu_common.md#build_splash_kernel), [`_fwd`](jax_nn.md#JaxNnDotProductAttention._fwd), [`fold_q_sequence_heads`](base.md#fold_q_sequence_heads), [`bind`](base.md#DotProductAttention.bind), [`_run_test_with_inputs`](xla_chunked_test.md#XlaPagedAttentionTest._run_test_with_inputs), [`test_autotune_vjp`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.test_autotune_vjp), [`test_intersection`](base_test.md#MaskTest.test_intersection), [`_attend_chunked`](xla_chunked.md#_attend_chunked), [`__call__`](base.md#DotProductAttention.__call__), [`test_k_range`](base_test.md#MaskTest.test_k_range), [`test_is_causal`](base_test.md#MaskTest.test_is_causal), [`test_q_range`](base_test.md#MaskTest.test_q_range), [`test_take`](base_test.md#MaskTest.test_take), [`Mask`](arg_specs.md#Mask), [`Mask`](xla_chunked.md#Mask), [`CAUSAL_MASK`](base.md#CAUSAL_MASK), [`f`](base_test.md#MaskTest.f), [`test_as_array`](base_test.md#MaskTest.test_as_array), [`Mask`](jax_nn.md#Mask), [`Mask`](pallas_triton_vjp.md#Mask), [`Mask`](pallas_triton.md#Mask), [`Mask`](../flex_attention/wrapper.md#Mask), [`mask_mod`](../flex_attention/wrapper.md#WrappedFlexAttention.mask_mod), [`Mask`](pallas_mosaic_gpu_vjp.md#Mask), [`Mask`](pallas_mosaic_gpu.md#Mask), [`Mask`](../gated_linear_unit/arg_specs.md#Mask)  (7 test-only)

### `PagingInfo`
- def: [`tokamax/_src/ops/attention/base.py:190`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L190)
- doc: Information about the paged kv cache.
- signature: `class PagingInfo:`
- members:
  - `active_page_indices` — [`L208`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L208)
  - `lengths` — [`L209`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L209)
  - `num_active_pages` — [`L207`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L207)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_fwd`](base.md#DotProductAttention._fwd), [`_fwd`](base.md#DotProductAttentionVjp._fwd), [`bind`](base.md#DotProductAttention.bind), [`__call__`](base.md#DotProductAttention.__call__), [`PagingInfo`](xla_chunked.md#PagingInfo), [`bq_loop_fn`](xla_chunked.md#_attend_paged.bq_loop_fn), [`PagingInfo`](jax_nn.md#PagingInfo), [`PagingInfo`](pallas_mosaic_gpu_vjp.md#PagingInfo), [`PagingInfo`](pallas_mosaic_gpu.md#PagingInfo), [`PagingInfo`](pallas_mosaic_tpu_vjp.md#PagingInfo), [`PagingInfo`](pallas_mosaic_tpu.md#PagingInfo), [`PagingInfo`](pallas_triton_vjp.md#PagingInfo), [`PagingInfo`](pallas_triton.md#PagingInfo), [`PagingInfo`](../flex_attention/wrapper.md#PagingInfo), [`PagingInfo`](pallas_mosaic_gpu_kernel_sm100.md#PagingInfo)

## Functions
- `_softmax(x: jax.Array, normalize: bool)` — [`L664`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L664) — Computes softmax, possibly using residual from the forward pass.
- `_softmax_jvp(normalize, primals, tangents)` — [`L676`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L676)
- `combine_partial_results(out: jax.Array, residuals: Residuals, normalize_output: bool)` — [`L790`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L790) — Combine partial attention results (reducing over leading axes).
- `fold(x, has_k_seq=True)` — [`L717`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L717)
- `fold_q_sequence_heads(q: Float[Array, "*B T H D"], bias: Float[Array, "*#B #H #T #t"] | None, mask: Mask, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, q_indices: Int[Array, "*#B #H T"] | None, seq_len_k: int, num_heads_k: int)` — [`L689`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L689) — Folds the q-heads into the q-sequence dimension.
- `needs_stable_softmax(logits_dtype: DTypeLike, logits_soft_cap: float | None)` — [`L645`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L645) — Returns `True` if stable softmax is needed.
- `unfold_q_sequence_heads(out: Float[Array, "*B Tg h d"], residuals: Residuals | None, orig_seq_len_q: int)` — [`L761`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L761) — Unfolds the q-sequence heads from the q-sequence dimension.
- `vmap(f, *args: _P.args, **kwargs: _P.kwargs)` — [`L781`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L781)
- `vmap_batch_dims(f: Callable[_P, _T])` — [`L778`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L778) — Returns `f` vmapped over the batch dims of its first argument.

## Module values
- `CAUSAL_MASK` — [`L185`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L185)
- `CanonicalPrecision` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L45)
- `QArray` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L46)
- `Residuals` — [`L216`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L216)
- `_Config` — [`L212`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L212)
- `_Key` — [`L213`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L213)
- `_P` — [`L774`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L774)
- `_T` — [`L775`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/base.py#L775)

