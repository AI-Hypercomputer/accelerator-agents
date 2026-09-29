---
title: 'Module: tokamax/_src/ops/attention/pallas_triton.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_triton.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_triton`/
symbols:
  PallasTritonFlashAttention._fwd: PallasTritonFlashAttention#_fwd().
  _fwd: _fwd().
  PallasTritonFlashAttention._get_heuristics_config: PallasTritonFlashAttention#_get_heuristics_config().
  PallasTritonFlashAttention._get_autotuning_configs: PallasTritonFlashAttention#_get_autotuning_configs().
  PallasTritonFlashAttention: PallasTritonFlashAttention#
  QArray: QArray.
  _fwd_kernel_impl.body: _fwd_kernel_impl().body().
  Config: Config#
  _fwd_kernel_impl: _fwd_kernel_impl().
  _broadcast_to_rank: _broadcast_to_rank().
  _fwd.input_spec: _fwd().input_spec().
  Config.block_q: Config#block_q.
  _fwd.spec: _fwd().spec().
  _fwd_kernel: _fwd_kernel().
  Config.block_k: Config#block_k.
  Config.num_stages: Config#num_stages.
  Config.num_warps: Config#num_warps.
  _fwd_kernel_impl.get_values_and_scales: _fwd_kernel_impl().get_values_and_scales().
  _fwd.bias_mask_spec: _fwd().bias_mask_spec().
  PallasTritonFlashAttention.__post_init__: PallasTritonFlashAttention#__post_init__().
  Residuals: Residuals.
  _fwd_kernel.fwd_kernel: _fwd_kernel().fwd_kernel().
  _fwd.range_k_spec: _fwd().range_k_spec().
  _fwd.range_q_spec: _fwd().range_q_spec().
  _can_have_block_d: _can_have_block_d().
  PallasTritonFlashAttention.use_stable_softmax: PallasTritonFlashAttention#use_stable_softmax.
  Config.block_d: Config#block_d.
  Config.block_d_out: Config#block_d_out.
  Mask: Mask.
  PagingInfo: PagingInfo.
  _decompose_mask: _decompose_mask().
  _rescale: _rescale().
  PallasTritonFlashAttention.config_cls: PallasTritonFlashAttention#config_cls.
  Config.split_k: Config#split_k.
  _fwd_kernel._zero_store: _fwd_kernel()._zero_store().
  _fwd.spec.wrapped_index_map: _fwd().spec().wrapped_index_map().
  Config.pack_mask: Config#pack_mask.
  PallasTritonFlashAttention.use_base2: PallasTritonFlashAttention#use_base2.
  PallasTritonFlashAttention.supports_symbolic_shapes: PallasTritonFlashAttention#supports_symbolic_shapes.
  PallasTritonFlashAttention.supported_on: PallasTritonFlashAttention#supported_on().
---
# Module: [`tokamax/_src/ops/attention/pallas_triton.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/attention/pallas_triton.py:526`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L526) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
- members:
  - `block_d` — [`L531`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L531) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
  - `block_d_out` — [`L532`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L532) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
  - `block_k` — [`L528`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L528) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
  - `block_q` — [`L527`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L527) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
  - `num_stages` — [`L529`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L529) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
  - `num_warps` — [`L530`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L530) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
  - `pack_mask` — [`L534`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L534)
  - `split_k` — [`L533`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L533)
- used by: [`_fwd`](pallas_triton.md#PallasTritonFlashAttention._fwd), [`_fwd`](pallas_triton.md#_fwd), [`test_block_d`](pallas_triton_test.md#PallasTritonFlashAttentionTest.test_block_d), [`_get_heuristics_config`](pallas_triton.md#PallasTritonFlashAttention._get_heuristics_config), [`test_small_block_q`](pallas_triton_test.md#PallasTritonFlashAttentionTest.test_small_block_q), [`impl`](pallas_triton_test.md#PallasTritonFlashAttentionTest.impl), [`_get_autotuning_configs`](pallas_triton.md#PallasTritonFlashAttention._get_autotuning_configs), [`PallasTritonFlashAttention`](pallas_triton.md#PallasTritonFlashAttention), [`config_cls`](pallas_triton.md#PallasTritonFlashAttention.config_cls)

### `PallasTritonFlashAttention`  ·  implements/extends DotProductAttention
- def: [`tokamax/_src/ops/attention/pallas_triton.py:538`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L538)
- doc: Pallas-Triton FlashAttention implementation.
- signature: `class PallasTritonFlashAttention(base.DotProductAttention[Config, None]):`
- members:
  - `supported_on(self, device: jax.Device)` — [`L693`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L693)
  - `config_cls` — [`L541`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L541)
  - `supports_symbolic_shapes` — [`L542`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L542)
  - `use_base2` — [`L543`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L543)
  - `use_stable_softmax` — [`L544`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L544) — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
- protocol/private: `__post_init__`[`L546`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L546), `_fwd`[`L552`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L552), `_get_autotuning_configs`[`L674`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L674), `_get_heuristics_config`[`L641`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L641)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`DotProductAttention`](base.md#DotProductAttention), [`_fwd`](pallas_triton.md#_fwd), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`args`](../op.md#BoundArguments.args), [`vjp`](../op.md#Op.vjp), [`AUTO`](base.md#AUTO), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`QArray`](pallas_triton.md#QArray), [`Config`](pallas_triton.md#Config), [`vmap_batch_dims`](base.md#vmap_batch_dims), [`_broadcast_to_rank`](pallas_triton.md#_broadcast_to_rank), [`vmap_split`](../../batching.md#vmap_split), [`infer_devices`](../op.md#infer_devices), [`PallasTritonFlashAttentionVjp`](pallas_triton_vjp.md#PallasTritonFlashAttentionVjp), [`block_q`](pallas_triton.md#Config.block_q), [`block_k`](pallas_triton.md#Config.block_k), [`num_stages`](pallas_triton.md#Config.num_stages), [`num_warps`](pallas_triton.md#Config.num_warps), [`Residuals`](pallas_triton.md#Residuals), [`_can_have_block_d`](pallas_triton.md#_can_have_block_d), [`combine_partial_results`](base.md#combine_partial_results), [`block_d`](pallas_triton.md#Config.block_d), [`block_d_out`](pallas_triton.md#Config.block_d_out), [`Mask`](pallas_triton.md#Mask), [`PagingInfo`](pallas_triton.md#PagingInfo), [`_decompose_mask`](pallas_triton.md#_decompose_mask), [`needs_stable_softmax`](base.md#needs_stable_softmax), [`split_k`](pallas_triton.md#Config.split_k)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`test_roundtrip`](../op_test.md#BoundArgumentsTest.test_roundtrip), [`_fwd`](base.md#DotProductAttention._fwd), [`test_block_d`](pallas_triton_test.md#PallasTritonFlashAttentionTest.test_block_d), [`test_small_block_q`](pallas_triton_test.md#PallasTritonFlashAttentionTest.test_small_block_q), [`_IMPLS`](bench.md#_IMPLS), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`__init__`](pallas_triton_test.md#PallasTritonFlashAttentionWithPallasTritonVjpTest.__init__), [`test_split_k`](pallas_triton_test.md#PallasTritonFlashAttentionTest.test_split_k), [`__init__`](pallas_triton_test.md#PallasTritonFlashAttentionTest.__init__)

## Functions
- `_broadcast_to_rank(x, rank)` — [`L465`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L465)
- `_can_have_block_d(*args)` — [`L514`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L514)
- `_decompose_mask(mask, q, k, q_indices, k_indices)` — [`L469`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L469) — Decomposes `mask` into a mask array, `is_causal` and {k,q}_{start,end}.
- `_fwd(q: Float[Array | QArray, "T H D"], k: Float[Array | QArray, "t h D"], v: Float[Array | QArray, "t h d"], bias: Float[Array, "#H #T #t"] | None, mask: Bool[Array, "#H #T #t"] | None, dropout_mask: Bool[Array, "#H #T #t"] | None, q_start: Int[Array, "#H #t"] | None, q_end: Int[Array, "#H #t"] | None, k_start: Int[Array, "#H #T"] | None, k_end: Int[Array, "#H #T"] | None, *, dropout_rate: float, logits_dtype: jnp.dtype, logits_scale: float, logits_soft_cap: float | None, is_causal: bool, config: Config, use_base2: bool, use_stable_softmax: bool, normalize_output: bool, return_residuals: bool, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset)` — [`L313`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L313) — Forward pass of Pallas FlashAttention.
- `_fwd_kernel(q_start_ref, q_end_ref, q_ref, *args, **kwargs)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L47) — Pallas MHA forward kernel conditioned on q_start and q_end.
- `_fwd_kernel_impl(q_start, q_end, q_ref, k_ref, v_ref, bias_ref, mask_ref, dropout_mask_ref, k_start_ref, k_end_ref, out_ref, l_ref, m_ref, block_k: int, block_d: int | None, block_d_out: int | None, sm_scale: float, is_causal: bool, dropout_rate: float, logits_dtype: jnp.dtype, logits_soft_cap: float | None, use_base2: bool, use_stable_softmax: bool, q_k_dot_precision: jax.lax.DotAlgorithmPreset, weights_v_dot_precision: jax.lax.DotAlgorithmPreset, normalize_output: bool, pack_mask: bool)` — [`L101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L101) — Pallas MHA forward kernel implementation. — documented in [tokamax-_src-ops-attention-pallas_triton](../../../../../concepts/tokamax-_src-ops-attention-pallas_triton.md)
- `_rescale(values, scales_ref, slice_k, slice_d, keep_quantized=False, const_scale=1)` — [`L75`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L75) — Load the scales correctly and apply them to the values if requested.
- `_zero_store()` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L57)
- `bias_mask_spec(x)` — [`L404`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L404)
- `body(i, carry, *, is_causal=False, use_k_end=False)` — [`L185`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L185)
- `fwd_kernel(q_start, q_end)` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L50)
- `get_values_and_scales(x)` — [`L132`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L132)
- `input_spec(x, index_map, block_shape)` — [`L390`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L390)
- `range_k_spec(x)` — [`L408`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L408)
- `range_q_spec(x)` — [`L411`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L411)
- `spec(x, index_map, block_shape)` — [`L377`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L377)
- `wrapped_index_map(*args)` — [`L381`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L381)

## Module values
- `Mask` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L41)
- `PagingInfo` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L44)
- `QArray` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L42)
- `Residuals` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_triton.py#L43)

