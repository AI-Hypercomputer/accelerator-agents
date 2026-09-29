---
title: 'Module: jax/_src/cudnn/fused_attention_stablehlo.py'
type: catalog
provenance: extracted
module: jax/_src/cudnn/fused_attention_stablehlo.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.cudnn.fused_attention_stablehlo`/
symbols:
  _dot_product_attention_bwd_p_wrapper: _dot_product_attention_bwd_p_wrapper.
  _dot_product_attention_fp8_bwd_p_wrapper: _dot_product_attention_fp8_bwd_p_wrapper.
  dot_product_attention: dot_product_attention().
  paged_attention: paged_attention().
  _dot_product_attention_fwd_p_wrapper: _dot_product_attention_fwd_p_wrapper.
  _dot_product_attention_fp8_fwd_p_wrapper: _dot_product_attention_fp8_fwd_p_wrapper.
  MaskType: MaskType#
  _attention_out_aval: _attention_out_aval().
  _dot_product_attention_bwd_batcher: _dot_product_attention_bwd_batcher().
  _dot_product_attention_fwd_p: _dot_product_attention_fwd_p.
  _dot_product_attention_bwd_p: _dot_product_attention_bwd_p.
  _dot_product_attention_fp8_fwd_p: _dot_product_attention_fp8_fwd_p.
  _dot_product_attention_fp8_bwd_p: _dot_product_attention_fp8_bwd_p.
  check_layout: check_layout().
  _dot_product_attention_fwd_cuda_lowering: _dot_product_attention_fwd_cuda_lowering().
  _dot_product_attention_fwd_batcher: _dot_product_attention_fwd_batcher().
  _dot_product_attention_fp8_fwd_cuda_lowering: _dot_product_attention_fp8_fwd_cuda_lowering().
  _dot_product_attention_fp8_bwd_cuda_lowering: _dot_product_attention_fp8_bwd_cuda_lowering().
  AttentionLayout: AttentionLayout#
  _dot_product_attention_bwd_cuda_lowering: _dot_product_attention_bwd_cuda_lowering().
  _dot_product_attention_fp8_fwd_batcher: _dot_product_attention_fp8_fwd_batcher().
  _dot_product_attention_fp8_bwd_batcher: _dot_product_attention_fp8_bwd_batcher().
  _infer_fwd_output_sharding: _infer_fwd_output_sharding().
  AttentionLayout.BNTH: AttentionLayout#BNTH.
  _fwd_shardy_rule: _fwd_shardy_rule().
  convert_mask_type_to_string: convert_mask_type_to_string().
  _dot_product_attention_fp8_fwd_rule: _dot_product_attention_fp8_fwd_rule().
  _dot_product_attention_fp8: _dot_product_attention_fp8().
  combine_bias_and_mask: combine_bias_and_mask().
  has_padding: has_padding().
  _dot_product_attention_fwd: _dot_product_attention_fwd().
  check_is_flash_attention: check_is_flash_attention().
  _infer_bwd_output_sharding: _infer_bwd_output_sharding().
  default_layouts: default_layouts().
  create_dot_product_attention_backend_config_base: create_dot_product_attention_backend_config_base().
  _dot_product_attention_fwd_abstract: _dot_product_attention_fwd_abstract().
  _dot_product_attention_fp8_fwd_abstract: _dot_product_attention_fp8_fwd_abstract().
  _fix_seqlen_offsets._shift_to_left: _fix_seqlen_offsets()._shift_to_left().
  _dot_product_attention_fwd_rule: _dot_product_attention_fwd_rule().
  _dot_product_attention_bwd_partition.sharded_impl: _dot_product_attention_bwd_partition().sharded_impl().
  _dot_product_attention_fwd_impl: _dot_product_attention_fwd_impl().
  _dot_product_attention_bwd_impl: _dot_product_attention_bwd_impl().
  _dot_product_attention: _dot_product_attention().
  _fix_seqlen_offsets: _fix_seqlen_offsets().
  _bwd_shardy_rule: _bwd_shardy_rule().
  _infer_fp8_fwd_output_sharding: _infer_fp8_fwd_output_sharding().
  _infer_fp8_bwd_output_sharding: _infer_fp8_bwd_output_sharding().
  _dot_product_attention_fp8_fwd: _dot_product_attention_fp8_fwd().
  _dot_product_attention_fp8_bwd_rule: _dot_product_attention_fp8_bwd_rule().
  check_compute_capability: check_compute_capability().
  _get_padded_spec: _get_padded_spec().
  _dot_product_attention_fp8_fwd_impl: _dot_product_attention_fp8_fwd_impl().
  _dot_product_attention_fp8_bwd_impl: _dot_product_attention_fp8_bwd_impl().
  MaskType.NO_MASK: MaskType#NO_MASK.
  MaskType.CAUSAL: MaskType#CAUSAL.
  _fix_seqlen_offsets._cu_offset: _fix_seqlen_offsets()._cu_offset().
  _check_qkv_bias_mask_spec: _check_qkv_bias_mask_spec().
  _dot_product_attention_fwd_lower: _dot_product_attention_fwd_lower.
  _dot_product_attention_bwd_lower: _dot_product_attention_bwd_lower.
  _dot_product_attention_fp8_bwd_abstract: _dot_product_attention_fp8_bwd_abstract().
  _dot_product_attention_fp8_fwd_lower: _dot_product_attention_fp8_fwd_lower.
  _dot_product_attention_fp8_bwd_lower: _dot_product_attention_fp8_bwd_lower.
  get_max_seg_per_batch: get_max_seg_per_batch().
  is_cuda_compute_capability_equal: is_cuda_compute_capability_equal().
  _dot_product_attention_bwd_rule: _dot_product_attention_bwd_rule().
  _dot_product_attention_fwd_partition: _dot_product_attention_fwd_partition().
  _dot_product_attention_bwd_partition: _dot_product_attention_bwd_partition().
  _dot_product_attention_fp8_fwd_partition: _dot_product_attention_fp8_fwd_partition().
  _dot_product_attention_fp8_bwd_partition: _dot_product_attention_fp8_bwd_partition().
  get_custom_call_name: get_custom_call_name().
  _normalize_layout: _normalize_layout().
  create_dot_product_attention_backend_config: create_dot_product_attention_backend_config().
  create_dot_product_attention_fp8_backend_config: create_dot_product_attention_fp8_backend_config().
  get_fp8_custom_call_name: get_fp8_custom_call_name.
  check_cudnn_version: check_cudnn_version().
  _dot_product_attention_bwd_abstract: _dot_product_attention_bwd_abstract().
  check_is_flash_attention_fp8: check_is_flash_attention_fp8.
  check_layout.check_seqlen_offsets: check_layout().check_seqlen_offsets().
  _check_valid_batch_dims: _check_valid_batch_dims().
  get_large_negative_number: get_large_negative_number().
  _dot_product_attention_fwd_infer_sharding_from_operands: _dot_product_attention_fwd_infer_sharding_from_operands().
  _dot_product_attention_fwd_shardy_rule: _dot_product_attention_fwd_shardy_rule().
  _dot_product_attention_bwd_infer_sharding_from_operands: _dot_product_attention_bwd_infer_sharding_from_operands().
  _dot_product_attention_bwd_shardy_rule: _dot_product_attention_bwd_shardy_rule().
  check_fp8_params: check_fp8_params().
  _dot_product_attention_fp8_fwd_infer_sharding_from_operands: _dot_product_attention_fp8_fwd_infer_sharding_from_operands().
  _dot_product_attention_fp8_fwd_shardy_rule: _dot_product_attention_fp8_fwd_shardy_rule().
  _dot_product_attention_fp8_bwd_infer_sharding_from_operands: _dot_product_attention_fp8_bwd_infer_sharding_from_operands().
  _dot_product_attention_fp8_bwd_shardy_rule: _dot_product_attention_fp8_bwd_shardy_rule().
  MaskType.PADDING: MaskType#PADDING.
  MaskType.PADDING_CAUSAL: MaskType#PADDING_CAUSAL.
  check_layout.check_eq: check_layout().check_eq().
  params_from_keys: params_from_keys.
  FP8Params: FP8Params#
  should_export_dbias: should_export_dbias().
  check_is_paged_attention: check_is_paged_attention().
  fp8_params_keys: fp8_params_keys.
  fp8_params_keys_fwd: fp8_params_keys_fwd.
  AttentionLayout.BTNH: AttentionLayout#BTNH.
  MaskType.ALIBI: MaskType#ALIBI.
  element_type_to_backend_config_type_mapping: element_type_to_backend_config_type_mapping().
  _custom_name_maps: _custom_name_maps.
  fp8_params_keys_bwd: fp8_params_keys_bwd.
  FP8Params.amax_dQ: FP8Params#amax_dQ.
  FP8Params.amax_dK: FP8Params#amax_dK.
  FP8Params.amax_dV: FP8Params#amax_dV.
  FP8Params.amax_dP: FP8Params#amax_dP.
  FP8Params.descale_q: FP8Params#descale_q.
  FP8Params.descale_k: FP8Params#descale_k.
  FP8Params.descale_v: FP8Params#descale_v.
  FP8Params.descale_s: FP8Params#descale_s.
  FP8Params.scale_s: FP8Params#scale_s.
  FP8Params.scale_o: FP8Params#scale_o.
  FP8Params.descale_o: FP8Params#descale_o.
  FP8Params.descale_dO: FP8Params#descale_dO.
  FP8Params.descale_dP: FP8Params#descale_dP.
  FP8Params.scale_dQ: FP8Params#scale_dQ.
  FP8Params.scale_dK: FP8Params#scale_dK.
  FP8Params.scale_dV: FP8Params#scale_dV.
  FP8Params.scale_dP: FP8Params#scale_dP.
---
# Module: [`jax/_src/cudnn/fused_attention_stablehlo.py`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py)

## Classes
### `AttentionLayout`  ·  implements/extends Enum
- def: [`jax/_src/cudnn/fused_attention_stablehlo.py:61`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L61)
- signature: `class AttentionLayout(enum.Enum):`
- members:
  - `BNTH` — [`L63`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L63)
  - `BTNH` — [`L62`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L62)
- used by: [`_dot_product_attention_bwd_batcher`](fused_attention_stablehlo.md#_dot_product_attention_bwd_batcher), [`check_layout`](fused_attention_stablehlo.md#check_layout), [`_dot_product_attention_fp8_bwd_cuda_lowering`](fused_attention_stablehlo.md#_dot_product_attention_fp8_bwd_cuda_lowering), [`_dot_product_attention_fp8_fwd_cuda_lowering`](fused_attention_stablehlo.md#_dot_product_attention_fp8_fwd_cuda_lowering), [`_dot_product_attention_fwd_batcher`](fused_attention_stablehlo.md#_dot_product_attention_fwd_batcher), [`_dot_product_attention_fwd_cuda_lowering`](fused_attention_stablehlo.md#_dot_product_attention_fwd_cuda_lowering), [`_dot_product_attention_bwd_cuda_lowering`](fused_attention_stablehlo.md#_dot_product_attention_bwd_cuda_lowering), [`_dot_product_attention_fp8_bwd_batcher`](fused_attention_stablehlo.md#_dot_product_attention_fp8_bwd_batcher), [`_dot_product_attention_fp8_fwd_batcher`](fused_attention_stablehlo.md#_dot_product_attention_fp8_fwd_batcher), [`_infer_fwd_output_sharding`](fused_attention_stablehlo.md#_infer_fwd_output_sharding), [`_fwd_shardy_rule`](fused_attention_stablehlo.md#_fwd_shardy_rule), [`check_is_flash_attention`](fused_attention_stablehlo.md#check_is_flash_attention), [`_dot_product_attention_fp8_fwd_abstract`](fused_attention_stablehlo.md#_dot_product_attention_fp8_fwd_abstract), [`_dot_product_attention_fwd_abstract`](fused_attention_stablehlo.md#_dot_product_attention_fwd_abstract), [`create_dot_product_attention_backend_config_base`](fused_attention_stablehlo.md#create_dot_product_attention_backend_config_base), [`sharded_impl`](fused_attention_stablehlo.md#_dot_product_attention_bwd_partition.sharded_impl), [`_check_qkv_bias_mask_spec`](fused_attention_stablehlo.md#_check_qkv_bias_mask_spec), [`_normalize_layout`](fused_attention_stablehlo.md#_normalize_layout)

### `FP8Params`  ·  implements/extends _TypedDict
- def: [`jax/_src/cudnn/fused_attention_stablehlo.py:41`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L41)
- signature: `class FP8Params(TypedDict):`
- members:
  - `amax_dK` — [`L43`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L43)
  - `amax_dP` — [`L45`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L45)
  - `amax_dQ` — [`L42`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L42)
  - `amax_dV` — [`L44`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L44)
  - `descale_dO` — [`L53`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L53)
  - `descale_dP` — [`L54`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L54)
  - `descale_k` — [`L47`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L47)
  - `descale_o` — [`L52`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L52)
  - `descale_q` — [`L46`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L46)
  - `descale_s` — [`L49`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L49)
  - `descale_v` — [`L48`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L48)
  - `scale_dK` — [`L56`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L56)
  - `scale_dP` — [`L58`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L58)
  - `scale_dQ` — [`L55`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L55)
  - `scale_dV` — [`L57`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L57)
  - `scale_o` — [`L51`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L51)
  - `scale_s` — [`L50`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L50)
- used by: [`dot_product_attention`](fused_attention_stablehlo.md#dot_product_attention), [`paged_attention`](fused_attention_stablehlo.md#paged_attention)

### `MaskType`  ·  implements/extends Enum
- def: [`jax/_src/cudnn/fused_attention_stablehlo.py:66`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L66)
- signature: `class MaskType(enum.Enum):`
- members:
  - `ALIBI` — [`L71`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L71)
  - `CAUSAL` — [`L69`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L69)
  - `NO_MASK` — [`L67`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L67)
  - `PADDING` — [`L68`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L68)
  - `PADDING_CAUSAL` — [`L70`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L70)
- used by: [`dot_product_attention`](fused_attention_stablehlo.md#dot_product_attention), [`paged_attention`](fused_attention_stablehlo.md#paged_attention), [`_dot_product_attention_fp8_bwd_cuda_lowering`](fused_attention_stablehlo.md#_dot_product_attention_fp8_bwd_cuda_lowering), [`_dot_product_attention_fp8_fwd_cuda_lowering`](fused_attention_stablehlo.md#_dot_product_attention_fp8_fwd_cuda_lowering), [`convert_mask_type_to_string`](fused_attention_stablehlo.md#convert_mask_type_to_string), [`has_padding`](fused_attention_stablehlo.md#has_padding)

## Functions
- `_attention_out_aval(in_aval, shape=None, dtype=None)` — [`L566`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L566)
- `_bwd_shardy_rule(num_args, has_dbias, is_fp8)` — [`L1051`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1051)
- `_check_qkv_bias_mask_spec(query_spec, key_spec, value_spec, bias_spec, layout)` — [`L920`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L920)
- `_check_valid_batch_dims(bdims)` — [`L802`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L802)
- `_cu_offset(offsets, max_seq)` — [`L506`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L506)
- `_dot_product_attention(query: Array, key: Array, value: Array, bias: Array, q_seqlen: Array, kv_seqlen: Array, q_offsets: Array, kv_offsets: Array, page_table_k: Array, page_table_v: Array, scale: float, seed: int, dropout_rate: float, variadic_args: tuple[bool, ...], mask_type: bool, layout: int, sliding_window_length: int | None, cudnn_version: int, return_residual: bool)` — [`L1216`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1216)
- `_dot_product_attention_bwd_abstract(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, activation, fwd_output, grad_output, *, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length)` — [`L611`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L611)
- `_dot_product_attention_bwd_batcher(batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length)` — [`L855`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L855)
- `_dot_product_attention_bwd_cuda_lowering(ctx, query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, activation, fwd_output, grad_output, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length)` — [`L712`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L712)
- `_dot_product_attention_bwd_impl(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, activation, fwd_output, grad_output, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length)` — [`L551`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L551)
- `_dot_product_attention_bwd_infer_sharding_from_operands(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, mesh, arg_shapes, result_shape)` — [`L1067`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1067)
- `_dot_product_attention_bwd_partition(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, mesh, arg_shapes, result_shape)` — [`L1078`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1078)
- `_dot_product_attention_bwd_rule(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training, return_residual, res, grad_output)` — [`L466`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L466)
- `_dot_product_attention_bwd_shardy_rule(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, mesh, value_types, result_types)` — [`L1072`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1072)
- `_dot_product_attention_fp8(query: Array, key: Array, value: Array, fp8_params: dict[str, Array], scale: float, use_causal_mask: bool, layout: int, cudnn_version: int)` — [`L1810`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1810)
- `_dot_product_attention_fp8_bwd_abstract(query, key, value, fwd_output, grad_output, activation, descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, scale, use_causal_mask, layout)` — [`L1390`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1390)
- `_dot_product_attention_fp8_bwd_batcher(batched_args, batch_dims, *, scale, use_causal_mask, layout)` — [`L1603`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1603)
- `_dot_product_attention_fp8_bwd_cuda_lowering(ctx, query, key, value, fwd_output, grad_output, activation, descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, scale, use_causal_mask, layout)` — [`L1477`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1477)
- `_dot_product_attention_fp8_bwd_impl(query, key, value, fwd_output, grad_output, activation, descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, scale, use_causal_mask, layout)` — [`L1349`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1349)
- `_dot_product_attention_fp8_bwd_infer_sharding_from_operands(scale, use_causal_mask, layout, mesh, arg_shapes, result_shape)` — [`L1702`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1702)
- `_dot_product_attention_fp8_bwd_partition(scale, use_causal_mask, layout, mesh, arg_shapes, result_shape)` — [`L1711`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1711)
- `_dot_product_attention_fp8_bwd_rule(scale, use_causal_mask, layout, cudnn_version, res, g)` — [`L1303`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1303)
- `_dot_product_attention_fp8_bwd_shardy_rule(scale, use_causal_mask, layout, mesh, value_types, result_types)` — [`L1707`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1707)
- `_dot_product_attention_fp8_fwd(query, key, value, fp8_params_fwd, scale, use_causal_mask, layout, cudnn_version)` — [`L1276`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1276)
- `_dot_product_attention_fp8_fwd_abstract(query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, scale, use_causal_mask, layout, is_training)` — [`L1362`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1362)
- `_dot_product_attention_fp8_fwd_batcher(batched_args, batch_dims, *, scale, use_causal_mask, layout, is_training)` — [`L1565`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1565)
- `_dot_product_attention_fp8_fwd_cuda_lowering(ctx, query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, scale, use_causal_mask, layout, is_training)` — [`L1406`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1406)
- `_dot_product_attention_fp8_fwd_impl(query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, scale, use_causal_mask, layout, is_training)` — [`L1328`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1328)
- `_dot_product_attention_fp8_fwd_infer_sharding_from_operands(scale, use_causal_mask, layout, is_training, mesh, arg_shapes, result_shape)` — [`L1659`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1659)
- `_dot_product_attention_fp8_fwd_partition(scale, use_causal_mask, layout, is_training, mesh, arg_shapes, result_shape)` — [`L1664`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1664)
- `_dot_product_attention_fp8_fwd_rule(query, key, value, fp8_params, scale, use_causal_mask, layout, cudnn_version)` — [`L1290`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1290)
- `_dot_product_attention_fp8_fwd_shardy_rule(scale, use_causal_mask, layout, is_training, mesh, value_types, result_types)` — [`L1676`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1676)
- `_dot_product_attention_fwd(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, cudnn_version, return_residual)` — [`L426`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L426)
- `_dot_product_attention_fwd_abstract(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, *, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training)` — [`L585`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L585)
- `_dot_product_attention_fwd_batcher(batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training)` — [`L808`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L808)
- `_dot_product_attention_fwd_cuda_lowering(ctx, query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training)` — [`L631`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L631)
- `_dot_product_attention_fwd_impl(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training)` — [`L537`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L537)
- `_dot_product_attention_fwd_infer_sharding_from_operands(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training, mesh, arg_shapes, result_shape)` — [`L999`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L999)
- `_dot_product_attention_fwd_partition(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training, mesh, arg_shapes, result_shape)` — [`L1009`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1009)
- `_dot_product_attention_fwd_rule(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, cudnn_version, return_residual)` — [`L445`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L445)
- `_dot_product_attention_fwd_shardy_rule(scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, is_training, mesh, value_types, result_types)` — [`L1004`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1004)
- `_fix_seqlen_offsets(q_seqlen, kv_seqlen, q_offsets, kv_offsets, query, key)` — [`L483`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L483)
- `_fwd_shardy_rule(value_types, result_types, layout, is_training, is_fp8)` — [`L970`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L970)
- `_get_padded_spec(arg_info)` — [`L912`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L912)
- `_infer_bwd_output_sharding(mesh, arg_shapes, layout, variadic_args)` — [`L1030`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1030)
- `_infer_fp8_bwd_output_sharding(mesh, arg_shapes, layout)` — [`L1681`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1681)
- `_infer_fp8_fwd_output_sharding(mesh, arg_shapes, is_training, layout)` — [`L1643`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1643)
- `_infer_fwd_output_sharding(mesh, arg_shapes, variadic_args, is_training, layout)` — [`L945`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L945)
- `_normalize_layout(layout: str)` — [`L103`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L103)
- `_shift_to_left(x, fill_value)` — [`L495`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L495)
- `check_compute_capability(capability)` — [`L410`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L410)
- `check_cudnn_version()` — [`L404`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L404)
- `check_eq(a, b, c, msg)` — [`L279`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L279)
- `check_fp8_params(params)` — [`L1264`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1264)
- `check_is_flash_attention(query, key, value, layout: int, cudnn_version, has_bias, is_training, is_packed=False, is_paged_attention=False, is_fp8=False)` — [`L354`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L354)
- `check_is_paged_attention(page_table_k)` — [`L123`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L123)
- `check_layout(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, page_table_k, page_table_v, layout)` — [`L277`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L277)
- `check_seqlen_offsets(tensor, name)` — [`L336`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L336)
- `combine_bias_and_mask(bias, mask, dtype)` — [`L1826`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1826)
- `convert_mask_type_to_string(mask_type: MaskType)` — [`L74`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L74)
- `create_dot_product_attention_backend_config(batch, num_heads, seq_q, seq_kv, dtype, fmha_scale, seed, dropout_rate, mask_type, layout, sliding_window_length, max_seg_per_batch, is_paged_attention, is_bwd)` — [`L219`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L219)
- `create_dot_product_attention_backend_config_base(batch, num_heads, seq_q, seq_kv, dtype, fmha_scale, mask_type, layout, is_bwd)` — [`L126`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L126)
- `create_dot_product_attention_fp8_backend_config(batch, num_heads, seq_q, seq_kv, dtype, fmha_scale, mask_type, layout, is_bwd)` — [`L248`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L248)
- `default_layouts(*shapes)` — [`L117`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L117)
- `dot_product_attention(query: Array, key: Array, value: Array, bias: Array | None = None, mask: Array | None = None, q_seqlen: Array | None = None, kv_seqlen: Array | None = None, q_offsets: Array | None = None, kv_offsets: Array | None = None, fp8_params: FP8Params | None = None, *, scale: float = 1, mask_type: MaskType = MaskType.NO_MASK, seed: int = 42, dropout_rate: float = 0, qkv_layout: str = "BTNH", sliding_window_length: int | None = None, use_fp8: bool = False, return_residual: bool = False)` — [`L1939`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1939) — Computes dot-product attention given query (Q), key (K), and value (V).
- `element_type_to_backend_config_type_mapping(dtype)` — [`L110`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L110)
- `get_custom_call_name(has_bias, has_dropout, is_bwd, is_fp8=False)` — [`L270`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L270)
- `get_large_negative_number(dtype)` — [`L94`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L94)
- `get_max_seg_per_batch(q_offsets)` — [`L120`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L120)
- `has_padding(mask_type: MaskType)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L88)
- `is_cuda_compute_capability_equal(capability)` — [`L418`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L418)
- `paged_attention(query: Array, key: Array, value: Array, q_seqlen: Array, kv_seqlen: Array, page_table_k: Array, page_table_v: Array, bias: Array | None = None, mask: Array | None = None, fp8_params: FP8Params | None = None, *, scale: float = 1, mask_type: MaskType = MaskType.NO_MASK, seed: int = 42, dropout_rate: float = 0, qkv_layout: str = "BTNH", sliding_window_length: int | None = None, use_fp8: bool = False, return_residual: bool = False)` — [`L1848`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1848) — Computes paged attention described in https://arxiv.org/pdf/2309.06180.
- `sharded_impl(*args)` — [`L1096`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1096)
- `should_export_dbias(bias_shape, query_shape, layout)` — [`L91`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L91)

## Module values
- `_custom_name_maps` — [`L255`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L255)
- `_dot_product_attention_bwd_lower` — [`L1063`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1063)
- `_dot_product_attention_bwd_p` — [`L1155`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1155)
- `_dot_product_attention_bwd_p_wrapper` — [`L1170`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1170)
- `_dot_product_attention_fp8_bwd_lower` — [`L1698`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1698)
- `_dot_product_attention_fp8_bwd_p` — [`L1749`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1749)
- `_dot_product_attention_fp8_bwd_p_wrapper` — [`L1764`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1764)
- `_dot_product_attention_fp8_fwd_lower` — [`L1656`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1656)
- `_dot_product_attention_fp8_fwd_p` — [`L1724`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1724)
- `_dot_product_attention_fp8_fwd_p_wrapper` — [`L1739`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1739)
- `_dot_product_attention_fwd_lower` — [`L996`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L996)
- `_dot_product_attention_fwd_p` — [`L1130`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1130)
- `_dot_product_attention_fwd_p_wrapper` — [`L1145`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1145)
- `check_is_flash_attention_fp8` — [`L1270`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1270)
- `fp8_params_keys` — [`L1247`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1247)
- `fp8_params_keys_bwd` — [`L1258`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1258)
- `fp8_params_keys_fwd` — [`L1255`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1255)
- `get_fp8_custom_call_name` — [`L273`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L273)
- `params_from_keys` — [`L1262`](../../../../../../../raw/code/jax/jax/_src/cudnn/fused_attention_stablehlo.py#L1262)

