---
title: 'Module: python/sgl_jax/srt/utils/weight_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/utils/weight_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.utils.weight_utils`/
symbols:
  WeightLoader.load_weights_from_safetensors: WeightLoader#load_weights_from_safetensors().
  logger: logger.
  WeightLoader._split_qkv_weight: WeightLoader#_split_qkv_weight().
  WeightLoader._handle_single_weight: WeightLoader#_handle_single_weight().
  WeightLoader.mesh: WeightLoader#mesh.
  WeightLoader.num_kv_heads: WeightLoader#num_kv_heads.
  WeightLoader._load_dummy_weights: WeightLoader#_load_dummy_weights().
  WeightLoader.head_dim_original: WeightLoader#head_dim_original.
  WeightLoader._apply_kv_head_padding: WeightLoader#_apply_kv_head_padding().
  WeightLoader._create_stacked_moe_lazy_tensor: WeightLoader#_create_stacked_moe_lazy_tensor().
  WeightMapping.target_path: WeightMapping#target_path.
  WeightMapping.sharding: WeightMapping#sharding.
  WeightLoader.model_config: WeightLoader#model_config.
  WeightLoader.num_heads: WeightLoader#num_heads.
  WeightLoader.head_dim: WeightLoader#head_dim.
  WeightLoader.head_dim_pad: WeightLoader#head_dim_pad.
  WeightLoader._process_and_assign_weight: WeightLoader#_process_and_assign_weight().
  WeightLoader.hidden_size: WeightLoader#hidden_size.
  WeightLoader.sharding_size: WeightLoader#sharding_size.
  WeightLoader._load_stacked_slice: WeightLoader#_load_stacked_slice().
  WeightMapping: WeightMapping#
  WeightLoader._create_stacked_split_moe_lazy_tensor: WeightLoader#_create_stacked_split_moe_lazy_tensor().
  WeightLoader.create_bf16_linear: WeightLoader#create_bf16_linear().
  WeightLoader.dequant_fused_qkv: WeightLoader#dequant_fused_qkv().
  WeightLoader._scan_weight_info: WeightLoader#_scan_weight_info().
  WeightLoader._create_lazy_tensors: WeightLoader#_create_lazy_tensors().
  WeightLoader._create_split_lazy_tensor: WeightLoader#_create_split_lazy_tensor().
  WeightLoader._maybe_expand_linear_block_scale: WeightLoader#_maybe_expand_linear_block_scale().
  WeightLoader.dequant_fused_kv: WeightLoader#dequant_fused_kv().
  WeightLoader.load_and_fill_expert: WeightLoader#load_and_fill_expert().
  WeightMapping.head_dim_padding: WeightMapping#head_dim_padding.
  WeightLoader.dequant_fp8_linear: WeightLoader#dequant_fp8_linear().
  _reinterpret_dtype_if_needed: _reinterpret_dtype_if_needed().
  WeightMapping.transpose: WeightMapping#transpose.
  WeightLoader._get_param: WeightLoader#_get_param().
  SequentialSafetensorManager.get_handle: SequentialSafetensorManager#get_handle().
  WeightLoader.dequant_fp8_layers: WeightLoader#dequant_fp8_layers().
  WeightLoader.replicate_kv_heads: WeightLoader#replicate_kv_heads().
  WeightLoader.model: WeightLoader#model.
  WeightLoader._normalize_physical_to_logical_map: WeightLoader#_normalize_physical_to_logical_map().
  WeightLoader._maybe_convert_epmoe_scale_for_kernel: WeightLoader#_maybe_convert_epmoe_scale_for_kernel().
  WeightLoader._load_single_expert_slice: WeightLoader#_load_single_expert_slice().
  WeightLoader._is_excluded_layer_weight: WeightLoader#_is_excluded_layer_weight().
  WeightMapping.concat_axis: WeightMapping#concat_axis.
  SequentialSafetensorManager.handles: SequentialSafetensorManager#handles.
  WeightLoader._load_slice: WeightLoader#_load_slice().
  WeightLoader._smart_load_slice: WeightLoader#_smart_load_slice().
  WeightLoader._handle_split_weight: WeightLoader#_handle_split_weight().
  WeightMappingSpec: WeightMappingSpec.
  WeightMapping.__post_init__: WeightMapping#__post_init__().
  SequentialSafetensorManager: SequentialSafetensorManager#
  WeightLoader.v_head_dim: WeightLoader#v_head_dim.
  WeightLoader._shard_weight: WeightLoader#_shard_weight().
  WeightMapping.transpose_axes: WeightMapping#transpose_axes.
  _SAFETENSORS_DTYPE_TO_JAX._SAFETENSORS_DTYPE_TO_JAX: _SAFETENSORS_DTYPE_TO_JAX._SAFETENSORS_DTYPE_TO_JAX.
  WeightMapping._infer_default_sharding: WeightMapping#_infer_default_sharding().
  SequentialSafetensorManager.close_all: SequentialSafetensorManager#close_all().
  WeightLoader.moe_abstract_mesh: WeightLoader#moe_abstract_mesh.
  WeightLoader._block_dequant: WeightLoader#_block_dequant().
  WeightLoader._infer_qkv_shards: WeightLoader#_infer_qkv_shards().
  WeightLoader._make_load_slice: WeightLoader#_make_load_slice().
  WeightLoader._build_and_put: WeightLoader#_build_and_put().
  WeightMapping.reshape: WeightMapping#reshape.
  WeightMapping.repeat: WeightMapping#repeat.
  WeightMapping.kv_head_padding: WeightMapping#kv_head_padding.
  WeightLoader._weight_info_cache: WeightLoader#_weight_info_cache.
  SequentialSafetensorManager.__exit__: SequentialSafetensorManager#__exit__().
  WeightLoader.__init__: WeightLoader#__init__().
  WeightLoader.is_static_quant: WeightLoader#is_static_quant().
  WeightLoader.is_quant_ignored: WeightLoader#is_quant_ignored().
  WeightLoader.has_weight_on_disk: WeightLoader#has_weight_on_disk().
  WeightLoader._uniform_block_dequant: WeightLoader#_uniform_block_dequant().
  WeightLoader._bulk_read_file: WeightLoader#_bulk_read_file().
  WeightMapping.physical_to_logical_map: WeightMapping#physical_to_logical_map.
  WeightLoader.dummy_mode: WeightLoader#dummy_mode.
  WeightLoader.fn: WeightLoader#fn().
  WeightLoader._prewarm: WeightLoader#_prewarm().
  WeightLoader.make_shard: WeightLoader#make_shard().
  WeightLoader.make_expert_shard: WeightLoader#make_expert_shard().
  WeightMapping.is_eagle3: WeightMapping#is_eagle3.
  SequentialSafetensorManager.__init__: SequentialSafetensorManager#__init__().
  SequentialSafetensorManager.__enter__: SequentialSafetensorManager#__enter__().
  WeightLoader: WeightLoader#
  WeightLoader.dtype: WeightLoader#dtype.
---
# Module: [`python/sgl_jax/srt/utils/weight_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py)

## Classes
### `SequentialSafetensorManager`
- def: [`python/sgl_jax/srt/utils/weight_utils.py:118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L118)
- doc: Manages open file handles during a weight loading session to prevent
- signature: `class SequentialSafetensorManager:`
- members:
  - `close_all(self)` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L134)
  - `get_handle(self, filename)` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L127)
  - `handles` — [`L125`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L125)
- protocol/private: `__enter__`[`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L139), `__exit__`[`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L142), `__init__`[`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L124)
- used by: [`load_weights_from_safetensors`](weight_utils.md#WeightLoader.load_weights_from_safetensors), [`_create_stacked_moe_lazy_tensor`](weight_utils.md#WeightLoader._create_stacked_moe_lazy_tensor), [`_load_stacked_slice`](weight_utils.md#WeightLoader._load_stacked_slice), [`_create_stacked_split_moe_lazy_tensor`](weight_utils.md#WeightLoader._create_stacked_split_moe_lazy_tensor), [`_create_lazy_tensors`](weight_utils.md#WeightLoader._create_lazy_tensors), [`_create_split_lazy_tensor`](weight_utils.md#WeightLoader._create_split_lazy_tensor), [`load_and_fill_expert`](weight_utils.md#WeightLoader.load_and_fill_expert), [`_load_single_expert_slice`](weight_utils.md#WeightLoader._load_single_expert_slice), [`_load_slice`](weight_utils.md#WeightLoader._load_slice), [`_smart_load_slice`](weight_utils.md#WeightLoader._smart_load_slice)

### `WeightLoader`
- def: [`python/sgl_jax/srt/utils/weight_utils.py:146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L146)
- signature: `class WeightLoader:`
- members:
  - `_apply_kv_head_padding(self, weight: jax.Array, hf_key: str)` — [`L2840`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2840) — Apply KV head padding/replication when tp_size > total_kv_heads. — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `_block_dequant(weight_q: jax.Array, weight_scale: jax.Array, head_dim: int | None = None)` — [`L274`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L274) — Block-dequantize weight_q using 3D scale [in_blocks, 1, out_dim].
  - `_create_lazy_tensors(self, hf_key: str, infos: list[dict], file_manager: SequentialSafetensorManager, target_sharding: jax.sharding.NamedSharding | None = None)` — [`L1107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1107) — Create a list of JAX arrays that lazy load data from safetensors via callback.
  - `_create_split_lazy_tensor(self, hf_key: str, infos: list[dict], file_manager: SequentialSafetensorManager, concat_axis: int, target_sharding: jax.sharding.NamedSharding | None = None)` — [`L1150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1150) — Lazy loader for TP-Split weights (e.g., Grok Attention/MLP).
  - `_create_stacked_split_moe_lazy_tensor(self, expected_hf_keys: list[str], weight_infos: dict[str, list[dict]], file_manager: SequentialSafetensorManager, concat_axis: int, do_transpose: bool = False, target_sharding: jax.sharding.NamedSharding | None = None, physical_to_logical_map: np.ndarray | None = None)` — [`L1240`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1240) — Lazy loader for TP-Split MOE weights (e.g., Grok MOE).
  - `_infer_qkv_shards(total_out_dim: int, total_scale_blocks: int, num_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int, block_size: int)` — [`L720`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L720) — Infer the number of TP shards used during FP8 quantization.
  - `_maybe_convert_epmoe_scale_for_kernel(self, weight: jax.Array, model_param: nnx.Variable, target_path: str)` — [`L820`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L820) — Convert offline EPMoE/FusedEPMoE scales into kernel-ready 4D layout.
  - `_maybe_expand_linear_block_scale(self, weight: jax.Array, model_param: nnx.Variable, target_path: str)` — [`L962`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L962) — Expand 2D block-quant scale [out_blocks, in_blocks] to 3D [in_blocks, 1, n_out] at load time.
  - `_scan_weight_info(self)` — [`L1016`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1016) — Scan all safetensors files to build a mapping from HF key to file info.
  - `_uniform_block_dequant(weight, scale, block_size)` — [`L439`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L439) — Uniform block dequant for weight[out_dim, in_dim] * scale[out_blocks, in_blocks].
  - `create_bf16_linear(weight: jax.Array, kernel_axes, mesh, use_bias=False, bias=None)` — [`L219`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L219) — Create a bf16 LinearBase from a weight array [in, out].
  - `dequant_fp8_layers(self, layers: list, specs: list[tuple[str, int | None]], *, layer_filter=None)` — [`L343`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L343) — Dequantize specified QuantizedLinear projections → bf16 LinearBase.
  - `dequant_fp8_linear(self, ql, head_dim: int | None = None)` — [`L240`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L240) — Dequantize a single QuantizedLinear → bf16 LinearBase.
  - `dequant_fused_kv(self, kv_buffers: dict[int, dict], layers: list, config)` — [`L452`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L452) — Dequantize FP8 K+V weights with per-layer quantization scheme detection.
  - `dequant_fused_qkv(self, fused_qkv_buffers: dict[int, dict], layers: list, config)` — [`L553`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L553) — Dequantize per-shard-interleaved fused QKV FP8 weights.
  - `fn(idx, m=merged)` — [`L693`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L693)
  - `has_weight_on_disk(self, hf_key: str)` — [`L210`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L210) — Return whether a concrete HF weight key exists in the safetensors files.
  - `is_quant_ignored(self, hf_path: str)` — [`L202`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L202) — Check if a HuggingFace weight path is in the quantization ignored_layers list.
  - `is_static_quant(self)` — [`L197`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L197) — Check if the model uses a static FP8 checkpoint.
  - `load_and_fill_expert(log_idx)` — [`L1367`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1367)
  - `load_weights_from_safetensors(self, weight_mappings: Mapping[str, WeightMappingSpec], safetensors_partition=1, dummy=False)` — [`L1839`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1839) — Load weights using JAX lazy evaluation and parallel I/O. — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `make_expert_shard(indices, weight_shape=expert_weight_shape, weight_dtype=dtype, idx=expert_idx)` — [`L2396`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2396)
  - `make_shard(indices, shape=shape, dtype=dtype)` — [`L2328`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2328)
  - `replicate_kv_heads(self, layers: list, specs: list[tuple[str, int]], target_kv_heads_fn)` — [`L376`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L376) — Replicate KV heads for TP alignment.
  - `dtype` — [`L157`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L157)
  - `dummy_mode` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L158)
  - `head_dim` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L174) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `head_dim_original` — [`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L171) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `head_dim_pad` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L173) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `hidden_size` — [`L164`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L164)
  - `mesh` — [`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L156) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `model` — [`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L154)
  - `model_config` — [`L155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L155) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `moe_abstract_mesh` — [`L186`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L186)
  - `num_heads` — [`L161`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L161) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `num_kv_heads` — [`L163`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L163) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `sharding_size` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L177)
  - `v_head_dim` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L175)
- protocol/private: `__init__`[`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L147), `_build_and_put`[`L1741`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1741), `_bulk_read_file`[`L1691`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1691), `_create_stacked_moe_lazy_tensor`[`L1387`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1387), `_get_param`[`L2823`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2823), `_handle_single_weight`[`L2478`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2478), `_handle_split_weight`[`L2570`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2570), `_is_excluded_layer_weight`[`L2948`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2948), `_load_dummy_weights`[`L2287`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2287), `_load_single_expert_slice`[`L1298`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1298), `_load_slice`[`L1135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1135), `_load_stacked_slice`[`L1328`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1328), `_make_load_slice`[`L1134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1134), `_normalize_physical_to_logical_map`[`L781`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L781), `_prewarm`[`L1796`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1796), `_process_and_assign_weight`[`L2458`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2458), `_shard_weight`[`L2811`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2811), `_smart_load_slice`[`L1195`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L1195), `_split_qkv_weight`[`L2575`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L2575), `_weight_info_cache`[`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L159)
- uses (calls/refs, reference-scoped): [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`logger`](weight_utils.md#logger), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`hf_text_config`](../configs/model_config.md#ModelConfig.hf_text_config), [`target_path`](weight_utils.md#WeightMapping.target_path), [`sharding`](weight_utils.md#WeightMapping.sharding), [`LinearBase`](../layers/linear.md#LinearBase), [`model_path`](../configs/model_config.md#ModelConfig.model_path), [`get_total_num_kv_heads`](../configs/model_config.md#ModelConfig.get_total_num_kv_heads), [`num_hidden_layers`](../configs/model_config.md#ModelConfig.num_hidden_layers), [`bias`](../layers/linear.md#LinearBase.bias), [`WeightMapping`](weight_utils.md#WeightMapping), [`num_attention_heads`](../configs/model_config.md#ModelConfig.num_attention_heads), [`head_dim_padding`](weight_utils.md#WeightMapping.head_dim_padding), [`weight`](../layers/linear.md#LinearBase.weight), [`_reinterpret_dtype_if_needed`](weight_utils.md#_reinterpret_dtype_if_needed), [`transpose`](weight_utils.md#WeightMapping.transpose), [`get_handle`](weight_utils.md#SequentialSafetensorManager.get_handle), [`expand_block_scale`](../kernels/quantized_matmul/blockwise_utils.md#expand_block_scale), [`needs_kv_head_replication`](../configs/model_config.md#ModelConfig.needs_kv_head_replication), [`concat_axis`](weight_utils.md#WeightMapping.concat_axis), [`handles`](weight_utils.md#SequentialSafetensorManager.handles), [`WeightMappingSpec`](weight_utils.md#WeightMappingSpec), [`QuantizedLinear`](../layers/linear.md#QuantizedLinear), [`SequentialSafetensorManager`](weight_utils.md#SequentialSafetensorManager), [`get_kv_padding_strategy`](../configs/model_config.md#ModelConfig.get_kv_padding_strategy), [`get_num_kv_head_replicas`](../configs/model_config.md#ModelConfig.get_num_kv_head_replicas), [`hidden_size`](../configs/model_config.md#ModelConfig.hidden_size), [`_SAFETENSORS_DTYPE_TO_JAX`](weight_utils.md#_SAFETENSORS_DTYPE_TO_JAX._SAFETENSORS_DTYPE_TO_JAX), [`ep_size`](../configs/model_config.md#ModelConfig.ep_size), [`transpose_axes`](weight_utils.md#WeightMapping.transpose_axes), [`get_swa_weight_params`](../configs/model_config.md#ModelConfig.get_swa_weight_params), [`kv_head_padding`](weight_utils.md#WeightMapping.kv_head_padding), [`repeat`](weight_utils.md#WeightMapping.repeat), [`reshape`](weight_utils.md#WeightMapping.reshape)

### `WeightMapping`
- def: [`python/sgl_jax/srt/utils/weight_utils.py:67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L67)
- signature: `class WeightMapping:`
- members:
  - `concat_axis` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L78)
  - `head_dim_padding` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L76)
  - `is_eagle3` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L79)
  - `kv_head_padding` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L77)
  - `physical_to_logical_map` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L80)
  - `repeat` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L75)
  - `reshape` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L74)
  - `sharding` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L69) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `target_path` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L68) — documented in [python-sgl_jax-srt-utils-weight_utils](../../../../../concepts/python-sgl_jax-srt-utils-weight_utils.md)
  - `transpose` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L70)
  - `transpose_axes` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L71)
- protocol/private: `__post_init__`[`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L82), `_infer_default_sharding`[`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L86)
- used by: [`load_weights_from_safetensors`](weight_utils.md#WeightLoader.load_weights_from_safetensors), [`_split_qkv_weight`](weight_utils.md#WeightLoader._split_qkv_weight), [`_handle_single_weight`](weight_utils.md#WeightLoader._handle_single_weight), [`_load_dummy_weights`](weight_utils.md#WeightLoader._load_dummy_weights), [`_process_and_assign_weight`](weight_utils.md#WeightLoader._process_and_assign_weight), [`create_moe_weights_mapping`](../layers/moe.md#create_moe_weights_mapping), [`_handle_split_weight`](weight_utils.md#WeightLoader._handle_split_weight), [`WeightMappingSpec`](weight_utils.md#WeightMappingSpec)

## Functions
- `_reinterpret_dtype_if_needed(data: np.ndarray, target_dtype: jnp.dtype)` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L55)

## Module values
- `WeightMappingSpec` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L115)
- `_SAFETENSORS_DTYPE_TO_JAX` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L43)
- `logger` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/weight_utils.py#L34)

