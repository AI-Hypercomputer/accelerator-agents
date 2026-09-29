---
title: 'Module: tpu_inference/models/jax/utils/qwix/qwix_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/utils/qwix/qwix_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.utils.qwix.qwix_utils`/
symbols:
  qwix_quantize_nnx_model: qwix_quantize_nnx_model().
  apply_qwix_quantization: apply_qwix_quantization().
  logger: logger.
  update_vllm_config_for_qwix_quantization: update_vllm_config_for_qwix_quantization().
  load_random_weights_into_qwix_abstract_model: load_random_weights_into_qwix_abstract_model().
  get_default_qwix_quantization_config: get_default_qwix_quantization_config().
  get_random_sharded_array: get_random_sharded_array().
  quantization_config_file_path_to_dict: quantization_config_file_path_to_dict().
  DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS: DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS.
  apply_qwix_on_abstract_model: apply_qwix_on_abstract_model().
  QUANTIZATION_CONFIG_PATH: QUANTIZATION_CONFIG_PATH.
  DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS: DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS.
  parse_qwix_config_to_rules: parse_qwix_config_to_rules().
  get_random_sharded_array.get_slice: get_random_sharded_array().get_slice().
  manually_quantize_qwix_weight: manually_quantize_qwix_weight().
  DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE: DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE.
  DEFAULT_MAX_NUM_BLOCKS_PER_REQ: DEFAULT_MAX_NUM_BLOCKS_PER_REQ.
  DEFAULT_LLAMA4_FP8_CONFIG: DEFAULT_LLAMA4_FP8_CONFIG.
  DEFAULT_GPT_OSS_FP4_CONFIG: DEFAULT_GPT_OSS_FP4_CONFIG.
  apply_qwix_quantization.create_and_quantize_model_factory: apply_qwix_quantization().create_and_quantize_model_factory().
  manually_quantize_qwix_activation: manually_quantize_qwix_activation().
  get_quant_dtype_from_qwix_config: get_quant_dtype_from_qwix_config().
---
# Module: [`tpu_inference/models/jax/utils/qwix/qwix_utils.py`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py)

## Functions
- `apply_qwix_on_abstract_model(vllm_config: VllmConfig)` — [`L433`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L433) — Determines whether to apply Qwix quantization on the abstract model
- `apply_qwix_quantization(vllm_config: VllmConfig, model_or_model_fn: Callable | nnx.Module, rng: jax.Array, mesh: Mesh, apply_to_abstract_model: bool)` — [`L256`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L256) — Will apply quantization if a valid quantization config with Qwix rules is provided.  See README
- `create_and_quantize_model_factory()` — [`L423`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L423) — Helper function to create and quantize the abstract model.
- `get_default_qwix_quantization_config(hf_config: dict, skip_quantization: bool)` — [`L447`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L447) — Some models are pre-quantized and in those cases, we want to return a default set of
- `get_quant_dtype_from_qwix_config(vllm_config: VllmConfig)` — [`L681`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L681) — Gets the quantization dtype from the Qwix config.
- `get_random_sharded_array(key: PRNGKey, mesh: Mesh, param: nnx.Param, param_shape: tuple, dtype: jnp.dtype, param_name: str)` — [`L526`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L526) — Returns a random sharded array for the given parameter for the given shape.
- `get_slice(index)` — [`L559`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L559)
- `load_random_weights_into_qwix_abstract_model(rng: PRNGKey, model: nnx.Module, mesh: Mesh, quantization_config: dict)` — [`L578`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L578) — Loads random weights for an abstract, Qwix-quantized model.
- `manually_quantize_qwix_activation(inputs: jax.Array, rule_name: str, qtype: jnp.dtype, channelwise_axes: List[int], tiled_axes: dict, calibration_method: str)` — [`L647`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L647) — Manually quantizes an activation tensor using Qwix.  Needed for the SparseMatmul
- `manually_quantize_qwix_weight(name: str, weight: jax.Array, qtype: jnp.dtype, channelwise_axes: List[int], tiled_axes: dict, calibration_method: str)` — [`L628`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L628) — Manually quantizes a weight tensor using Qwix.  Only needed for the SparseMatmul DeepSeek case right now, since
- `parse_qwix_config_to_rules(qwix_config: List[dict])` — [`L79`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L79) — Parse a list of dictionaries containing Qwix quantization rules into a list of QuantizationRule objects.
- `quantization_config_file_path_to_dict(quantization_config_file_path: str)` — [`L220`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L220) — Converts a quantization config YAML file path to a dictionary.
- `qwix_quantize_nnx_model(model: nnx.Module, qwix_config: List[dict], rng: jax.Array, mesh: Mesh, num_hidden_layers: int, kv_cache_block_size: int, kv_cache_num_kv_heads: int | tuple[int, ...], kv_cache_head_size: int | tuple[int, ...], kv_cache_dtype: str)` — [`L97`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L97) — Quantizes a Flax NNX model using Qwix.
- `update_vllm_config_for_qwix_quantization(vllm_config: VllmConfig)` — [`L479`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L479) — Updates the vLLM config to unpack the Qwix quantization config if it exists.

## Module values
- `DEFAULT_GPT_OSS_FP4_CONFIG` — [`L61`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L61)
- `DEFAULT_LLAMA4_FP8_CONFIG` — [`L38`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L38)
- `DEFAULT_MAX_NUM_BLOCKS_PER_REQ` — [`L36`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L36)
- `DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS` — [`L35`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L35)
- `DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE` — [`L33`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L33)
- `DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS` — [`L34`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L34)
- `QUANTIZATION_CONFIG_PATH` — [`L32`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L32)
- `logger` — [`L30`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/qwix/qwix_utils.py#L30)

