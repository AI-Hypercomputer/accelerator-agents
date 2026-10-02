---
title: 'Module: src/maxtext/utils/layerwise_quantization.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/layerwise_quantization.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.layerwise_quantization`/
symbols:
  LayerwiseQuantization.load_and_quantize: LayerwiseQuantization#load_and_quantize().
  LayerwiseQuantization._load_and_quantize_nnx: LayerwiseQuantization#_load_and_quantize_nnx().
  LayerwiseQuantization.unboxed_abstract_state: LayerwiseQuantization#unboxed_abstract_state.
  LayerwiseQuantization._load_layer: LayerwiseQuantization#_load_layer().
  LayerwiseQuantization.config: LayerwiseQuantization#config.
  DictKey: DictKey.
  get_quantized_param_paths: get_quantized_param_paths().
  main: main().
  LayerwiseQuantization._mesh: LayerwiseQuantization#_mesh.
  LayerwiseQuantization._strip_kernels_at_quantized_paths: LayerwiseQuantization#_strip_kernels_at_quantized_paths().
  remove_quantized_params: remove_quantized_params().
  LayerwiseQuantization._map_fn: LayerwiseQuantization#_map_fn().
  LayerwiseQuantization.rng: LayerwiseQuantization#rng.
  LayerwiseQuantization._wrap_value: LayerwiseQuantization#_wrap_value().
  LayerwiseQuantization.walk: LayerwiseQuantization#walk().
  LayerwiseQuantization.quant: LayerwiseQuantization#quant.
  get_original_path_key: get_original_path_key().
  LayerwiseQuantization.model_apply: LayerwiseQuantization#model_apply().
  LayerwiseQuantization._copy_kernel_leaves_: LayerwiseQuantization#_copy_kernel_leaves_().
  LayerwiseQuantization._create_partial_abstract_params: LayerwiseQuantization#_create_partial_abstract_params().
  LayerwiseQuantization.__init__: LayerwiseQuantization#__init__().
  LayerwiseQuantization: LayerwiseQuantization#
  IGNORE: IGNORE.
  PRNGKeyType: PRNGKeyType.
  get_quantized_param_paths.is_qtensor: get_quantized_param_paths().is_qtensor().
  remove_quantized_params._map_fn: remove_quantized_params()._map_fn().
  insert_deepseekmoeblock_scope: insert_deepseekmoeblock_scope().
  LayerwiseQuantization._should_keep: LayerwiseQuantization#_should_keep().
  validate_config: validate_config().
---
# Module: [`src/maxtext/utils/layerwise_quantization.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py)

## Classes
### `LayerwiseQuantization`
- def: [`src/maxtext/utils/layerwise_quantization.py:160`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L160)
- doc: Layerwise quantization for large models.
- signature: `class LayerwiseQuantization:`
- members:
  - `_copy_kernel_leaves_(dst_model, src_model)` — [`L369`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L369) — Copy the full-precision parameter leaves (kernel/embedding/scale/bias)
  - `_create_partial_abstract_params(self, abstract_unboxed_params, layer)` — [`L428`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L428) — Creates a partial abstract params structure using ocp.PLACEHOLDER.
  - `_load_and_quantize_nnx(self)` — [`L287`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L287) — Whole-model NNX convert: load full-precision via TRAIN-mode `from_pretrained`,
  - `_load_layer(self, layer_name)` — [`L413`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L413) — Loads a specific layer's parameters from the checkpoint.
  - `_strip_kernels_at_quantized_paths(state_dict)` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L394) — Drop `kernel` keys at any node that has a sibling `AqtDotGeneral_0`.
  - `load_and_quantize(self)` — [`L195`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L195) — Load parameters layer by layer and quantize them.
  - `model_apply(_p, _rng, layer)` — [`L228`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L228)
  - `walk(d_node, s_node)` — [`L377`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L377)
  - `config` — [`L166`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L166)
  - `quant` — [`L182`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L182)
  - `rng` — [`L167`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L167)
  - `unboxed_abstract_state` — [`L186`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L186)
- protocol/private: `__init__`[`L165`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L165), `_map_fn`[`L435`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L435), `_mesh`[`L180`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L180), `_should_keep`[`L431`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L431), `_wrap_value`[`L353`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L353)
- uses (calls/refs, reference-scoped): [`remove_quantized_params`](layerwise_quantization.md#remove_quantized_params), [`IGNORE`](layerwise_quantization.md#IGNORE), [`PRNGKeyType`](layerwise_quantization.md#PRNGKeyType), [`insert_deepseekmoeblock_scope`](layerwise_quantization.md#insert_deepseekmoeblock_scope)
- used by: [`main`](layerwise_quantization.md#main)

## Functions
- `_map_fn(path, value)` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L126)
- `get_original_path_key(aqt_k_tuple: tuple[DictKey, ...])` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L61) — Maps an AQT PyTree path (tuple of keys) to its corresponding original parameter path.
- `get_quantized_param_paths(aqt_params: Any, params: Any)` — [`L91`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L91) — Identifies the set of paths in the original params tree that have been quantized.
- `insert_deepseekmoeblock_scope(aqt_layer_tree: dict[str, Any])` — [`L133`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L133) — Moves top-level AqtEinsum_* entries into the existing 'DeepSeekMoeBlock_0'
- `is_qtensor(x)` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L96)
- `main(argv: Sequence[str])` — [`L452`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L452)
- `remove_quantized_params(params: Any, aqt_vars: Any)` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L120) — Replaces the values in the original params tree that are now quantized with empty dicts.
- `validate_config(config)` — [`L463`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L463)

## Module values
- `DictKey` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L58)
- `IGNORE` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L56)
- `PRNGKeyType` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/utils/layerwise_quantization.py#L57)

