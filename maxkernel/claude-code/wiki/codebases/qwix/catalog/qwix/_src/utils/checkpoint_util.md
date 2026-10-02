---
title: 'Module: qwix/_src/utils/checkpoint_util.py'
type: catalog
provenance: extracted
module: qwix/_src/utils/checkpoint_util.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.utils.checkpoint_util`/
symbols:
  process_prequantized_params: process_prequantized_params().
  _process_quantized_param: _process_quantized_param().
  _dequantize_quantized_param: _dequantize_quantized_param().
  restore_quantization_rules: restore_quantization_rules().
  _apply_sharding_and_dtype: _apply_sharding_and_dtype().
  _resolve_template_param: _resolve_template_param().
  _validate_prequantized_dict: _validate_prequantized_dict().
  _get_template_field: _get_template_field().
  _get_sharding: _get_sharding().
  _PREQUANTIZED_ARRAY_LEAF_NAMES: _PREQUANTIZED_ARRAY_LEAF_NAMES.
  _is_leaf: _is_leaf().
  _DEFAULT_ACT_QTYPE: _DEFAULT_ACT_QTYPE.
---
# Module: [`qwix/_src/utils/checkpoint_util.py`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py)

## Functions
- `_apply_sharding_and_dtype(checkpoint_value: Any, template_value: Any, path: tuple[str, ...], allow_broadcast: bool = False, use_checkpoint_sharding: bool = False)` — [`L126`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L126) — Converts a host/device array-like value into the template's array shape. — documented in [qwix-_src-utils-checkpoint_util](../../../../concepts/qwix-_src-utils-checkpoint_util.md)
- `_dequantize_quantized_param(checkpoint_param: Mapping[str, Any], template_param: Any, path: tuple[str, ...], *, use_checkpoint_sharding: bool)` — [`L244`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L244) — Dequantizes a prequantized parameter dictionary to a JAX array. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `_get_sharding(sharding: Any, path: tuple[str, ...])` — [`L95`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L95) — Resolves abstract mesh shardings into concrete device shardings. — documented in [qwix-_src-utils-checkpoint_util](../../../../concepts/qwix-_src-utils-checkpoint_util.md)
- `_get_template_field(obj: Any, field_name: str)` — [`L48`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L48) — Gets a field from the template parameter.
- `_is_leaf(path, x)` — [`L31`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L31) — Checks if x is a pre-quantized leaf.
- `_process_quantized_param(checkpoint_param: Mapping[str, Any], template_param: Any, path: tuple[str, ...], *, use_checkpoint_sharding: bool)` — [`L180`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L180) — Builds a QArray leaf from a quantized parameter dictionary and a quantized template. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `_resolve_template_param(path: tuple[str, ...], template_params: Any)` — [`L320`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L320) — Resolves the template parameter and its path. — documented in [qwix-_src-utils-checkpoint_util](../../../../concepts/qwix-_src-utils-checkpoint_util.md)
- `_validate_prequantized_dict(checkpoint_param: Any, path: tuple[str, ...])` — [`L55`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L55) — Validates the flat quantized parameter dictionary format. — documented in [qwix-_src-utils-checkpoint_util](../../../../concepts/qwix-_src-utils-checkpoint_util.md)
- `process_prequantized_params(checkpoint_params: Mapping[str, Any], template_params: Any, *, allow_extra_params: bool = False, use_checkpoint_sharding: bool = False)` — [`L351`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L351) — Converts external pre-quantized params into an `nnx.update`-friendly pure dict. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `restore_quantization_rules(checkpoint_params: Mapping[str, Any], rule_type: type[qconfig.QuantizationRule] | type[qt.QtRule], *, tile_size: int, act_qtype: jax.typing.DTypeLike | None | object = _DEFAULT_ACT_QTYPE, **kwargs)` — [`L472`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L472) — Restores quantization rules from pre-quantized checkpoint. — documented in [qwix-_src-utils-checkpoint_util](../../../../concepts/qwix-_src-utils-checkpoint_util.md)

## Module values
- `_DEFAULT_ACT_QTYPE` — [`L469`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L469)
- `_PREQUANTIZED_ARRAY_LEAF_NAMES` — [`L28`](../../../../../../../raw/code/qwix/qwix/_src/utils/checkpoint_util.py#L28)

