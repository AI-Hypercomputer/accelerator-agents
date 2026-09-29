---
title: 'Module: qwix/_src/providers/lora.py'
type: catalog
provenance: extracted
module: qwix/_src/providers/lora.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.providers.lora`/
symbols:
  LoraProvider.dot_general: LoraProvider#dot_general().
  LoraProvider.einsum: LoraProvider#einsum().
  _get_or_create_lora_params: _get_or_create_lora_params().
  LoraProvider.conv_general_dilated: LoraProvider#conv_general_dilated().
  apply_lora_to_model: apply_lora_to_model().
  _parse_einsum_str_for_lora: _parse_einsum_str_for_lora().
  LoraProvider: LoraProvider#
  LoraRule: LoraRule#
  LoraRule.rank: LoraRule#rank.
  LoraRule.dropout: LoraRule#dropout.
  LoraRule.alpha: LoraRule#alpha.
  _get_or_create_lora_params.init_with_sharding: _get_or_create_lora_params().init_with_sharding().
  LoraProvider.__init__: LoraProvider#__init__().
  _get_or_create_lora_params.get_canonical_pspec: _get_or_create_lora_params().get_canonical_pspec().
  LoraRule.lora_b_initializer: LoraRule#lora_b_initializer.
  LoraRule.lora_a_initializer: LoraRule#lora_a_initializer.
  _find_lora_dim_char: _find_lora_dim_char().
  _create_lora_layer_shapes: _create_lora_layer_shapes().
  _compute_lora_delta: _compute_lora_delta().
---
# Module: [`qwix/_src/providers/lora.py`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py)

## Classes
### `LoraProvider`  ·  implements/extends PtqProvider
- def: [`qwix/_src/providers/lora.py:208`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L208)
- doc: Provider for (Q)LoRA.
- signature: `class LoraProvider(ptq.PtqProvider):`
- members:
  - `__init__(self, rules=None, *, disable_jit: bool = False, **kwargs)` — [`L215`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L215) — Initializes the LoraProvider.
  - `conv_general_dilated(self, lhs: jax.Array, rhs: jax.Array | ptq.WithAux[qarray.QArray], window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L373`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L373) — LoRA conv_general_dilated. — documented in [qwix-_src-providers-lora](../../../../concepts/qwix-_src-providers-lora.md)
  - `dot_general(self, lhs: jax.Array, rhs: jax.Array | ptq.WithAux[qarray.QArray], dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding: jax.sharding.NamedSharding | None = None)` — [`L238`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L238) — LoRA dot_general. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
  - `einsum(self, einsum_str: str, *operands: jax.Array | ptq.WithAux[qarray.QArray], **kwargs)` — [`L319`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L319) — LoRA einsum. — documented in [qwix-_src-providers-lora](../../../../concepts/qwix-_src-providers-lora.md)
- uses (calls/refs, reference-scoped): [`QArray`](../core/qarray.md#QArray), [`PtqProvider`](ptq.md#PtqProvider), [`WithAux`](ptq.md#WithAux), [`dot_general`](ptq.md#PtqProvider.dot_general), [`_get_current_rule_and_op_id`](../qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`einsum`](ptq.md#PtqProvider.einsum), [`conv_general_dilated`](ptq.md#PtqProvider.conv_general_dilated), [`_get_or_create_lora_params`](lora.md#_get_or_create_lora_params), [`find_param`](../utils/flax_util.md#find_param), [`_parse_einsum_str_for_lora`](lora.md#_parse_einsum_str_for_lora), [`get_current_module`](../utils/flax_util.md#get_current_module), [`__init__`](ptq.md#PtqProvider.__init__), [`LoraRule`](lora.md#LoraRule), [`rank`](lora.md#LoraRule.rank), [`make_rng`](../utils/flax_util.md#make_rng), [`dropout`](lora.md#LoraRule.dropout), [`alpha`](lora.md#LoraRule.alpha), [`_compute_lora_delta`](lora.md#_compute_lora_delta), [`_create_lora_layer_shapes`](lora.md#_create_lora_layer_shapes)
- used by: [`PtqProvider`](ptq.md#PtqProvider), [`dot_general`](ptq.md#PtqProvider.dot_general), [`einsum`](ptq.md#PtqProvider.einsum), [`conv_general_dilated`](ptq.md#PtqProvider.conv_general_dilated)  (11 test-only)

### `LoraRule`  ·  implements/extends QuantizationRule
- def: [`qwix/_src/providers/lora.py:61`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L61) — documented in [qwix-_src-providers-lora](../../../../concepts/qwix-_src-providers-lora.md)
- doc: LoRA rules that match and configure the LoRA behavior.
- signature: `class LoraRule(qconfig.QuantizationRule):`
- members:
  - `alpha` — [`L73`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L73)
  - `dropout` — [`L76`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L76)
  - `lora_a_initializer` — [`L79`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L79)
  - `lora_b_initializer` — [`L82`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L82) — ---
  - `rank` — [`L69`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L69) — documented in [qwix-_src-providers-lora](../../../../concepts/qwix-_src-providers-lora.md)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../qconfig.md#QuantizationRule)
- used by: [`QuantizationRule`](../qconfig.md#QuantizationRule), [`dot_general`](lora.md#LoraProvider.dot_general), [`einsum`](lora.md#LoraProvider.einsum), [`_get_or_create_lora_params`](lora.md#_get_or_create_lora_params), [`conv_general_dilated`](lora.md#LoraProvider.conv_general_dilated), [`__init__`](lora.md#LoraProvider.__init__)  (4 test-only)

## Functions
- `_compute_lora_delta(lhs: jax.Array, lora_a: jax.Array, lora_b: jax.Array, lhs_ca: Sequence[int], lhs_ba: Sequence[int], contract_shape: typing.Shape, batch_shape: typing.Shape, remain_shape: typing.Shape, rank: int, precision: jax.lax.PrecisionLike = None)` — [`L164`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L164) — Computes the raw LoRA delta.
- `_create_lora_layer_shapes(rhs_ca: Sequence[int], rhs_ba: Sequence[int], rhs_ra: Sequence[int], contract_shape: typing.Shape, batch_shape: typing.Shape, remain_shape: typing.Shape, rank: int)` — [`L135`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L135) — Returns lora param shapes and sharding transposes for dot_general.
- `_find_lora_dim_char(all_dims: set[str])` — [`L85`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L85)
- `_get_or_create_lora_params(*, name: str, rule: LoraRule, a_shape: typing.Shape, b_shape: typing.Shape, a_sharding_transpose: Sequence[int | None], b_sharding_transpose: Sequence[int | None])` — [`L452`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L452) — Get or create LoRA params. — documented in [qwix-_src-providers-lora](../../../../concepts/qwix-_src-providers-lora.md)
- `_parse_einsum_str_for_lora(lhs_shape: typing.Shape, rhs_shape: typing.Shape, einsum_str: str, lora_rank: int)` — [`L91`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L91) — Returns lora param shapes and einsum string for LoRA. — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
- `apply_lora_to_model(model: qwix_model.ModelType, provider: qconfig.QuantizationProvider, *model_inputs: Any, methods: Collection[str] = ('__call__',), **model_inputs_kwargs: Any)` — [`L34`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L34) — Applies LoRA to a model.
- `get_canonical_pspec(x)` — [`L488`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L488) — Returns the canonical sharding.spec if x contains a concrete array.
- `init_with_sharding(initializer, rng, shape, transpose)` — [`L508`](../../../../../../../raw/code/qwix/qwix/_src/providers/lora.py#L508)

