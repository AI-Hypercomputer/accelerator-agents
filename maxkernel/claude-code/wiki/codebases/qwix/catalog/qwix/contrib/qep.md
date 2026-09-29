---
title: 'Module: qwix/contrib/qep.py'
type: catalog
provenance: extracted
module: qwix/contrib/qep.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.qep`/
symbols:
  quantize: quantize().
  _quantize_weight: _quantize_weight().
  _CaptureProvider._collect_stats: _CaptureProvider#_collect_stats().
  quantize.replay_and_collect_stats: quantize().replay_and_collect_stats().
  _convert_internal_stage_to_public_metadata: _convert_internal_stage_to_public_metadata().
  QepRule: QepRule#
  quantize_params._quantize: quantize_params()._quantize().
  _group_discovered_ops_into_stages: _group_discovered_ops_into_stages().
  quantize.apply_quantization: quantize().apply_quantization().
  _update_flat_stats_with_moving_average: _update_flat_stats_with_moving_average().
  _StageSpec.members: _StageSpec#members.
  _CaptureProvider.prepare_for_discovery: _CaptureProvider#prepare_for_discovery().
  _create_ptq_model_and_abstract_quantized_params: _create_ptq_model_and_abstract_quantized_params().
  _CaptureProvider.prepare_for_capture: _CaptureProvider#prepare_for_capture().
  quantize_params: quantize_params().
  QepResult.stages: QepResult#stages.
  _MatchedOp.path: _MatchedOp#path.
  _CaptureProvider.discovered_ops: _CaptureProvider#discovered_ops().
  _flatten_params_tree_to_tuple_paths: _flatten_params_tree_to_tuple_paths().
  _CaptureProvider._discovered_ops: _CaptureProvider#_discovered_ops.
  QepResult.params: QepResult#params.
  _MatchedOp: _MatchedOp#
  _CaptureProvider.__init__: _CaptureProvider#__init__().
  _CaptureProvider.get_rule_type: _CaptureProvider#get_rule_type().
  _MatchedOp.rule: _MatchedOp#rule.
  _CaptureProvider: _CaptureProvider#
  QepResult.model: QepResult#model.
  _MatchedOp.op_key: _MatchedOp#op_key.
  _StageSpec: _StageSpec#
  _unflatten_tuple_paths_to_params_tree: _unflatten_tuple_paths_to_params_tree().
  _CaptureProvider._capture_keys: _CaptureProvider#_capture_keys.
  _CaptureProvider._captures: _CaptureProvider#_captures.
  QepStage: QepStage#
  QepStage.module_paths: QepStage#module_paths.
  _run_model_forward_with_injected_params: _run_model_forward_with_injected_params().
  QepRule.correction_factor: QepRule#correction_factor.
  QepRule.damping_factor: QepRule#damping_factor.
  QepRule.apply_correction: QepRule#apply_correction.
  QepResult: QepResult#
  _MatchedOp.lhs_id: _MatchedOp#lhs_id.
  _StageSpec.index: _StageSpec#index.
  _unfreeze_params_tree: _unfreeze_params_tree().
  _append_qep_suffix_to_path: _append_qep_suffix_to_path().
  QepStage.index: QepStage#index.
  QepStage.param_paths: QepStage#param_paths.
  QepResult.quant_stats: QepResult#quant_stats.
  _CaptureProvider.get_stats_suffix: _CaptureProvider#get_stats_suffix().
---
# Module: [`qwix/contrib/qep.py`](../../../../../../raw/code/qwix/qwix/contrib/qep.py)

## Classes
### `QepResult`
- def: [`qwix/contrib/qep.py:117`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L117)
- doc: Stagewise results from a QEP run.
- signature: `class QepResult:`
- members:
  - `model` — [`L147`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L147)
  - `params` — [`L148`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L148)
  - `quant_stats` — [`L149`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L149)
  - `stages` — [`L150`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L150)
- uses (calls/refs, reference-scoped): [`QepStage`](qep.md#QepStage)
- used by: [`quantize`](qep.md#quantize)  (4 test-only)

### `QepRule`  ·  implements/extends GptqRule
- def: [`qwix/contrib/qep.py:48`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L48) — documented in [qwix-contrib-qep](../../../concepts/qwix-contrib-qep.md)
- doc: Use this rule to enable QEP (input-compensated GPTQ).
- signature: `class QepRule(gptq.GptqRule):`
- members:
  - `apply_correction` — [`L76`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L76)
  - `correction_factor` — [`L74`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L74)
  - `damping_factor` — [`L75`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L75)
- uses (calls/refs, reference-scoped): [`GptqRule`](gptq.md#GptqRule)
- used by: [`quantize`](qep.md#quantize), [`_quantize_weight`](qep.md#_quantize_weight), [`_collect_stats`](qep.md#_CaptureProvider._collect_stats), [`GptqRule`](gptq.md#GptqRule), [`_quantize`](qep.md#quantize_params._quantize), [`apply_quantization`](qep.md#quantize.apply_quantization), [`get_rule_type`](qep.md#_CaptureProvider.get_rule_type), [`rule`](qep.md#_MatchedOp.rule)  (7 test-only)

### `QepStage`
- def: [`qwix/contrib/qep.py:80`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L80)
- doc: Metadata about one QEP stage.
- signature: `class QepStage:`
- members:
  - `index` — [`L111`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L111)
  - `module_paths` — [`L113`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L113)
  - `param_paths` — [`L112`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L112)
- used by: [`_convert_internal_stage_to_public_metadata`](qep.md#_convert_internal_stage_to_public_metadata), [`stages`](qep.md#QepResult.stages)  (1 test-only)

### `_CaptureProvider`  ·  implements/extends CalibrationProvider
- def: [`qwix/contrib/qep.py:330`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L330)
- doc: Provider that records matched operations and captures activations.
- signature: `class _CaptureProvider(calibration.CalibrationProvider):`
- members:
  - `__init__(self, rules: Sequence[qconfig.QuantizationRule])` — [`L350`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L350) — Initializes the provider with the matched rules.
  - `_collect_stats(self, lhs: jax.Array, weight_name: str, *, module_path: tuple[str, ...], op_name: str, op_id: str | None, lhs_id: int)` — [`L421`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L421) — Intervenes in `CalibrationProvider` callbacks to persist state traces. — documented in [qwix-contrib-qep](../../../concepts/qwix-contrib-qep.md)
  - `discovered_ops(self)` — [`L412`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L412) — Returns chronologically traced QEP ops mapping exact stage assignments.
  - `get_rule_type(self)` — [`L361`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L361) — Restricts activation capture strictly to operations matching `QepRule`.
  - `get_stats_suffix(self)` — [`L369`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L369) — Returns the dedicated suffix tracking the QEP calibration artifacts.
  - `prepare_for_capture(self, op_keys: Collection[tuple[Any, ...]])` — [`L389`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L389) — Prepares the provider to intercept activations for the selected ops.
  - `prepare_for_discovery(self)` — [`L377`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L377) — Readies the telemetry listener for a new topological discovery pass.
- protocol/private: `_capture_keys`[`L358`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L358), `_captures`[`L359`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L359), `_discovered_ops`[`L357`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L357)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`_get_current_rule_and_op_id`](../_src/qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`QepRule`](qep.md#QepRule), [`CalibrationProvider`](calibration.md#CalibrationProvider), [`__init__`](../_src/qconfig.md#QuantizationProvider.__init__), [`path`](qep.md#_MatchedOp.path), [`_MatchedOp`](qep.md#_MatchedOp), [`rule`](qep.md#_MatchedOp.rule), [`op_key`](qep.md#_MatchedOp.op_key), [`lhs_id`](qep.md#_MatchedOp.lhs_id)
- used by: [`quantize`](qep.md#quantize), [`replay_and_collect_stats`](qep.md#quantize.replay_and_collect_stats), [`get_rule_type`](calibration.md#CalibrationProvider.get_rule_type), [`get_stats_suffix`](calibration.md#CalibrationProvider.get_stats_suffix), [`CalibrationProvider`](calibration.md#CalibrationProvider), [`_collect_stats`](calibration.md#CalibrationProvider._collect_stats)

### `_MatchedOp`
- def: [`qwix/contrib/qep.py:154`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L154)
- doc: One supported op matched during discovery.
- signature: `class _MatchedOp:`
- members:
  - `lhs_id` — [`L188`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L188)
  - `op_key` — [`L186`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L186)
  - `path` — [`L187`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L187)
  - `rule` — [`L189`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L189)
- uses (calls/refs, reference-scoped): [`QepRule`](qep.md#QepRule)
- used by: [`quantize`](qep.md#quantize), [`_collect_stats`](qep.md#_CaptureProvider._collect_stats), [`replay_and_collect_stats`](qep.md#quantize.replay_and_collect_stats), [`_convert_internal_stage_to_public_metadata`](qep.md#_convert_internal_stage_to_public_metadata), [`_group_discovered_ops_into_stages`](qep.md#_group_discovered_ops_into_stages), [`members`](qep.md#_StageSpec.members), [`discovered_ops`](qep.md#_CaptureProvider.discovered_ops), [`_discovered_ops`](qep.md#_CaptureProvider._discovered_ops)

### `_StageSpec`
- def: [`qwix/contrib/qep.py:193`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L193)
- doc: Internal stage specification.
- signature: `class _StageSpec:`
- members:
  - `index` — [`L221`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L221)
  - `members` — [`L222`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L222)
- uses (calls/refs, reference-scoped): [`_MatchedOp`](qep.md#_MatchedOp)
- used by: [`quantize`](qep.md#quantize), [`replay_and_collect_stats`](qep.md#quantize.replay_and_collect_stats), [`_convert_internal_stage_to_public_metadata`](qep.md#_convert_internal_stage_to_public_metadata), [`_group_discovered_ops_into_stages`](qep.md#_group_discovered_ops_into_stages)

## Functions
- `_append_qep_suffix_to_path(path: tuple[str, ...])` — [`L282`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L282) — Appends the `_qep` suffix to the leaf weight name in a tuple path.
- `_convert_internal_stage_to_public_metadata(stage: _StageSpec)` — [`L527`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L527) — Translates internal stage structures into user-facing metadata.
- `_create_ptq_model_and_abstract_quantized_params(model: nn.Module, rules: Sequence[qconfig.QuantizationRule], methods: Collection[str], sample_args: Sequence[Any], sample_kwargs: Mapping[str, Any], abstract_quantized: Any)` — [`L583`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L583) — Creates the PTQ model and abstract quantized parameters.
- `_flatten_params_tree_to_tuple_paths(tree: Any)` — [`L245`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L245) — Flattens a nested parameters tree into a single-level dictionary.
- `_group_discovered_ops_into_stages(discovered_ops: tuple[_MatchedOp, ...])` — [`L477`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L477) — Constructs grouped sequential quantization stages from discovered layers. — documented in [qwix-contrib-qep](../../../concepts/qwix-contrib-qep.md)
- `_quantize(prepared: calibration.CalibratedQuantContext)` — [`L1015`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L1015) — documented in [qwix-contrib-qep](../../../concepts/qwix-contrib-qep.md)
- `_quantize_weight(ctx: calibration.CalibratedQuantContext, rule: QepRule, gptq_block_size: int, gptq_damping_factor: float)` — [`L631`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L631) — Generates a compressed discrete weight, adapting the QEP formula. — documented in [qwix-contrib-qep](../../../concepts/qwix-contrib-qep.md)
- `_run_model_forward_with_injected_params(model: nn.Module, variables: Any, params: Any, args: Sequence[Any], kwargs: Mapping[str, Any])` — [`L551`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L551) — Executes the flax linen graph, replacing the `params` branch dynamically.
- `_unflatten_tuple_paths_to_params_tree(flat_tree: dict[tuple[str, ...], Any])` — [`L264`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L264) — Reconstructs a deeply nested parameters tree from a flat dictionary.
- `_unfreeze_params_tree(tree: Any)` — [`L225`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L225) — Unfreezes a parameter tree if it is a flax FrozenDict.
- `_update_flat_stats_with_moving_average(flat_stats: dict[tuple[str, ...], Any], path: tuple[str, ...], stats: dict[str, jax.Array])` — [`L298`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L298) — Aggregates batch-level structural statistics via a moving average. — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
- `apply_quantization(stage_rule_by_path: dict[tuple[str, ...], QepRule], stage_flat_stats: dict[tuple[str, ...], Any])` — [`L889`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L889) — Rewrites discrete block parameters based on stage statistics.
- `quantize(model: nn.Module, calibration_data: Iterable[Any] | Callable[[], Iterable[Any]], rules: Sequence[QepRule], *, variables: Any = None, batch_adapter: (Callable[[Any], tuple[Sequence[Any], Mapping[str, Any]]] | None) = None, methods: Collection[str] = ('__call__',), abstract_quantized: Any = None, allow_extra_params: bool = False, gptq_block_size: int = 128, gptq_damping_factor: float = 0.01)` — [`L694`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L694) — Executes Quantization Error Propagation (QEP) on a flax linen model. — documented in [qwix-_src-providers-ptq](../../../concepts/qwix-_src-providers-ptq.md)
- `quantize_params(params: Any, abstract_quantized_params: Any, qep_quant_stats: Any, *, allow_extra_params: bool = False, gptq_block_size: int = 128, gptq_damping_factor: float = 0.01, correction_factor: float = 0.5, damping_factor: float = 0.01, apply_correction: bool = True)` — [`L961`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L961) — Quantizes parameters from precomputed QEP statistical metrics offline.
- `replay_and_collect_stats(stage: _StageSpec)` — [`L828`](../../../../../../raw/code/qwix/qwix/contrib/qep.py#L828) — Sweeps calibration data across dual paths to capture divergent artifacts.

