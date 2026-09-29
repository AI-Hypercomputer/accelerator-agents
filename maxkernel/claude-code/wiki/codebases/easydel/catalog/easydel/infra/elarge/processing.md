---
title: 'Module: easydel/infra/elarge/processing.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/processing.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.processing`/
symbols:
  normalize: normalize().
  TASK_ALIASES.TASK_ALIASES: TASK_ALIASES.TASK_ALIASES.
  resolve_task: resolve_task().
  materialize_base_config: materialize_base_config().
  make_serializable: make_serializable().
  materialize_base_config.set_maybe: materialize_base_config().set_maybe().
  save_elm_config: save_elm_config().
  coerce_dtype: coerce_dtype().
  load_elm_config: load_elm_config().
  normalize_task: normalize_task().
  _CodeEvalMetricProxy.compute: _CodeEvalMetricProxy#compute().
  override_lm_eval_code_exec: override_lm_eval_code_exec().
  validate: validate().
  deep_merge: deep_merge().
  _run_code_eval_metric_compute: _run_code_eval_metric_compute().
  prune_nones: prune_nones().
  coerce_precision: coerce_precision().
  write_text_atomic: write_text_atomic().
  _patch_loaded_code_eval_metric: _patch_loaded_code_eval_metric().
  as_map: as_map().
  override_lm_eval_code_exec._patched_load: override_lm_eval_code_exec()._patched_load().
  _CodeEvalMetricProxy.__getattr__: _CodeEvalMetricProxy#__getattr__().
  _CodeEvalMetricProxy._num_workers: _CodeEvalMetricProxy#_num_workers.
  _CodeEvalMetricProxy._timeout: _CodeEvalMetricProxy#_timeout.
  _CodeEvalMetricProxy: _CodeEvalMetricProxy#
  _CodeEvalMetricProxy._metric: _CodeEvalMetricProxy#_metric.
  _CODE_EVAL_METRIC_RUNNER: _CODE_EVAL_METRIC_RUNNER.
  _estimate_code_eval_timeout: _estimate_code_eval_timeout().
  _stringify_callable: _stringify_callable().
  logger: logger.
  _CodeEvalMetricProxy.__init__: _CodeEvalMetricProxy#__init__().
---
# Module: [`easydel/infra/elarge/processing.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py)

## Classes
### `_CodeEvalMetricProxy`
- def: [`easydel/infra/elarge/processing.py:429`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L429)
- doc: Proxy a Hugging Face `code_eval` metric with overridden execution kwargs.
- signature: `class _CodeEvalMetricProxy:`
- members:
  - `compute(self, *args: Any, **kwargs: Any)` — [`L437`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L437) — Run code-eval in a clean subprocess while injecting override kwargs.
- protocol/private: `__getattr__`[`L445`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L445), `__init__`[`L432`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L432), `_metric`[`L433`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L433), `_num_workers`[`L434`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L434), `_timeout`[`L435`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L435)
- uses (calls/refs, reference-scoped): [`_run_code_eval_metric_compute`](processing.md#_run_code_eval_metric_compute)
- used by: [`_patch_loaded_code_eval_metric`](processing.md#_patch_loaded_code_eval_metric)

## Functions
- `_estimate_code_eval_timeout(*, predictions: list[list[str]], num_workers: int | None, timeout: float | None)` — [`L480`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L480) — Estimate a generous wall clock bound for a full code-eval compute call.
- `_patch_loaded_code_eval_metric(metric: Any, *, num_workers: int | None, timeout: float | None, patched: list[tuple[Any, str, Any]], patched_code_eval_modules: set[tuple[str, str]])` — [`L530`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L530) — Wrap a loaded Hugging Face ``code_eval`` metric with isolated execution.
- `_patched_load(path: Any, *args: Any, **kwargs: Any)` — [`L572`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L572)
- `_run_code_eval_metric_compute(*args: Any, **kwargs: Any)` — [`L496`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L496) — Execute Hugging Face ``code_eval.compute`` in a standalone Python process.
- `_stringify_callable(obj: Any)` — [`L617`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L617) — Return a stable, human-readable identifier for a callable-like object.
- `as_map(cfg: Any)` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L124) — Convert a configuration object to a dictionary representation.
- `coerce_dtype(x: DTypeLike | None)` — [`L213`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L213) — Convert a dtype-like specification to a JAX numpy dtype.
- `coerce_precision(p: PrecisionLike)` — [`L297`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L297) — Convert a precision-like specification to a JAX Precision enum.
- `deep_merge(base: dict[str, Any], overlay: dict[str, Any])` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L169) — Deep merge two dictionaries with overlay values taking precedence.
- `load_elm_config(json_file_path: str | os.PathLike | ePathLike)` — [`L1022`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L1022) — Load an eLMConfig from a JSON file.
- `make_serializable(obj: Any)` — [`L628`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L628) — Convert arbitrary config-like objects into JSON/YAML-safe primitives.
- `materialize_base_config(cfg: eLMConfig, prefer: tp.Literal["base", "sections"] = "base")` — [`L786`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L786) — Materialize a complete base configuration from ELM config sections.
- `normalize(cfg: eLMConfig | Mapping[str, Any])` — [`L730`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L730) — Normalize an ELM configuration by merging with defaults and processing values.
- `normalize_task(t: TaskType | str | None)` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L380) — Normalize a task type specification to a TaskType enum value.
- `override_lm_eval_code_exec(*, num_workers: int | None = None, timeout: float | None = None)` — [`L545`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L545) — Temporarily override lm-eval code-task execution settings.
- `prune_nones(obj: Any)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L81) — Recursively remove None values from nested data structures.
- `resolve_task(cfg: eLMConfig)` — [`L690`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L690) — Resolve the task type from an ELM configuration.
- `save_elm_config(config: eLMConfig | Mapping[str, Any], json_file_path: str | os.PathLike | ePathLike)` — [`L979`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L979) — Save an eLMConfig to a JSON file.
- `set_maybe(k: str, v: Any)` — [`L853`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L853) — Set a key in base dict if value is not None, respecting prefer mode.
- `validate(cfg_like: eLMConfig | Mapping[str, Any])` — [`L908`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L908) — Validate an ELM configuration for correctness and consistency.
- `write_text_atomic(path: str | os.PathLike | ePathLike, data: str, *, encoding: str = "utf-8")` — [`L661`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L661) — Atomically write text to a file.

## Module values
- `TASK_ALIASES` — [`L349`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L349)
- `_CODE_EVAL_METRIC_RUNNER` — [`L449`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L449)
- `logger` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/processing.py#L78)

