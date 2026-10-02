---
title: 'Module: easydel/infra/elarge/benchmarking.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/benchmarking.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.benchmarking`/
symbols:
  run_lm_eval_with_esurge: run_lm_eval_with_esurge().
  normalize_benchmark_configs: normalize_benchmark_configs().
  _get_or_create_task_manager: _get_or_create_task_manager().
  _cached_task_manager._cached_task_manager: _cached_task_manager._cached_task_manager.
  maybe_resolve_instruct_task_variants: maybe_resolve_instruct_task_variants().
  _probe_task_names: _probe_task_names().
  maybe_disable_chat_template_for_prefilled_tasks: maybe_disable_chat_template_for_prefilled_tasks().
  _default_benchmark_name: _default_benchmark_name().
  _normalize_tasks: _normalize_tasks().
  logger: logger.
  is_benchmark_config_like: is_benchmark_config_like().
  override_lm_eval_code_exec: override_lm_eval_code_exec().
  _task_name_fragment: _task_name_fragment().
  task_uses_code_eval: task_uses_code_eval().
  _task_declares_generation_prefix: _task_declares_generation_prefix().
  task_uses_code_eval._matches: task_uses_code_eval()._matches().
  _cached_task_manager_key._cached_task_manager_key: _cached_task_manager_key._cached_task_manager_key.
  flatten_benchmark_metrics: flatten_benchmark_metrics().
  auto_code_eval_num_workers: auto_code_eval_num_workers().
---
# Module: [`easydel/infra/elarge/benchmarking.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py)

## Functions
- `_default_benchmark_name(tasks: collections.abc.Sequence[BenchmarkTask], index: int)` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L137) — Build an auto-generated benchmark name from its task list.
- `_get_or_create_task_manager(*, verbosity: str | None = None, include_path: str | list | None = None, include_defaults: bool = True, metadata: dict | None = None, summary_logger: Any = None)` — [`L40`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L40) — Return a cached ``lm_eval.tasks.TaskManager``, building one on first call.
- `_matches(value: Any)` — [`L263`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L263) — Check if a value contains a code-eval task hint.
- `_normalize_tasks(tasks: BenchmarkTasks)` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L161) — Coerce a flexible task specification into a flat, non-empty list.
- `_probe_task_names(names: list[str])` — [`L339`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L339) — Check which task names are registered in lm-eval without building a full index.
- `_task_declares_generation_prefix(task: BenchmarkTask, task_manager: Any | None)` — [`L358`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L358) — Return True when an lm-eval task config already defines a generation prefill.
- `_task_name_fragment(task: BenchmarkTask)` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L109) — Extract a short human-readable label from a task specification.
- `auto_code_eval_num_workers()` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L283) — Return a conservative default worker count for Hugging Face ``code_eval``.
- `flatten_benchmark_metrics(benchmark_name: str, results: collections.abc.Mapping[str, Any])` — [`L401`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L401) — Flatten lm-eval result metrics into a single dict with namespaced keys.
- `is_benchmark_config_like(value: Any)` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L183) — Return ``True`` when *value* looks like a :class:`BenchmarkConfig` mapping.
- `maybe_disable_chat_template_for_prefilled_tasks(tasks: collections.abc.Sequence[BenchmarkTask], *, task_manager: Any | None, apply_chat_template: bool | str, summary_logger: Any | None = None)` — [`L373`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L373) — Disable lm-eval chat templating when tasks already define a generation prefix.
- `maybe_resolve_instruct_task_variants(tasks: collections.abc.Sequence[BenchmarkTask], *, task_manager: Any | None, apply_chat_template: bool | str, summary_logger: Any | None = None)` — [`L295`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L295) — Swap raw task names for available ``*_instruct`` variants when chat mode is enabled.
- `normalize_benchmark_configs(benchmarks: BenchmarkConfig | collections.abc.Sequence[BenchmarkConfig] | None, *, default_eval_config: collections.abc.Mapping[str, Any] | None = None)` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L198) — Validate and resolve a raw benchmark specification into a list of :class:`ResolvedBenchmarkConfig`.
- `override_lm_eval_code_exec(*, num_workers: int | None = None, timeout: float | None = None)` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L89) — Return a context manager that patches lm-eval's code-execution scorer.
- `run_lm_eval_with_esurge(*, surge: Any, processor: Any, tasks: str | collections.abc.Sequence[BenchmarkTask], max_length: int, fallback_batch_size: int | str | None, num_fewshot: int | None = None, eval_config: collections.abc.Mapping[str, Any] | None = None, stop_engine: bool = True, summary_logger: Any | None = None)` — [`L438`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L438) — Run lm-evaluation-harness tasks against an :class:`~easydel.inference.eSurge` engine.
- `task_uses_code_eval(task: Any)` — [`L259`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L259) — Return True when a task specification looks like Humaneval/MBPP style code eval.

## Module values
- `_cached_task_manager` — [`L36`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L36)
- `_cached_task_manager_key` — [`L37`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L37)
- `logger` — [`L34`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/benchmarking.py#L34)

