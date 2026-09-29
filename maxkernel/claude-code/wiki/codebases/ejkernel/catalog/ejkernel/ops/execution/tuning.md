---
title: 'Module: ejkernel/ops/execution/tuning.py'
type: catalog
provenance: extracted
module: ejkernel/ops/execution/tuning.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.execution.tuning`/
symbols:
  FNAutotuner.tune: FNAutotuner#tune().
  autotune_recorded: autotune_recorded().
  FNAutotuner.wrapped: FNAutotuner#wrapped().
  Autotuner.autotune: Autotuner#autotune().
  autotune_logger: autotune_logger.
  AutotuningResult.as_overlay: AutotuningResult#as_overlay().
  FNAutotuner._time_fn: FNAutotuner#_time_fn().
  AutotuningResult._ctx: AutotuningResult#_ctx.
  FNAutotuner._try_call: FNAutotuner#_try_call().
  AutotuneData.fastest_config: AutotuneData#fastest_config().
  FNAutotuner.profiler: FNAutotuner#profiler.
  FNAutotuner._calculate_timing_score: FNAutotuner#_calculate_timing_score().
  benchmark: benchmark().
  Entry: Entry#
  autotune: autotune().
  AutotuneData: AutotuneData#
  AutotuningResult: AutotuningResult#
  TimingResult: TimingResult#
  FNAutotuner._create_parameterized_function: FNAutotuner#_create_parameterized_function().
  FNAutotuner.decorate: FNAutotuner#decorate().
  Autotuner: Autotuner#
  Cfg: Cfg.
  Measurement: Measurement#
  AutotuningResult.__exit__: AutotuningResult#__exit__().
  AutotuneData.measurements: AutotuneData#measurements.
  _get_random_value: _get_random_value().
  autotune.decorator: autotune().decorator().
  AutotuningResult.entries: AutotuningResult#entries.
  _get_global_mesh: _get_global_mesh().
  _normalize_sharding: _normalize_sharding().
  _try_hash_input: _try_hash_input().
  _try_hash_input.array_to_hashable: _try_hash_input().array_to_hashable().
  FNAutotuner._timing_closure: FNAutotuner#_timing_closure().
  FNAutotuner: FNAutotuner#
  FNAutotuner.parameterized_function: FNAutotuner#parameterized_function().
  Autotuner.iters: Autotuner#iters.
  _get_default_device: _get_default_device().
  FNAutotuner.calls_per_round: FNAutotuner#calls_per_round.
  FNAutotuner._execute_and_block: FNAutotuner#_execute_and_block().
  FNAutotuner._extract_array_type: FNAutotuner#_extract_array_type().
  Measurement.cfg: Measurement#cfg.
  Measurement.seconds: Measurement#seconds.
  Autotuner.warmup: Autotuner#warmup.
  Entry.op_id_v: Entry#op_id_v.
  Entry.call_key: Entry#call_key.
  Entry.cfg: Entry#cfg.
  AutotuningResult.device: AutotuningResult#device.
  autotune_recorded.mk: autotune_recorded().mk().
  _split_static_callable_kwargs: _split_static_callable_kwargs().
  benchmark.fn_wrapped: benchmark().fn_wrapped().
  TimingResult.hyperparams: TimingResult#hyperparams.
  TimingResult.t_mean: TimingResult#t_mean.
  TimingResult.t_std: TimingResult#t_std.
  _ensure_dtype: _ensure_dtype().
  _try_hash_input._get_sharding: _try_hash_input()._get_sharding().
  FNAutotuner.PREFIX_FN: FNAutotuner#PREFIX_FN.
  FNAutotuner.allow_fallback_timing: FNAutotuner#allow_fallback_timing.
  FNAutotuner.profiling_samples: FNAutotuner#profiling_samples.
  FNAutotuner.must_find_profiler_fraction: FNAutotuner#must_find_profiler_fraction.
  FNAutotuner.enable_detailed_logging: FNAutotuner#enable_detailed_logging.
  FNAutotuner.find_optimal_layouts_automatically: FNAutotuner#find_optimal_layouts_automatically.
  FNAutotuner.max_compilation_time_seconds: FNAutotuner#max_compilation_time_seconds.
  FNAutotuner.timing_warmup_iterations: FNAutotuner#timing_warmup_iterations.
  FNAutotuner.timing_rounds: FNAutotuner#timing_rounds.
  FNAutotuner.cache_size_limit: FNAutotuner#cache_size_limit.
  FNAutotuner.to_shape: FNAutotuner#to_shape().
  FNAutotuner.place_array_on_optimal_device: FNAutotuner#place_array_on_optimal_device().
  Autotuner.__init__: Autotuner#__init__().
  AutotuningResult.__enter__: AutotuningResult#__enter__().
  _suppress_stdout_stderr: _suppress_stdout_stderr().
  FNAutotuner.__init__: FNAutotuner#__init__().
  FNAutotuner._cache_lock: FNAutotuner#_cache_lock.
---
# Module: [`ejkernel/ops/execution/tuning.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py)

## Classes
### `AutotuneData`  ·  implements/extends Generic
- def: [`ejkernel/ops/execution/tuning.py:104`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L104) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- doc: Container for all optimization measurements and results.
- signature: `class AutotuneData(Generic[Cfg]):`
- members:
  - `fastest_config(self)` — [`L120`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L120) — Get the configuration with the fastest execution time.
  - `measurements` — [`L117`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L117)
- uses (calls/refs, reference-scoped): [`Measurement`](tuning.md#Measurement), [`Cfg`](tuning.md#Cfg), [`cfg`](tuning.md#Measurement.cfg), [`seconds`](tuning.md#Measurement.seconds)
- used by: [`autotune`](tuning.md#Autotuner.autotune)

### `Autotuner`  ·  implements/extends Generic
- def: [`ejkernel/ops/execution/tuning.py:137`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L137)
- doc: Core autotuning engine for hyperparameter optimization.
- signature: `class Autotuner(Generic[Cfg]):`
- members:
  - `__init__(self, warmup=1, iters=3)` — [`L152`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L152) — Initialize the autotuner with timing parameters.
  - `autotune(self, make_fn, args, kwargs, candidates: Iterable[Cfg])` — [`L161`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L161) — Optimize hyperparameters by testing candidate configurations. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `iters` — [`L159`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L159) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `warmup` — [`L159`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L159) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- uses (calls/refs, reference-scoped): [`autotune_logger`](tuning.md#autotune_logger), [`AutotuneData`](tuning.md#AutotuneData), [`_is_autotune_progress_enabled`](../config/selection.md#_is_autotune_progress_enabled), [`Measurement`](tuning.md#Measurement), [`Cfg`](tuning.md#Cfg)
- used by: [`Tuner`](__init__.md#Tuner)

### `AutotuningResult`
- def: [`ejkernel/ops/execution/tuning.py:255`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L255) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- doc: Result container for device-specific optimization results.
- signature: `class AutotuningResult:`
- members:
  - `__enter__(self)` — [`L281`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L281) — Enter context manager to apply optimization results as cache overlay.
  - `__exit__(self, exc_type, exc, tb)` — [`L294`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L294) — Exit context manager and restore previous cache state.
  - `as_overlay(self)` — [`L269`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L269) — Convert results to cache overlay mapping format.
  - `device` — [`L266`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L266)
  - `entries` — [`L267`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L267)
- protocol/private: `_ctx`[`L290`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L290)
- uses (calls/refs, reference-scoped): [`__enter__`](../config/cache.md#overlay_cache.__enter__), [`__exit__`](../config/cache.md#overlay_cache.__exit__), [`Entry`](tuning.md#Entry), [`overlay_cache`](../config/cache.md#overlay_cache), [`call_key`](tuning.md#Entry.call_key), [`cfg`](tuning.md#Entry.cfg), [`op_id_v`](tuning.md#Entry.op_id_v)
- used by: [`autotune_lowered`](offline.md#autotune_lowered), [`autotune_recorded`](tuning.md#autotune_recorded)

### `Entry`
- def: [`ejkernel/ops/execution/tuning.py:237`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L237) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- doc: Cache entry for storing optimal configurations.
- signature: `class Entry:`
- members:
  - `call_key` — [`L250`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L250)
  - `cfg` — [`L251`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L251)
  - `op_id_v` — [`L249`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L249)
- used by: [`autotune_lowered`](offline.md#autotune_lowered), [`autotune_recorded`](tuning.md#autotune_recorded), [`as_overlay`](tuning.md#AutotuningResult.as_overlay), [`entries`](tuning.md#AutotuningResult.entries)

### `FNAutotuner`
- def: [`ejkernel/ops/execution/tuning.py:728`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L728)
- doc: Advanced JAX autotuner with profiler-first timing and Python fallback.
- signature: `class FNAutotuner:`
- members:
  - `__init__(self, *, allow_fallback_timing: bool = True, profiling_samples: int = 5, must_find_profiler_fraction: float = 0.5, enable_detailed_logging: bool = False, find_optimal_layouts_automatically: bool = False, max_compilation_time_seconds: float = 300, timing_warmup_iterations: int = 2, timing_rounds: int = 5, calls_per_round: int = 3, cache_size_limit: int = 1000, profiler_prefix_filter: str = "jit_", profiler_event_regex: str | None = None, profiler_min_duration_ns: float = 1000, profiler_max_events: int | None = 10000, profiler_verbose: bool = False)` — [`L781`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L781) — Initialize the autotuner with timing and profiling configuration.
  - `_calculate_timing_score(tr: TimingResult)` — [`L843`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L843) — Calculate a composite timing score for ranking configurations. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `_create_parameterized_function(self, target_function: Callable[..., Any], hyperparameter_values: dict[str, Any], output_shardings: Any = None, function_id: int = 0)` — [`L860`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L860) — Create a JIT-compiled function with embedded hyperparameters.
  - `_execute_and_block()` — [`L1001`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1001) — Execute the target function and block until all outputs are ready. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `_extract_array_type(x)` — [`L1094`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1094) — Extract the abstract type of a JAX array, or pass through non-array values.
  - `_time_fn(self, target_function: Callable[[], None])` — [`L985`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L985) — Perform high-precision Python-level timing with statistical analysis. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `_timing_closure()` — [`L1208`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1208) — Execute all hyperparameter configurations in random order for profiler timing. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `_try_call(self, fn: Callable[..., Any], resolved_args, resolved_kwargs, compile_only: bool = False, compute_layouts: bool = False, optimal_formats: Any | None = None, timeout: float | None = None)` — [`L897`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L897) — Safely compile or execute a function with comprehensive error handling. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `decorate(self, fn: Callable[..., Any], *, hyperparams: dict[str, list[int | float | str]] | dict[Any, Any] | None = None, max_workers: int = 32, in_shardings: Any = None, out_shardings: Any = None, device: jax.Device | str | None = None, example_args: tuple[Any, ...] | None = None, example_kws: dict[str, Any] | None = None, sample_num: int = 2 ** 63 - 1, event_filter_regex: str | None = None, timeout: float | None = None, cache_key: str | None = None)` — [`L1262`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1262) — Create a decorated version of a function with automatic hyperparameter tuning. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `parameterized_function(*function_args, **function_kwargs)` — [`L886`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L886) — Execute the target function with embedded hyperparameter values merged into kwargs.
  - `place_array_on_optimal_device(array_data, target_format)` — [`L956`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L956) — Place an array on its optimal device layout, passing non-arrays through.
  - `to_shape(x)` — [`L933`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L933) — Convert a JAX array to its ShapeDtypeStruct representation for layout optimization.
  - `tune(self, fn: Callable[..., Any], *, args: tuple[Any, ...], kwargs: dict[str, Any], hyperparams: dict[str, list[int | float | str]] | dict[Any, Any] | None = None, max_workers: int = 32, in_shardings: Any = None, out_shardings: Any = None, device: jax.Device | str | None = None, example_args: tuple[Any, ...] | None = None, example_kws: dict[Any, Any] | None = None, sample_num: int = 2 ** 63 - 1, event_filter_regex: str | None = None, timeout: float | None = None)` — [`L1033`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1033) — Tune hyperparameters for a function and return optimal configuration. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `wrapped(*args, **kws)` — [`L1314`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1314) — Wrapper function that performs autotuning on first call and caches results.
  - `PREFIX_FN` — [`L779`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L779)
  - `allow_fallback_timing` — [`L821`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L821) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `cache_size_limit` — [`L830`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L830) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `calls_per_round` — [`L829`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L829) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `enable_detailed_logging` — [`L824`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L824)
  - `find_optimal_layouts_automatically` — [`L825`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L825) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `max_compilation_time_seconds` — [`L826`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L826) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `must_find_profiler_fraction` — [`L823`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L823) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `profiler` — [`L832`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L832) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `profiling_samples` — [`L822`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L822) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `timing_rounds` — [`L828`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L828)
  - `timing_warmup_iterations` — [`L827`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L827)
- protocol/private: `_cache_lock`[`L840`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L840)
- uses (calls/refs, reference-scoped): [`profile_time_by_function_id`](profiler.md#Profiler.profile_time_by_function_id), [`autotune_logger`](tuning.md#autotune_logger), [`TimingResult`](tuning.md#TimingResult), [`_level`](../../loggings.md#LazyLogger._level), [`_get_random_value`](tuning.md#_get_random_value), [`_normalize_sharding`](tuning.md#_normalize_sharding), [`_try_hash_input`](tuning.md#_try_hash_input), [`_pattern`](profiler.md#Profiler._pattern), [`Profiler`](profiler.md#Profiler), [`_get_default_device`](tuning.md#_get_default_device), [`hyperparams`](tuning.md#TimingResult.hyperparams), [`t_mean`](tuning.md#TimingResult.t_mean), [`t_std`](tuning.md#TimingResult.t_std)
- used by: [`autotune`](tuning.md#autotune), [`decorator`](tuning.md#autotune.decorator)

### `Measurement`
- def: [`ejkernel/ops/execution/tuning.py:88`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L88) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- doc: Container for a single performance measurement.
- signature: `class Measurement:`
- members:
  - `cfg` — [`L99`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L99)
  - `seconds` — [`L100`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L100)
- used by: [`autotune`](tuning.md#Autotuner.autotune), [`fastest_config`](tuning.md#AutotuneData.fastest_config), [`measurements`](tuning.md#AutotuneData.measurements)

### `TimingResult`
- def: [`ejkernel/ops/execution/tuning.py:492`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L492) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- doc: Statistical timing result for a single hyperparameter configuration.
- signature: `class TimingResult:`
- members:
  - `hyperparams` — [`L505`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L505) — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
  - `t_mean` — [`L506`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L506)
  - `t_std` — [`L507`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L507)
- used by: [`tune`](tuning.md#FNAutotuner.tune), [`_calculate_timing_score`](tuning.md#FNAutotuner._calculate_timing_score), [`decorate`](tuning.md#FNAutotuner.decorate)

## Functions
- `_ensure_dtype(dt)` — [`L622`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L622) — Return the ``.dtype`` of an array, or the input itself if not an array.
- `_get_default_device()` — [`L530`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L530) — Return the default JAX compute device.
- `_get_global_mesh()` — [`L510`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L510) — Retrieve the currently active global JAX mesh from thread-local state.
- `_get_random_value(sds, sharding=None)` — [`L643`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L643) — Generate a concrete random array matching a shape/dtype specification.
- `_get_sharding(x)` — [`L711`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L711) — Extract sharding from array or its abstract type.
- `_normalize_sharding(arg: jax.Array | np.ndarray | Any, sharding_or_spec: PartitionSpec | Sharding | None, default_device: jax.Device)` — [`L576`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L576) — Convert a sharding specification into a concrete :class:`~jax.sharding.Sharding`.
- `_split_static_callable_kwargs(kwargs)` — [`L434`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L434) — Split keyword arguments into static and dynamic components.
- `_suppress_stdout_stderr()` — [`L546`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L546) — Context manager to temporarily suppress stdout and stderr output.
- `_try_hash_input(args, kws, must_be_concrete: bool = True)` — [`L684`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L684) — Attempt to create a hashable signature key from function input arguments. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- `array_to_hashable(x)` — [`L718`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L718) — Convert array to hashable representation based on type and sharding.
- `autotune(fn: Callable[..., Any] | None = None,, *, allow_fallback_timing: bool = True, profiling_samples: int = 5, must_find_profiler_fraction: float = 0.5, enable_detailed_logging: bool = False, find_optimal_layouts_automatically: bool = False, max_compilation_time_seconds: float = 300, timing_warmup_iterations: int = 2, timing_rounds: int = 5, calls_per_round: int = 3, cache_size_limit: int = 1000, profiler_prefix_filter: str = "jit_", profiler_event_regex: str | None = None, profiler_min_duration_ns: float = 1000, profiler_max_events: int | None = 10000, profiler_verbose: bool = False, hyperparams: dict[str, list[int | float | str]] | dict[Any, Any] | None = None, max_workers: int = 32, in_shardings: Any = None, out_shardings: Any = None, device: jax.Device | str | None = None, example_args: tuple[Any, ...] | None = None, example_kws: dict[str, Any] | None = None, sample_num: int = 2 ** 63 - 1, event_filter_regex: str | None = None, timeout: float | None = None, cache_key: str | None = None)` — [`L1393`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1393) — Advanced JAX function autotuning decorator with comprehensive optimization features.
- `autotune_recorded(hyperparameter_selector, *, show_progress=False, repetition_count=1)` — [`L309`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L309) — Autotune all kernel invocations recorded for the current device. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- `benchmark(fn, *args, warmup=1, iters=5, **kwargs)` — [`L450`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L450) — Benchmark function execution time with JAX compilation. — documented in [ejkernel-ops-execution-tuning](../../../../concepts/ejkernel-ops-execution-tuning.md)
- `decorator(func: Callable[..., Any])` — [`L1573`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L1573) — Internal decorator function that applies autotuning to a target function.
- `fn_wrapped(*a, _fn=fn, _static=static, **k)` — [`L470`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L470) — Wrap function to merge static callable kwargs with dynamic kwargs at call time.
- `mk(c, _run=kernel.run, _static=static_fun_kwargs)` — [`L396`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L396) — Create a partial function binding a specific configuration to the kernel run method.

## Module values
- `Cfg` — [`L84`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L84)
- `autotune_logger` — [`L81`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/tuning.py#L81)

