---
title: 'Module: tokamax/_src/benchmarking.py'
type: catalog
provenance: extracted
module: tokamax/_src/benchmarking.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.benchmarking`/
symbols:
  compile_benchmark.runner: compile_benchmark().runner().
  XprofProfileSession.__enter__: XprofProfileSession#__enter__().
  XprofProfileSession.__exit__: XprofProfileSession#__exit__().
  T: T.
  standardize_function: standardize_function().
  _TIMERS._TIMERS: _TIMERS._TIMERS.
  XprofProfileSession.total_op_time: XprofProfileSession#total_op_time().
  register_benchmark.bmark: register_benchmark().bmark().
  benchmark: benchmark().
  XprofProfileSession._jax_profiler_mode: XprofProfileSession#_jax_profiler_mode.
  BenchmarkData.median_evaluation_time_ms: BenchmarkData#median_evaluation_time_ms().
  xprof_timer: xprof_timer().
  compile_benchmark: compile_benchmark().
  hermetic_xprof_timer: hermetic_xprof_timer().
  XprofProfileSession._profile_tempdir: XprofProfileSession#_profile_tempdir.
  standardize_function.forward: standardize_function().forward().
  standardize_function.vjp_full: standardize_function().vjp_full().
  wallclock_timer: wallclock_timer().
  xprof_timer.timer: xprof_timer().timer().
  XprofProfileSession._profiler_wallclock_time: XprofProfileSession#_profiler_wallclock_time.
  PyTree: PyTree.
  BenchmarkData: BenchmarkData#
  XprofProfileSession._profiler_wallclock_start_time: XprofProfileSession#_profiler_wallclock_start_time.
  standardize_function.convert_batched: standardize_function().convert_batched().
  cupti_timer: cupti_timer().
  _optimization_barrier: _optimization_barrier().
  RetT.RetT: RetT.RetT.
  BenchmarkData.evaluation_times_ms: BenchmarkData#evaluation_times_ms.
  XprofProfileSession._xprof_session: XprofProfileSession#_xprof_session.
  Timer.Timer: Timer.Timer.
  register_benchmark: register_benchmark().
  get_benchmark_registrar: get_benchmark_registrar().
  profile_data: profile_data.
  BenchmarkMode.BenchmarkMode: BenchmarkMode.BenchmarkMode.
  get_benchmark_registrar.registrar: get_benchmark_registrar().registrar().
  _DEFAULT_TIMING_METHOD._DEFAULT_TIMING_METHOD: _DEFAULT_TIMING_METHOD._DEFAULT_TIMING_METHOD.
  XprofProfileSession._profile: XprofProfileSession#_profile.
  xprof_session: xprof_session.
  TimingMethod.TimingMethod: TimingMethod.TimingMethod.
  BenchmarkData.asdict: BenchmarkData#asdict().
  logger: logger.
  BenchmarkData.metadata: BenchmarkData#metadata.
  XprofProfileSession.xprof_url: XprofProfileSession#xprof_url.
  XprofProfileSession._event_filter_regex: XprofProfileSession#_event_filter_regex.
  XprofProfileSession._retain_artifacts: XprofProfileSession#_retain_artifacts.
  WORKLOAD_ARTIFACTS_DIR_VARNAME.WORKLOAD_ARTIFACTS_DIR_VARNAME: WORKLOAD_ARTIFACTS_DIR_VARNAME.WORKLOAD_ARTIFACTS_DIR_VARNAME.
  get_tempdir: get_tempdir().
  BenchmarkData.compile_time_ms: BenchmarkData#compile_time_ms.
  BenchmarkData.lower_time_ms: BenchmarkData#lower_time_ms.
  BenchmarkData.peak_memory_mb: BenchmarkData#peak_memory_mb.
  XprofProfileSession: XprofProfileSession#
  XprofProfileSession.IGNORE_LINE_PATTERNS: XprofProfileSession#IGNORE_LINE_PATTERNS.
  XprofProfileSession._hermetic: XprofProfileSession#_hermetic.
  XprofProfileSession._xprof_session_kwargs: XprofProfileSession#_xprof_session_kwargs.
  _ARRAY_TYPES: _ARRAY_TYPES.
  wallclock_timer.timer: wallclock_timer().timer().
  _FALLBACK_TIMING_METHOD: _FALLBACK_TIMING_METHOD.
  RETAIN_ARTIFACTS_VARNAME.RETAIN_ARTIFACTS_VARNAME: RETAIN_ARTIFACTS_VARNAME.RETAIN_ARTIFACTS_VARNAME.
  XprofProfileSession.__init__: XprofProfileSession#__init__().
---
# Module: [`tokamax/_src/benchmarking.py`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py)

## Classes
### `BenchmarkData`
- def: [`tokamax/_src/benchmarking.py:85`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L85)
- doc: Time and memory benchmarking data.
- signature: `class BenchmarkData:`
- members:
  - `asdict(self)` — [`L99`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L99) — Represent the BenchmarkData object as a dictionary.
  - `median_evaluation_time_ms(self)` — [`L96`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L96)
  - `compile_time_ms` — [`L88`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L88)
  - `evaluation_times_ms` — [`L90`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L90)
  - `lower_time_ms` — [`L89`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L89)
  - `metadata` — [`L91`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L91)
  - `peak_memory_mb` — [`L93`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L93)
- used by: [`runner`](benchmarking.md#compile_benchmark.runner), [`benchmark`](ops/op.md#BoundArguments.benchmark), [`test_tgmm_perf_regression`](ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.md#GmmPerfTest.test_tgmm_perf_regression), [`bmark`](benchmarking.md#register_benchmark.bmark), [`benchmark`](benchmarking.md#benchmark), [`compile_benchmark`](benchmarking.md#compile_benchmark), [`test_gmm_perf_regression`](ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.md#GmmPerfTest.test_gmm_perf_regression)

### `XprofProfileSession`  ·  implements/extends AbstractContextManager
- def: [`tokamax/_src/benchmarking.py:106`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L106)
- doc: XProf context manager for profiling XLA Ops.
- signature: `class XprofProfileSession(contextlib.AbstractContextManager):`
- members:
  - `__init__(self, hermetic: bool = True, use_jax_profiler: bool = False, event_filter_regex: str | None = None, **xprof_session_kwargs)` — [`L126`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L126) — Initializer.
  - `total_op_time(self)` — [`L165`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L165) — Returns the total device time of XLA operators.
  - `IGNORE_LINE_PATTERNS` — [`L122`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L122)
  - `xprof_url` — [`L153`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L153)
- protocol/private: `__enter__`[`L225`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L225), `__exit__`[`L259`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L259), `_event_filter_regex`[`L160`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L160), `_hermetic`[`L152`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L152), `_jax_profiler_mode`[`L154`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L154), `_profile`[`L150`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L150), `_profile_tempdir`[`L159`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L159), `_profiler_wallclock_start_time`[`L157`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L157), `_profiler_wallclock_time`[`L158`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L158), `_retain_artifacts`[`L162`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L162), `_xprof_session`[`L151`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L151), `_xprof_session_kwargs`[`L161`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L161)
- uses (calls/refs, reference-scoped): [`profile_data`](benchmarking.md#profile_data), [`xprof_session`](benchmarking.md#xprof_session), [`WORKLOAD_ARTIFACTS_DIR_VARNAME`](benchmarking.md#WORKLOAD_ARTIFACTS_DIR_VARNAME.WORKLOAD_ARTIFACTS_DIR_VARNAME), [`logger`](benchmarking.md#logger), [`RETAIN_ARTIFACTS_VARNAME`](benchmarking.md#RETAIN_ARTIFACTS_VARNAME.RETAIN_ARTIFACTS_VARNAME), [`get_tempdir`](benchmarking.md#get_tempdir)
- used by: [`timer`](benchmarking.md#xprof_timer.timer)

## Functions
- `_optimization_barrier(x: T)` — [`L74`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L74)
- `benchmark(f: Callable[[T], Any], x: T, *, iterations: int = 5, method: TimingMethod | None = None, event_filter_regex: str | None = None)` — [`L568`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L568) — Benchmarks a function on a specific input.
- `bmark(state, metadata=metadata)` — [`L636`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L636)
- `compile_benchmark(f: Callable[[T], Any], x: T)` — [`L479`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L479) — Compiles a function and returns a function to benchmark it. — documented in [tokamax-_src-benchmarking](../../../concepts/tokamax-_src-benchmarking.md)
- `convert_batched(x)` — [`L396`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L396)
- `cupti_timer(f: Callable[[T], Any], args: T)` — [`L437`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L437)
- `forward(arrays: list[jax.Array])` — [`L375`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L375)
- `get_benchmark_registrar(impls: dict[str, Callable[..., Any]])` — [`L671`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L671) — Returns a function that registers benchmarks by implementation name.
- `get_tempdir(prefix: str, dir: str | pathlib.Path | None = None)` — [`L64`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L64)
- `hermetic_xprof_timer(f: Callable[[T], Any], args: T, event_filter_regex: str | None = None)` — [`L458`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L458)
- `register_benchmark(name: str, impl_name: str, impl: Callable[..., Any], kwargs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None, *, mode: Literal['forward', 'forward_res', 'vjp', 'forward_and_vjp'] = 'forward', items_processed_fn: Callable[..., int] | None = None, raise_on_error: bool = True, metadata: dict[str, Any] | None = None, **bmark_kwargs: Any)` — [`L610`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L610) — Creates and registers a Google benchmark.
- `registrar(name, impl_name, kwargs, **bmark_kwargs)` — [`L676`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L676)
- `runner(x: T, *, iterations: int = 5, method: TimingMethod | None = None, event_filter_regex: str | None = None)` — [`L502`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L502) — Runs the compiled benchmark. — documented in [tokamax-_src-benchmarking](../../../concepts/tokamax-_src-benchmarking.md)
- `standardize_function(f: Callable[..., T], *args: PyTree, kwargs: Mapping[str, PyTree] | None = None, mode: BenchmarkMode = ..., seed: int = ...)` — [`L311`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L311)
- `timer(_)` — [`L428`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L428)
- `timer(return_metadata)` — [`L445`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L445)
- `vjp_full(arrays: list[jax.Array])` — [`L411`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L411)
- `wallclock_timer(f: Callable[[T], Any], args: T)` — [`L427`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L427)
- `xprof_timer(f: Callable[[T], Any], args: T, event_filter_regex: str | None = None)` — [`L442`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L442)

## Module values
- `BenchmarkMode` — [`L41`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L41)
- `PyTree` — [`L45`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L45)
- `RETAIN_ARTIFACTS_VARNAME` — [`L61`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L61)
- `RetT` — [`L51`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L51)
- `T` — [`L50`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L50)
- `Timer` — [`L48`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L48)
- `TimingMethod` — [`L53`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L53)
- `WORKLOAD_ARTIFACTS_DIR_VARNAME` — [`L60`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L60)
- `_ARRAY_TYPES` — [`L305`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L305)
- `_DEFAULT_TIMING_METHOD` — [`L472`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L472)
- `_FALLBACK_TIMING_METHOD` — [`L476`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L476)
- `_TIMERS` — [`L465`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L465)
- `logger` — [`L57`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L57)
- `profile_data` — [`L39`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L39)
- `xprof_session` — [`L39`](../../../../../../raw/code/tokamax/tokamax/_src/benchmarking.py#L39)

