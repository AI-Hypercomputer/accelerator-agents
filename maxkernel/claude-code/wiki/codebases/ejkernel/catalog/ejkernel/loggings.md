---
title: 'Module: ejkernel/loggings.py'
type: catalog
provenance: extracted
module: ejkernel/loggings.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.loggings`/
symbols:
  create_step_profiler.profile_step: create_step_profiler().profile_step().
  ProgressLogger.complete: ProgressLogger#complete().
  ProgressLogger.update: ProgressLogger#update().
  LazyLogger._ensure_initialized: LazyLogger#_ensure_initialized().
  COLORS.COLORS: COLORS.COLORS.
  ProgressLogger.update_simple: ProgressLogger#update_simple().
  get_logger: get_logger().
  LazyLogger.wrapped_log_method: LazyLogger#wrapped_log_method().
  ProgressLogger._logger: ProgressLogger#_logger.
  ColorFormatter.format: ColorFormatter#format().
  LazyLogger._level: LazyLogger#_level.
  LazyLogger.__getattr__: LazyLogger#__getattr__().
  ProgressLogger._last_message_length: ProgressLogger#_last_message_length.
  create_step_profiler: create_step_profiler().
  logger: logger.
  ProfilerState.completed: ProfilerState#completed.
  ProgressLogger.name: ProgressLogger#name.
  _LOGGING_LEVELS._LOGGING_LEVELS: _LOGGING_LEVELS._LOGGING_LEVELS.
  extinguish_profiler: extinguish_profiler().
  _pulse_output_during_wait: _pulse_output_during_wait().
  LEVEL_COLORS.LEVEL_COLORS: LEVEL_COLORS.LEVEL_COLORS.
  LazyLogger: LazyLogger#
  LazyLogger._logger: LazyLogger#_logger.
  ProgressLogger.use_tty: ProgressLogger#use_tty.
  ProfilerState.active: ProfilerState#active.
  LazyLogger.__init__: LazyLogger#__init__().
  LazyLogger.level: LazyLogger#level().
  LazyLogger.name: LazyLogger#name().
  ProgressLogger.__init__: ProgressLogger#__init__().
  ProgressLogger.__enter__: ProgressLogger#__enter__().
  ProgressLogger.__exit__: ProgressLogger#__exit__().
  LazyLogger._name: LazyLogger#_name.
  ProgressLogger.start_time: ProgressLogger#start_time.
  ColorFormatter: ColorFormatter#
  ProgressLogger: ProgressLogger#
  create_step_profiler.ProfilerState: create_step_profiler().ProfilerState#
  ignite_profiler: ignite_profiler().
  _pulse_output_during_wait.pulse_output: _pulse_output_during_wait().pulse_output().
  create_step_profiler.ProfilerState.__init__: create_step_profiler().ProfilerState#__init__().
---
# Module: [`ejkernel/loggings.py`](../../../../../raw/code/ejkernel/ejkernel/loggings.py)

## Classes
### `ColorFormatter`  ·  implements/extends Formatter
- def: [`ejkernel/loggings.py:116`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L116)
- doc: Custom formatter that adds colors and timestamps to log messages.
- signature: `class ColorFormatter(logging.Formatter):`
- members:
  - `format(self, record: logging.LogRecord)` — [`L126`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L126) — Format a log record with colors and timestamp.
- uses (calls/refs, reference-scoped): [`COLORS`](loggings.md#COLORS.COLORS), [`LEVEL_COLORS`](loggings.md#LEVEL_COLORS.LEVEL_COLORS)
- used by: [`_ensure_initialized`](loggings.md#LazyLogger._ensure_initialized)

### `LazyLogger`
- def: [`ejkernel/loggings.py:149`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L149)
- doc: Lazy-initialized logger that defers creation until first use.
- signature: `class LazyLogger:`
- members:
  - `__getattr__(self, name: str)` — [`L229`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L229) — Dynamically provide logging methods.
  - `__init__(self, name: str, level: int | None = None)` — [`L165`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L165) — Initialize a lazy logger.
  - `_ensure_initialized(self)` — [`L199`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L199) — Initialize the underlying logger if not already done.
  - `level(self)` — [`L182`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L182) — Return the current logging level as an integer.
  - `name(self)` — [`L191`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L191) — Return the logger name.
  - `wrapped_log_method(*args: tp.Any, **kwargs: tp.Any)` — [`L244`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L244)
- protocol/private: `_level`[`L178`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L178), `_logger`[`L179`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L179), `_name`[`L177`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L177)
- uses (calls/refs, reference-scoped): [`_LOGGING_LEVELS`](loggings.md#_LOGGING_LEVELS._LOGGING_LEVELS), [`ColorFormatter`](loggings.md#ColorFormatter)
- used by: [`profiler`](ops/execution/tuning.md#FNAutotuner.profiler), [`get_logger`](loggings.md#get_logger), [`__init__`](loggings.md#ProgressLogger.__init__)

### `ProfilerState`
- def: [`ejkernel/loggings.py:478`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L478)
- doc: Mutable state container for the step-profiler lifecycle.
- signature: `class ProfilerState:`
- members:
  - `__init__(self)` — [`L492`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L492) — Initialize profiler state as inactive and not yet completed.
  - `active` — [`L494`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L494)
  - `completed` — [`L495`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L495)
- used by: [`profile_step`](loggings.md#create_step_profiler.profile_step)

### `ProgressLogger`
- def: [`ejkernel/loggings.py:273`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L273)
- doc: A progress logger that displays updating progress bars and messages.
- signature: `class ProgressLogger:`
- members:
  - `__enter__(self)` — [`L410`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L410) — Enter the context manager, returning this instance.
  - `__exit__(self, exc_type, exc_val, exc_tb)` — [`L422`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L422) — Exit the context manager and finalize progress display.
  - `__init__(self, name: str = "Progress", logger_instance: LazyLogger | None = None)` — [`L294`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L294) — Initialize the progress logger.
  - `complete(self, message: str | None = None, show_time: bool = True)` — [`L378`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L378) — Complete the progress and show final message.
  - `update(self, current: int, total: int, message: str = "", bar_width: int = 20, show_eta: bool = True, extra_info: str = "")` — [`L307`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L307) — Update the progress display.
  - `update_simple(self, message: str)` — [`L361`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L361) — Update with a simple message without progress bar.
  - `name` — [`L301`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L301)
  - `start_time` — [`L303`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L303)
  - `use_tty` — [`L302`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L302)
- protocol/private: `_last_message_length`[`L305`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L305), `_logger`[`L304`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L304)
- uses (calls/refs, reference-scoped): [`get_logger`](loggings.md#get_logger), [`LazyLogger`](loggings.md#LazyLogger)

## Functions
- `_pulse_output_during_wait(completion_signal: threading.Event)` — [`L569`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L569) — Keep output streams alive during blocking profiler shutdown.
- `create_step_profiler(profile_path: str, start_step: int, duration_steps: int, enable_perfetto: bool)` — [`L444`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L444) — Create a step-aware profiler that activates during a specific training window.
- `extinguish_profiler(enable_perfetto: bool)` — [`L544`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L544) — Stop the profiler and handle Perfetto link generation.
- `get_logger(name: str, level: int | None = None)` — [`L252`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L252) — Create a lazy logger that only initializes when first used.
- `ignite_profiler(profile_path: str, enable_perfetto: bool = False)` — [`L526`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L526) — Start the JAX profiler with optional Perfetto integration.
- `profile_step(step: int)` — [`L499`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L499) — Handle profiling lifecycle based on current step.
- `pulse_output()` — [`L588`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L588) — Flush streams once, then print keep-alive messages every 5 seconds.

## Module values
- `COLORS` — [`L73`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L73)
- `LEVEL_COLORS` — [`L87`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L87)
- `_LOGGING_LEVELS` — [`L96`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L96)
- `logger` — [`L441`](../../../../../raw/code/ejkernel/ejkernel/loggings.py#L441)

