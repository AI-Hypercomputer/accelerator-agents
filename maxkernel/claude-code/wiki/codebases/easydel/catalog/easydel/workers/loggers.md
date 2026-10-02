---
title: 'Module: easydel/workers/loggers.py'
type: catalog
provenance: extracted
module: easydel/workers/loggers.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.loggers`/
symbols:
  ProgressLogger.complete: ProgressLogger#complete().
  ProgressLogger.update: ProgressLogger#update().
  get_logger: get_logger().
  eLogger._ensure_initialized: eLogger#_ensure_initialized().
  COLORS.COLORS: COLORS.COLORS.
  ProgressLogger.update_simple: ProgressLogger#update_simple().
  eLogger.wrapped_log_method: eLogger#wrapped_log_method().
  _LOGGING_LEVELS._LOGGING_LEVELS: _LOGGING_LEVELS._LOGGING_LEVELS.
  ProgressLogger._logger: ProgressLogger#_logger.
  ColorFormatter.format: ColorFormatter#format().
  eLogger.__getattr__: eLogger#__getattr__().
  ProgressLogger._last_message_length: ProgressLogger#_last_message_length.
  ProgressLogger.name: ProgressLogger#name.
  LEVEL_COLORS.LEVEL_COLORS: LEVEL_COLORS.LEVEL_COLORS.
  eLogger: eLogger#
  eLogger._level: eLogger#_level.
  eLogger._logger: eLogger#_logger.
  ProgressLogger.use_tty: ProgressLogger#use_tty.
  eLogger.__init__: eLogger#__init__().
  eLogger.level: eLogger#level().
  eLogger.name: eLogger#name().
  ProgressLogger.__init__: ProgressLogger#__init__().
  ProgressLogger.__exit__: ProgressLogger#__exit__().
  eLogger._name: eLogger#_name.
  ProgressLogger.start_time: ProgressLogger#start_time.
  ColorFormatter: ColorFormatter#
  ProgressLogger: ProgressLogger#
  ProgressLogger.__enter__: ProgressLogger#__enter__().
---
# Module: [`easydel/workers/loggers.py`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py)

## Classes
### `ColorFormatter`  ·  implements/extends Formatter
- def: [`easydel/workers/loggers.py:68`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L68)
- doc: Logging formatter that adds ANSI color codes to log output.
- signature: `class ColorFormatter(logging.Formatter):`
- members:
  - `format(self, record: logging.LogRecord)` — [`L76`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L76)
- uses (calls/refs, reference-scoped): [`COLORS`](loggers.md#COLORS.COLORS), [`LEVEL_COLORS`](loggers.md#LEVEL_COLORS.LEVEL_COLORS)
- used by: [`_ensure_initialized`](loggers.md#eLogger._ensure_initialized)

### `ProgressLogger`
- def: [`easydel/workers/loggers.py:167`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L167)
- doc: A progress logger that displays updating progress bars and messages.
- signature: `class ProgressLogger:`
- members:
  - `__enter__(self)` — [`L304`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L304) — Context manager entry.
  - `__exit__(self, exc_type, exc_val, exc_tb)` — [`L308`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L308) — Context manager exit - complete the progress.
  - `__init__(self, name: str = "Progress", logger_instance: eLogger | None = None)` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L188) — Initialize the progress logger.
  - `complete(self, message: str | None = None, show_time: bool = True)` — [`L272`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L272) — Complete the progress and show final message.
  - `update(self, current: int, total: int, message: str = "", bar_width: int = 20, show_eta: bool = True, extra_info: str = "")` — [`L201`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L201) — Update the progress display.
  - `update_simple(self, message: str)` — [`L255`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L255) — Update with a simple message without progress bar.
  - `name` — [`L195`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L195)
  - `start_time` — [`L197`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L197)
  - `use_tty` — [`L196`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L196)
- protocol/private: `_last_message_length`[`L199`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L199), `_logger`[`L198`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L198)
- uses (calls/refs, reference-scoped): [`get_logger`](loggers.md#get_logger), [`eLogger`](loggers.md#eLogger)

### `eLogger`
- def: [`easydel/workers/loggers.py:91`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L91)
- doc: Lazy-initializing logger that defers handler setup until first use.
- signature: `class eLogger:`
- members:
  - `level(self)` — [`L116`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L116)
  - `name(self)` — [`L120`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L120)
  - `wrapped_log_method(*args: tp.Any, **kwargs: tp.Any)` — [`L145`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L145)
- protocol/private: `__getattr__`[`L141`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L141), `__init__`[`L104`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L104), `_ensure_initialized`[`L123`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L123), `_level`[`L112`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L112), `_logger`[`L113`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L113), `_name`[`L111`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L111)
- uses (calls/refs, reference-scoped): [`_LOGGING_LEVELS`](loggers.md#_LOGGING_LEVELS._LOGGING_LEVELS), [`ColorFormatter`](loggers.md#ColorFormatter)
- used by: [`get_logger`](loggers.md#get_logger), [`__init__`](loggers.md#ProgressLogger.__init__)

## Functions
- `get_logger(name: str, level: int | None = None)` — [`L153`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L153) — Function to create a lazy logger that only initializes when first used.

## Module values
- `COLORS` — [`L25`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L25)
- `LEVEL_COLORS` — [`L39`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L39)
- `_LOGGING_LEVELS` — [`L48`](../../../../../../raw/code/EasyDeL/easydel/workers/loggers.py#L48)

