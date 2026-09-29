---
title: 'Module: src/maxdiffusion/utils/logging.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/logging.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.logging`/
symbols:
  get_logger: get_logger().
  _configure_library_root_logger: _configure_library_root_logger().
  _get_library_root_logger: _get_library_root_logger().
  _default_handler._default_handler: _default_handler._default_handler.
  _tqdm_active: _tqdm_active.
  set_verbosity: set_verbosity().
  _reset_library_root_logger: _reset_library_root_logger().
  disable_default_handler: disable_default_handler().
  enable_default_handler: enable_default_handler().
  _get_default_logging_level: _get_default_logging_level().
  get_verbosity: get_verbosity().
  add_handler: add_handler().
  remove_handler: remove_handler().
  disable_propagation: disable_propagation().
  enable_propagation: enable_propagation().
  _tqdm_cls.__call__: _tqdm_cls#__call__().
  log_levels: log_levels.
  get_log_levels_dict: get_log_levels_dict().
  set_verbosity_info: set_verbosity_info().
  set_verbosity_warning: set_verbosity_warning().
  set_verbosity_debug: set_verbosity_debug().
  set_verbosity_error: set_verbosity_error().
  enable_explicit_format: enable_explicit_format().
  reset_format: reset_format().
  EmptyTqdm.__iter__: EmptyTqdm#__iter__().
  EmptyTqdm.__getattr__: EmptyTqdm#__getattr__().
  _tqdm_cls.set_lock: _tqdm_cls#set_lock().
  _tqdm_cls.get_lock: _tqdm_cls#get_lock().
  tqdm: tqdm.
  is_progress_bar_enabled: is_progress_bar_enabled().
  enable_progress_bar: enable_progress_bar().
  disable_progress_bar: disable_progress_bar().
  _lock: _lock.
  _get_library_name: _get_library_name().
  _default_log_level: _default_log_level.
  warning_advice: warning_advice().
  EmptyTqdm: EmptyTqdm#
  EmptyTqdm._iterator: EmptyTqdm#_iterator.
  EmptyTqdm.empty_fn: EmptyTqdm#empty_fn().
  _tqdm_cls: _tqdm_cls#
  EmptyTqdm.__init__: EmptyTqdm#__init__().
  EmptyTqdm.__enter__: EmptyTqdm#__enter__().
  EmptyTqdm.__exit__: EmptyTqdm#__exit__().
  _tqdm_cls._lock: _tqdm_cls#_lock.
---
# Module: [`src/maxdiffusion/utils/logging.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py)

## Classes
### `EmptyTqdm`
- def: [`src/maxdiffusion/utils/logging.py:277`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L277)
- doc: Dummy tqdm which doesn't do anything.
- signature: `class EmptyTqdm:`
- members:
  - `__getattr__(self, _)` — [`L286`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L286) — Return empty function.
  - `empty_fn(*args, **kwargs)` — [`L289`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L289)
- protocol/private: `__enter__`[`L294`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L294), `__exit__`[`L297`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L297), `__init__`[`L280`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L280), `__iter__`[`L283`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L283), `_iterator`[`L281`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L281)
- used by: [`__call__`](logging.md#_tqdm_cls.__call__)

### `_tqdm_cls`
- def: [`src/maxdiffusion/utils/logging.py:301`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L301)
- signature: `class _tqdm_cls:`
- members:
  - `get_lock(self)` — [`L314`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L314)
  - `set_lock(self, *args, **kwargs)` — [`L309`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L309)
- protocol/private: `__call__`[`L303`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L303), `_lock`[`L310`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L310)
- uses (calls/refs, reference-scoped): [`_tqdm_active`](logging.md#_tqdm_active), [`EmptyTqdm`](logging.md#EmptyTqdm)
- used by: [`tqdm`](logging.md#tqdm)

## Functions
- `_configure_library_root_logger()` — [`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L74)
- `_get_default_logging_level()` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L50) — If DIFFUSERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
- `_get_library_name()` — [`L66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L66)
- `_get_library_root_logger()` — [`L70`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L70)
- `_reset_library_root_logger()` — [`L91`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L91)
- `add_handler(handler: logging.Handler)` — [`L199`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L199) — adds a handler to the HuggingFace Diffusers' root logger.
- `disable_default_handler()` — [`L181`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L181) — Disable the default handler of the 🤗 Diffusers' root logger.
- `disable_progress_bar()` — [`L334`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L334) — Disable tqdm progress bar.
- `disable_propagation()` — [`L217`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L217) — Disable propagation of the library log outputs. Note that log propagation is disabled by default.
- `enable_default_handler()` — [`L190`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L190) — Enable the default handler of the 🤗 Diffusers' root logger.
- `enable_explicit_format()` — [`L236`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L236) — Enable explicit formatting for every 🤗 Diffusers' logger. The explicit formatter is as follows:
- `enable_progress_bar()` — [`L328`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L328) — Enable tqdm progress bar.
- `enable_propagation()` — [`L226`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L226) — Enable propagation of the library log outputs. Please disable the HuggingFace Diffusers' default handler to prevent
- `get_log_levels_dict()` — [`L104`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L104)
- `get_logger(name: Optional[str] = None)` — [`L108`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L108) — Return a logger with the specified name.
- `get_verbosity()` — [`L122`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L122) — Return the current level for the 🤗 Diffusers' root logger as an `int`.
- `is_progress_bar_enabled()` — [`L322`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L322) — Return a boolean indicating whether tqdm progress bars are enabled.
- `remove_handler(handler: logging.Handler)` — [`L208`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L208) — removes given handler from the HuggingFace Diffusers' root logger.
- `reset_format()` — [`L251`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L251) — Resets the formatting for 🤗 Diffusers' loggers.
- `set_verbosity(verbosity: int)` — [`L142`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L142) — Set the verbosity level for the 🤗 Diffusers' root logger.
- `set_verbosity_debug()` — [`L171`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L171) — Set the verbosity to the `DEBUG` level.
- `set_verbosity_error()` — [`L176`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L176) — Set the verbosity to the `ERROR` level.
- `set_verbosity_info()` — [`L161`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L161) — Set the verbosity to the `INFO` level.
- `set_verbosity_warning()` — [`L166`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L166) — Set the verbosity to the `WARNING` level.
- `warning_advice(self, *args, **kwargs)` — [`L263`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L263) — This method is identical to `logger.warning()`, but if env var DIFFUSERS_NO_ADVISORY_WARNINGS=1 is set, this

## Module values
- `_default_handler` — [`L35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L35)
- `_default_log_level` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L45)
- `_lock` — [`L34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L34)
- `_tqdm_active` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L47)
- `log_levels` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L37)
- `tqdm` — [`L319`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/logging.py#L319)

