---
title: 'Module: jax/_src/traceback_util.py'
type: catalog
provenance: extracted
module: jax/_src/traceback_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.traceback_util`/
symbols:
  api_boundary: api_boundary().
  register_exclusion: register_exclusion().
  api_boundary.reraise_with_filtered_traceback: api_boundary().reraise_with_filtered_traceback().
  filter_traceback: filter_traceback().
  _filtering_mode: _filtering_mode().
  include_filename: include_filename().
  _add_tracebackhide_to_hidden_frames: _add_tracebackhide_to_hidden_frames().
  _add_call_stack_frames: _add_call_stack_frames().
  include_frame: include_frame().
  _is_under_reraiser: _is_under_reraiser().
  repro_is_enabled: repro_is_enabled.
  C: C.
  SimplifiedTraceback.__str__: SimplifiedTraceback#__str__().
  _is_reraiser_frame: _is_reraiser_frame().
  format_exception_only: format_exception_only().
  _simplified_tb_msg: _simplified_tb_msg.
  SimplifiedTraceback: SimplifiedTraceback#
  _exclude_paths._exclude_paths: _exclude_paths._exclude_paths.
  _jax_message_append: _jax_message_append.
  _path_starts_with: _path_starts_with().
  _ignore_known_hidden_frame: _ignore_known_hidden_frame().
  UnfilteredStackTrace: UnfilteredStackTrace#
  _running_under_ipython: _running_under_ipython().
  _ipython_supports_tracebackhide: _ipython_supports_tracebackhide().
---
# Module: [`jax/_src/traceback_util.py`](../../../../../../raw/code/jax/jax/_src/traceback_util.py)

## Classes
### `SimplifiedTraceback`  ·  implements/extends Exception
- def: [`jax/_src/traceback_util.py:135`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L135)
- signature: `class SimplifiedTraceback(Exception):`
- protocol/private: `__str__`[`L136`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L136)
- uses (calls/refs, reference-scoped): [`_simplified_tb_msg`](traceback_util.md#_simplified_tb_msg)

### `UnfilteredStackTrace`  ·  implements/extends Exception
- def: [`jax/_src/traceback_util.py:129`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L129)
- signature: `class UnfilteredStackTrace(Exception):`
- used by: [`reraise_with_filtered_traceback`](traceback_util.md#api_boundary.reraise_with_filtered_traceback)

## Functions
- `_add_call_stack_frames(tb: types.TracebackType)` — [`L89`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L89)
- `_add_tracebackhide_to_hidden_frames(tb: types.TracebackType | None)` — [`L73`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L73)
- `_filtering_mode()` — [`L154`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L154)
- `_ignore_known_hidden_frame(f: types.FrameType)` — [`L70`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L70)
- `_ipython_supports_tracebackhide()` — [`L149`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L149) — Returns true if the IPython version supports __tracebackhide__.
- `_is_reraiser_frame(f: traceback.FrameSummary | types.FrameType)` — [`L113`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L113)
- `_is_under_reraiser(e: BaseException)` — [`L120`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L120)
- `_path_starts_with(path: str, path_prefix: str)` — [`L45`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L45)
- `_running_under_ipython()` — [`L141`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L141) — Returns true if we appear to be in an IPython session.
- `api_boundary(fun: C, *, repro_api_name: str | None = None, repro_user_func: bool = False)` — [`L163`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L163) — Wraps ``fun`` to form a boundary for filtering exception tracebacks.
- `filter_traceback(tb: types.TracebackType)` — [`L80`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L80)
- `format_exception_only(e: BaseException)` — [`L126`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L126)
- `include_filename(filename: str)` — [`L64`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L64)
- `include_frame(f: types.FrameType)` — [`L61`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L61)
- `register_exclusion(path: str)` — [`L33`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L33)
- `reraise_with_filtered_traceback(*args, **kwargs)` — [`L192`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L192)

## Module values
- `C` — [`L29`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L29)
- `_exclude_paths` — [`L31`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L31)
- `_jax_message_append` — [`L40`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L40)
- `_simplified_tb_msg` — [`L131`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L131)
- `repro_is_enabled` — [`L237`](../../../../../../raw/code/jax/jax/_src/traceback_util.py#L237)

