---
title: 'Module: axlearn/common/traceback_util.py'
type: catalog
provenance: extracted
module: axlearn/common/traceback_util.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.traceback_util`/
symbols:
  no_stack_summary: no_stack_summary().
  annotate_stack: annotate_stack().
  wrap: wrap().
  _ipython: _ipython.
  _InContextException._format_summary: _InContextException#_format_summary().
  _InContextException.wrap: _InContextException#wrap().
  _InContextException.in_context_exception_wrapper: _InContextException#in_context_exception_wrapper().
  annotate_stack.decorator: annotate_stack().decorator().
  _is_annotation_frame: _is_annotation_frame().
  _walk_annotated_tb: _walk_annotated_tb().
  _ipython_exception_handler: _ipython_exception_handler().
  is_stack_summary_enabled: is_stack_summary_enabled().
  _InContextException.fqname: _InContextException#fqname().
  _InContextException.__str__: _InContextException#__str__().
  _excepthook: _excepthook().
  annotate_stack.decorator.stack_annotation_wrapper: annotate_stack().decorator().stack_annotation_wrapper().
  _InContextException: _InContextException#
---
# Module: [`axlearn/common/traceback_util.py`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py)

## Classes
### `_InContextException`
- def: [`axlearn/common/traceback_util.py:114`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L114)
- members:
  - `_format_summary(self)` — [`L211`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L211) — Returns a formatted stack summary.
  - `fqname(cls)` — [`L220`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L220)
  - `in_context_exception_wrapper(*args, **kwargs)` — [`L166`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L166)
  - `wrap(cls, fn: Callable)` — [`L147`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L147) — Wraps the given function so that if an Exception `e` is encountered, it has an instance
- protocol/private: `__str__`[`L200`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L200)
- uses (calls/refs, reference-scoped): [`no_stack_summary`](traceback_util.md#no_stack_summary), [`_walk_annotated_tb`](traceback_util.md#_walk_annotated_tb)
- used by: [`wrap`](traceback_util.md#wrap)

## Functions
- `_excepthook(type: type[BaseException], value: BaseException, traceback: types.TracebackType, old_excepthook: Optional[Callable] = sys.excepthook)` — [`L353`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L353) — Calls the original `sys.excepthook` with `value`. If `value` has a `_stack_summary`
- `_ipython_exception_handler(self, etype, evalue, tb, tb_offset=None)` — [`L389`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L389)
- `_is_annotation_frame(frame: types.FrameType)` — [`L301`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L301) — Returns whether the given frame is an annotation frame.
- `_walk_annotated_tb(tb: types.TracebackType)` — [`L314`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L314) — Similar to `traceback.walk_tb`, except that annotation frames are elided
- `annotate_stack(**aux)` — [`L267`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L267) — A decorator that creates a wrapper around `fn` that annotates the call with annotations given
- `decorator(fn: Callable)` — [`L289`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L289)
- `is_stack_summary_enabled()` — [`L72`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L72) — Check AXLEARN_ENABLE_STACK_SUMMARY env variable enabled.
- `no_stack_summary(fn: Callable)` — [`L246`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L246) — A decorator that wraps `fn` so that calls to it are excluded from the stack summary generated
- `stack_annotation_wrapper(*args, **kwargs)` — [`L291`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L291)
- `wrap(fn: Callable)` — [`L90`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L90) — Wraps the given function so that if an Exception `e` is encountered, it has an Exception

## Module values
- `_ipython` — [`L397`](../../../../../../raw/code/axlearn/axlearn/common/traceback_util.py#L397)

