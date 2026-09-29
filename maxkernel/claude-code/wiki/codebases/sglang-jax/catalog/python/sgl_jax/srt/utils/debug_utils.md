---
title: 'Module: python/sgl_jax/srt/utils/debug_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/utils/debug_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.utils.debug_utils`/
symbols:
  log_shardings.decorator: log_shardings().decorator().
  print_parameter_shardings: print_parameter_shardings().
  FRAMEWORK_LOG_LEVEL: FRAMEWORK_LOG_LEVEL.
  FrameworkLogLevel: FrameworkLogLevel#
  log_shardings: log_shardings().
  FrameworkLogLevel.DEBUG: FrameworkLogLevel#DEBUG.
  log_shardings.decorator.wrapper: log_shardings().decorator().wrapper().
  FrameworkLogLevel.ERROR: FrameworkLogLevel#ERROR.
  FrameworkLogLevel.WARN: FrameworkLogLevel#WARN.
  FrameworkLogLevel.INFO: FrameworkLogLevel#INFO.
  FrameworkLogLevel.TRACE: FrameworkLogLevel#TRACE.
---
# Module: [`python/sgl_jax/srt/utils/debug_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py)

## Classes
### `FrameworkLogLevel`  ·  implements/extends IntEnum
- def: [`python/sgl_jax/srt/utils/debug_utils.py:6`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L6)
- signature: `class FrameworkLogLevel(IntEnum):`
- members:
  - `DEBUG` — [`L10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L10)
  - `ERROR` — [`L7`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L7)
  - `INFO` — [`L9`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L9)
  - `TRACE` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L11)
  - `WARN` — [`L8`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L8)
- used by: [`decorator`](debug_utils.md#log_shardings.decorator), [`print_parameter_shardings`](debug_utils.md#print_parameter_shardings), [`FRAMEWORK_LOG_LEVEL`](debug_utils.md#FRAMEWORK_LOG_LEVEL)

## Functions
- `decorator(fn)` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L25)
- `log_shardings(name)` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L24)
- `print_parameter_shardings(model)` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L17)
- `wrapper(*args, **kwargs)` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L30)

## Module values
- `FRAMEWORK_LOG_LEVEL` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/debug_utils.py#L14)

