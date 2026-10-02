---
title: 'Module: src/maxtext/utils/max_logging.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/max_logging.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.max_logging`/
symbols:
  log: log().
  debug: debug().
  info: info().
  warning: warning().
  error: error().
  NoisyLogFilter: NoisyLogFilter#
  NoisyLogFilter.filter: NoisyLogFilter#filter().
---
# Module: [`src/maxtext/utils/max_logging.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py)

## Classes
### `NoisyLogFilter`  ·  implements/extends Filter
- def: [`src/maxtext/utils/max_logging.py:47`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L47)
- doc: Class for defining log patterns to filter out
- signature: `class NoisyLogFilter(std_logging.Filter):`
- members:
  - `filter(self, record)` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L52)

## Functions
- `debug(user_str)` — [`L26`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L26) — Logs a message at the DEBUG level.
- `error(user_str)` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L41) — Logs a message at the ERROR level.
- `info(user_str, stacklevel=2)` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L31) — Logs a message at the INFO level.
- `log(user_str)` — [`L20`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L20) — Logs a message at the INFO level.
- `warning(user_str)` — [`L36`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_logging.py#L36) — Logs a message at the WARNING level.

