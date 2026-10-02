---
title: 'Module: src/maxtext/common/goodput.py'
type: catalog
provenance: extracted
module: src/maxtext/common/goodput.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.goodput`/
symbols:
  RECORD_JOB_START_TIME: RECORD_JOB_START_TIME.
  RECORD_JOB_END_TIME: RECORD_JOB_END_TIME.
  maybe_monitor_goodput: maybe_monitor_goodput().
  create_goodput_recorder: create_goodput_recorder().
  maybe_record_goodput: maybe_record_goodput().
  GoodputEvent: GoodputEvent#
  GoodputEvent.JOB: GoodputEvent#JOB.
  record_goodput: record_goodput().
  monitoring: monitoring.
  _GOODPUT_STUB: _GOODPUT_STUB.
  goodput: goodput.
  GoodputEvent.TPU_INIT: GoodputEvent#TPU_INIT.
  GoodputEvent.TRAINING_PREPARATION: GoodputEvent#TRAINING_PREPARATION.
  GoodputEvent.DATA_LOADING: GoodputEvent#DATA_LOADING.
  GoodputEvent.STEP: GoodputEvent#STEP.
---
# Module: [`src/maxtext/common/goodput.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py)

## Classes
### `GoodputEvent`  ·  implements/extends Enum
- def: [`src/maxtext/common/goodput.py:31`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L31)
- signature: `class GoodputEvent(Enum):`
- members:
  - `DATA_LOADING` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L35)
  - `JOB` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L32)
  - `STEP` — [`L36`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L36)
  - `TPU_INIT` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L33)
  - `TRAINING_PREPARATION` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L34)
- used by: [`RECORD_JOB_END_TIME`](goodput.md#RECORD_JOB_END_TIME), [`RECORD_JOB_START_TIME`](goodput.md#RECORD_JOB_START_TIME)

## Functions
- `create_goodput_recorder(config)` — [`L118`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L118) — Create goodput recorder if `enable_goodput_recording=True`.
- `maybe_monitor_goodput(config)` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L46) — Monitor cumulative goodput if enabled on the lead host.
- `maybe_record_goodput(recorder, event_name, *args)` — [`L91`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L91) — Record goodput if `enable_goodput_recording=True`.
- `record_goodput(recorder, event_name, *args)` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L110) — Record goodput to cloud logging.

## Module values
- `RECORD_JOB_END_TIME` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L42)
- `RECORD_JOB_START_TIME` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L41)
- `_GOODPUT_STUB` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L28)
- `goodput` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L28)
- `monitoring` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/common/goodput.py#L28)

