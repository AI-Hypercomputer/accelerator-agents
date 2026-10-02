---
title: 'Module: qwix/_src/averaging.py'
type: catalog
provenance: extracted
module: qwix/_src/averaging.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.averaging`/
symbols:
  SimpleMovingAverage.get_calibration: SimpleMovingAverage#get_calibration().
  SimpleMovingAverage.init: SimpleMovingAverage#init().
  SimpleMovingAverage.update: SimpleMovingAverage#update().
  SimpleMovingAverage: SimpleMovingAverage#
  Calibration.Calibration: Calibration.Calibration.
  QuantStat.QuantStat: QuantStat.QuantStat.
  SimpleMovingAverage._bootstrap_steps: SimpleMovingAverage#_bootstrap_steps.
  SimpleMovingAverage.__init__: SimpleMovingAverage#__init__().
---
# Module: [`qwix/_src/averaging.py`](../../../../../../raw/code/qwix/qwix/_src/averaging.py)

## Classes
### `SimpleMovingAverage`
- def: [`qwix/_src/averaging.py:25`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L25) — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
- doc: Simple moving average maintains a count and sums of the calibration.
- signature: `class SimpleMovingAverage:`
- members:
  - `__init__(self, bootstrap_steps: int = 0)` — [`L31`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L31) — Initializes the simple moving average.
  - `get_calibration(self, quant_stat: QuantStat, default_calibration: Calibration | None = None)` — [`L66`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L66) — Returns the average calibration. — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
  - `init(self, calibration: Calibration)` — [`L40`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L40) — Initializes the quantization statistics from the calibration. — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
  - `update(self, quant_stat: QuantStat, calibration: Calibration)` — [`L50`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L50) — Updates the quantization statistics. — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
- protocol/private: `_bootstrap_steps`[`L38`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L38)
- uses (calls/refs, reference-scoped): [`Calibration`](averaging.md#Calibration.Calibration), [`QuantStat`](averaging.md#QuantStat.QuantStat)
- used by: [`quantize_params`](providers/ptq.md#quantize_params), [`extract_calibrated_quant_context`](../contrib/calibration.md#extract_calibrated_quant_context), [`quantize_params`](../contrib/smooth_quant.md#quantize_params), [`_update_and_get_quant_stat`](providers/qt.md#QtProvider._update_and_get_quant_stat), [`_compute_static_scale_zero_point`](providers/odml.md#OdmlConversionProvider._compute_static_scale_zero_point), [`_update_and_get_quant_stat`](providers/odml.md#OdmlQatProvider._update_and_get_quant_stat), [`init`](providers/ptq.md#quantize_act.init), [`_collect_stats`](../contrib/calibration.md#SinglePassCalibrationProvider._collect_stats), [`_update_flat_stats_with_moving_average`](../contrib/qep.md#_update_flat_stats_with_moving_average)  (8 test-only)

## Module values
- `Calibration` — [`L22`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L22) — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
- `QuantStat` — [`L21`](../../../../../../raw/code/qwix/qwix/_src/averaging.py#L21)

