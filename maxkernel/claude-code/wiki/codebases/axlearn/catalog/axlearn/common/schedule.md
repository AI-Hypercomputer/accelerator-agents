---
title: 'Module: axlearn/common/schedule.py'
type: catalog
provenance: extracted
module: axlearn/common/schedule.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.schedule`/
symbols:
  Schedule: Schedule.
  ScheduleFn: ScheduleFn.
  as_schedule_fn: as_schedule_fn().
  polynomial: polynomial().
  segment_wise: segment_wise().
  stepwise: stepwise().
  cosine_with_linear_warmup: cosine_with_linear_warmup().
  warmup_stable_decay: warmup_stable_decay().
  constant_with_linear_warmup: constant_with_linear_warmup().
  linear_schedule_with_warmup: linear_schedule_with_warmup().
  exponential: exponential().
  adafactor_decay_rate: adafactor_decay_rate().
  adafactor: adafactor().
  decay_bias_correction: decay_bias_correction().
  ema_schedule: ema_schedule().
  polynomial.fn: polynomial().fn().
  exponential.fn: exponential().fn().
  stepwise.fn: stepwise().fn().
  segment_wise.fn: segment_wise().fn().
  adafactor.fn: adafactor().fn().
  adafactor_decay_rate.fn: adafactor_decay_rate().fn().
  decay_bias_correction.fn: decay_bias_correction().fn().
  ema_schedule.fn: ema_schedule().fn().
  inverse_sqrt: inverse_sqrt().
---
# Module: [`axlearn/common/schedule.py`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py)

## Functions
- `adafactor(scale: float = 1, *, warmup_steps: int = 10000, decay_power: float = -0.5, step_offset: int = 0)` — [`L135`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L135) — Transformer learning rate scheduler.
- `adafactor_decay_rate(c: float = 0.8, step_offset: int = 0)` — [`L159`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L159) — Returns the beta2 schedule described in section 7.2 of https://arxiv.org/abs/1804.04235.
- `as_schedule_fn(s: Optional[Schedule])` — [`L25`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L25)
- `constant_with_linear_warmup(peak_lr: float, *, warmup_steps: int = 500, begin_value: float = 0)` — [`L412`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L412) — Constant learning rate schedule with linear warm-up.
- `cosine_with_linear_warmup(peak_lr: float, *, max_step: int, warmup_steps: int = 500, begin_value: float = 0, alpha: float = 0, decay_begin_step: Optional[int] = None)` — [`L301`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L301) — Cosine learning rate schedule with linear warm-up.
- `decay_bias_correction(decay: float)` — [`L184`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L184) — Returns a ScheduleFn that applies bias correction to the given decay.
- `ema_schedule(decay: float = 0.9999, *, warmup_steps: int = 1, step_offset: int = 0)` — [`L481`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L481) — Ema decay schedule with warm-up.
- `exponential(*, begin_step: int = 0, begin_value: float = 0, end_step: int = 1, end_value: float = 0)` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L75) — An exponential schedule.
- `fn(step: Tensor)` — [`L67`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L67)
- `fn(step: Tensor)` — [`L111`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L111)
- `fn(step)` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L150)
- `fn(step)` — [`L177`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L177)
- `fn(step)` — [`L204`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L204)
- `fn(step: Tensor)` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L240)
- `fn(step: Tensor)` — [`L287`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L287)
- `fn(step)` — [`L503`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L503)
- `inverse_sqrt(step: int, warmup_steps: int = 1)` — [`L117`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L117) — Inverse sqrt schedule optionally with constant warmup, as seen in T5.
- `linear_schedule_with_warmup(peak_lr: float, *, max_step: int, warmup_steps: int, begin_value: float = 0, end_value: float = 0)` — [`L444`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L444) — Learning rate schedule with linear warm-up.
- `polynomial(*, begin_step: int = 0, begin_value: float = 0, end_step: int = 1, end_value: float = 0, power: float = 1)` — [`L36`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L36) — A polynomial (linear when power=1) schedule.
- `segment_wise(segments: list[Schedule], *, segment_steps: list[int])` — [`L254`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L254) — A composite schedule consisting of multiple segments, each with its own schedule.
- `stepwise(sub: list[Schedule], start_step: list[int])` — [`L211`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L211) — A composite schedule consisting of multiple sub-schedules.
- `warmup_stable_decay(peak_lr: float, *, max_step: int, decay_begin_step: int, warmup_steps: int = 500, begin_value: float = 0, alpha: float = 0)` — [`L357`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L357) — Warmup stable decay (WSD) learning rate schedule. Linear warmup + constant lr + linear decay.

## Module values
- `Schedule` — [`L22`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L22)
- `ScheduleFn` — [`L21`](../../../../../../raw/code/axlearn/axlearn/common/schedule.py#L21)

