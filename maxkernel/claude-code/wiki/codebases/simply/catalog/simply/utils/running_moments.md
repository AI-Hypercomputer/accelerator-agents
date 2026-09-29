---
title: 'Module: simply/utils/running_moments.py'
type: catalog
provenance: extracted
module: simply/utils/running_moments.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.running_moments`/
symbols:
  Scalar: Scalar.
  RunningMoments.mean: RunningMoments#mean().
  RunningMoments.var: RunningMoments#var().
  update: update().
  RunningMoments.m0: RunningMoments#m0.
  RunningMoments.count: RunningMoments#count().
  RunningMoments.std: RunningMoments#std().
  update.welford_fn: update().welford_fn().
  update.ema_fn: update().ema_fn().
  RunningMoments.m1: RunningMoments#m1.
  RunningMoments.m2: RunningMoments#m2.
  RunningMoments: RunningMoments#
  RunningMoments.momentum: RunningMoments#momentum.
---
# Module: [`simply/utils/running_moments.py`](../../../../../../raw/code/simply/simply/utils/running_moments.py)

## Classes
### `RunningMoments`  ·  implements/extends NamedTuple
- def: [`simply/utils/running_moments.py:25`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L25)
- doc: The running mean and var of a stream of value.
- signature: `class RunningMoments(NamedTuple):`
- members:
  - `count(self)` — [`L34`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L34)
  - `mean(self)` — [`L38`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L38)
  - `std(self)` — [`L54`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L54)
  - `var(self)` — [`L46`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L46)
  - `m0` — [`L29`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L29)
  - `m1` — [`L30`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L30)
  - `m2` — [`L31`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L31)
  - `momentum` — [`L28`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L28)
- uses (calls/refs, reference-scoped): [`Scalar`](running_moments.md#Scalar)
- used by: [`test_update`](running_moments_test.md#RunningMomentsTest.test_update), [`update`](running_moments.md#update), [`ema_fn`](running_moments.md#update.ema_fn), [`welford_fn`](running_moments.md#update.welford_fn)

## Functions
- `ema_fn(m0: Scalar, m1: Scalar, m2: Scalar, x: Scalar)` — [`L75`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L75)
- `update(running_moments: RunningMoments, x: Scalar)` — [`L58`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L58) — Update running_moments by adding x and returns the new running_moments.
- `welford_fn(m0: Scalar, m1: Scalar, m2: Scalar, x: Scalar)` — [`L64`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L64)

## Module values
- `Scalar` — [`L22`](../../../../../../raw/code/simply/simply/utils/running_moments.py#L22)

