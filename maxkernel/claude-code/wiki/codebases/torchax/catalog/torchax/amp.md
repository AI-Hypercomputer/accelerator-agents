---
title: 'Module: torchax/amp.py'
type: catalog
provenance: extracted
module: torchax/amp.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.amp`/
symbols:
  CastPolicy: CastPolicy#
  CastPolicy.FP32: CastPolicy#FP32.
  CastPolicy.LOWER_PRECISION_FP: CastPolicy#LOWER_PRECISION_FP.
  execute_policy: execute_policy().
  autocast_policy: autocast_policy.
  CastPolicy.PROMOTE: CastPolicy#PROMOTE.
  autocast: autocast().
  execute_policy.is_float: execute_policy().is_float().
  CastPolicy.FP32_SET_OPT_DTYPE: CastPolicy#FP32_SET_OPT_DTYPE.
  CastPolicy.FP32_APPEND_DTYPE: CastPolicy#FP32_APPEND_DTYPE.
---
# Module: [`torchax/amp.py`](../../../../../raw/code/torchax/torchax/amp.py)

## Classes
### `CastPolicy`  ·  implements/extends Enum
- def: [`torchax/amp.py:45`](../../../../../raw/code/torchax/torchax/amp.py#L45)
- signature: `class CastPolicy(enum.Enum):`
- members:
  - `FP32` — [`L47`](../../../../../raw/code/torchax/torchax/amp.py#L47)
  - `FP32_APPEND_DTYPE` — [`L49`](../../../../../raw/code/torchax/torchax/amp.py#L49)
  - `FP32_SET_OPT_DTYPE` — [`L48`](../../../../../raw/code/torchax/torchax/amp.py#L48)
  - `LOWER_PRECISION_FP` — [`L46`](../../../../../raw/code/torchax/torchax/amp.py#L46)
  - `PROMOTE` — [`L50`](../../../../../raw/code/torchax/torchax/amp.py#L50)
- used by: [`execute_policy`](amp.md#execute_policy), [`autocast_policy`](amp.md#autocast_policy)

## Functions
- `autocast(device, dtype=torch.bfloat16, env=None)` — [`L75`](../../../../../raw/code/torchax/torchax/amp.py#L75)
- `execute_policy(policy, args, kwargs, target_lower_fp)` — [`L53`](../../../../../raw/code/torchax/torchax/amp.py#L53)
- `is_float(a)` — [`L54`](../../../../../raw/code/torchax/torchax/amp.py#L54)

## Module values
- `autocast_policy` — [`L86`](../../../../../raw/code/torchax/torchax/amp.py#L86)

