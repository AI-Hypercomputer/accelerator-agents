---
title: 'Module: python/sgl_jax/srt/managers/dp_schedule_policy.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/dp_schedule_policy.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.dp_schedule_policy`/
symbols:
  pick_cache_aware_dp: pick_cache_aware_dp().
  pick_cache_aware_dp.least_loaded: pick_cache_aware_dp().least_loaded().
  req_prefix_match_key: req_prefix_match_key().
  BALANCE_ABS: BALANCE_ABS.
  BALANCE_REL: BALANCE_REL.
  CACHE_THRESHOLD: CACHE_THRESHOLD.
---
# Module: [`python/sgl_jax/srt/managers/dp_schedule_policy.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py)

## Functions
- `least_loaded(ranks: list[int])` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py#L87)
- `pick_cache_aware_dp(eligible: list[int], counts: list[int], token_counts: list[int], matches: dict[int, int], prompt_len: int)` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py#L64) — Cache-affinity DP policy with soft load balancing.
- `req_prefix_match_key(req)` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py#L20) — Effective ``(token_ids, extra_key)`` for a cache-affinity prefix probe.

## Module values
- `BALANCE_ABS` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py#L15)
- `BALANCE_REL` — [`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py#L16)
- `CACHE_THRESHOLD` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/dp_schedule_policy.py#L17)

