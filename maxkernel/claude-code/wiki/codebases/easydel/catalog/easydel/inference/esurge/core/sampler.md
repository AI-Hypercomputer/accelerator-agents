---
title: 'Module: easydel/inference/esurge/core/sampler.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/sampler.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.sampler`/
symbols:
  _regular_sample: _regular_sample().
  sample_tokens: sample_tokens().
  _regular_sample.apply_topk: _regular_sample().apply_topk().
  _regular_sample.apply_topp: _regular_sample().apply_topp().
  apply_history_penalties_from_counts: apply_history_penalties_from_counts().
  apply_history_penalties: apply_history_penalties().
  _regular_sample.apply_topk.topk_per_sample: _regular_sample().apply_topk().topk_per_sample().
  _regular_sample.apply_topp.topp_per_sample: _regular_sample().apply_topp().topp_per_sample().
  _regular_sample.apply_temp: _regular_sample().apply_temp().
  build_history_token_counts: build_history_token_counts().
  update_token_counts: update_token_counts().
  _greedy_sample: _greedy_sample().
  _regular_sample.sample_one: _regular_sample().sample_one().
  apply_history_penalties_from_counts._apply: apply_history_penalties_from_counts()._apply().
---
# Module: [`easydel/inference/esurge/core/sampler.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py)

## Functions
- `_apply(legi: jax.Array)` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L207)
- `_greedy_sample(logits: jax.Array)` — [`L43`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L43) — Greedy sampling: select token with highest probability.
- `_regular_sample(logits: jax.Array, sampling_metadata: SamplingMetadata, rng: jax.Array)` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L55) — Stochastic sampling with top-k, top-p, and min-p filtering.
- `apply_history_penalties(logits: jax.Array, *, token_history: jax.Array, seq_lens: jax.Array, active_mask: jax.Array, presence_penalties: jax.Array, frequency_penalties: jax.Array, repetition_penalties: jax.Array)` — [`L255`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L255) — Apply history-aware penalties to logits.
- `apply_history_penalties_from_counts(logits: jax.Array, *, token_counts: jax.Array, active_mask: jax.Array, presence_penalties: jax.Array, frequency_penalties: jax.Array, repetition_penalties: jax.Array)` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L175) — Apply history penalties from precomputed token counts.
- `apply_temp(legi)` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L111) — Scale logits by per-sample temperature values.
- `apply_topk(legi)` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L83) — Apply top-k filtering to each sample in the batch.
- `apply_topp(legi)` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L97) — Apply top-p (nucleus) filtering to each sample in the batch.
- `build_history_token_counts(*, token_history: jax.Array, seq_lens: jax.Array, active_mask: jax.Array, vocab_size: int, count_dtype: jnp.dtype = jnp.uint32)` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L142) — Build exact per-token history counts from token IDs.
- `sample_one(logits_i, seed_i)` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L133) — Sample a single token from a filtered logit distribution.
- `sample_tokens(logits: jax.Array, sampling_metadata: SamplingMetadata, rng: jax.Array)` — [`L304`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L304) — Sample next tokens from logits with advanced filtering.
- `topk_per_sample(logits_i, k_i)` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L86) — Apply top-k mask to a single sample if k > 0.
- `topp_per_sample(logits_i, p_i)` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L100) — Apply top-p mask to a single sample if p < 1.0.
- `update_token_counts(token_counts: jax.Array, *, row_indices: jax.Array, sampled_tokens: jax.Array, valid_mask: jax.Array)` — [`L228`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampler.py#L228) — Increment exact token counts with newly sampled tokens.

