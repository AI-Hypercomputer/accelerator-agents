---
title: 'Module: tpu_inference/layers/jax/sample/rejection_sampler.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/sample/rejection_sampler.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.sample.rejection_sampler`/
symbols:
  RejectionSampler.forward: RejectionSampler#forward().
  RejectionSampler._forward: RejectionSampler#_forward().
  _compute_probs: _compute_probs().
  RejectionSampler._spec: RejectionSampler#_spec().
  _random_rejection_sample_with_segment: _random_rejection_sample_with_segment().
  PLACEHOLDER_TOKEN_ID: PLACEHOLDER_TOKEN_ID.
  rejection_sample: rejection_sample().
  _greedy_rejection_sample_with_segment: _greedy_rejection_sample_with_segment().
  RejectionSampler.__call__: RejectionSampler#__call__().
  RejectionSampler: RejectionSampler#
  _get_segment_info: _get_segment_info().
  RejectionSampler.parse_output: RejectionSampler#parse_output().
  RejectionSampler.mesh: RejectionSampler#mesh.
  _sample_recovered_tokens: _sample_recovered_tokens().
  GREEDY_TEMPERATURE: GREEDY_TEMPERATURE.
  RejectionSampler.__init__: RejectionSampler#__init__().
---
# Module: [`tpu_inference/layers/jax/sample/rejection_sampler.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py)

## Classes
### `RejectionSampler`
- def: [`tpu_inference/layers/jax/sample/rejection_sampler.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L38)
- doc: JAX-based rejection sampler for speculative decoding.
- signature: `class RejectionSampler:`
- members:
  - `__call__(self, draft_token_ids: jnp.ndarray, num_draft_tokens: jnp.ndarray, draft_probs: Optional[jnp.ndarray], target_logits: jnp.ndarray, bonus_token_ids: jnp.ndarray, sampling_metadata: TPUSupportedSamplingMetadata, key: Optional[jax.random.PRNGKey] = None)` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L49) — Perform rejection sampling on draft tokens with flattened inputs.
  - `forward(self, draft_token_ids: jnp.ndarray, num_draft_tokens: jnp.ndarray, draft_probs: Optional[jnp.ndarray], target_logits: jnp.ndarray, bonus_token_ids: jnp.ndarray, sampling_metadata: TPUSupportedSamplingMetadata, key: Optional[jax.random.PRNGKey] = None)` — [`L90`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L90) — Perform rejection sampling on draft tokens with flattened inputs.
  - `parse_output(output_token_ids: jnp.ndarray, vocab_size: int, num_draft_tokens_cpu: np.ndarray, batch_size: int, padded_tokens_length: int, dp_size: int, req_indices_dp: dict)` — [`L202`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L202) — Parse the output of the rejection sampler.
  - `mesh` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L47)
- protocol/private: `__init__`[`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L46), `_forward`[`L124`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L124), `_spec`[`L166`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L166)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`TPUSupportedSamplingMetadata`](sampling_metadata.md#TPUSupportedSamplingMetadata), [`_compute_probs`](rejection_sampler.md#_compute_probs), [`do_sampling`](sampling_metadata.md#TPUSupportedSamplingMetadata.do_sampling), [`temperature`](sampling_metadata.md#TPUSupportedSamplingMetadata.temperature), [`top_k`](sampling_metadata.md#TPUSupportedSamplingMetadata.top_k), [`top_p`](sampling_metadata.md#TPUSupportedSamplingMetadata.top_p), [`_cache_collision_dummy`](sampling_metadata.md#TPUSupportedSamplingMetadata._cache_collision_dummy), [`logprobs`](sampling_metadata.md#TPUSupportedSamplingMetadata.logprobs), [`PLACEHOLDER_TOKEN_ID`](rejection_sampler.md#PLACEHOLDER_TOKEN_ID), [`rejection_sample`](rejection_sampler.md#rejection_sample)
- used by: `rejection_sampler`

## Functions
- `_compute_probs(logits: jnp.ndarray, num_draft_tokens: jnp.ndarray, sampling_metadata: TPUSupportedSamplingMetadata)` — [`L284`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L284) — Apply top-k, top-p, and temperature to logits and compute probabilities.
- `_get_segment_info(num_draft_tokens: jax.Array, total_tokens: int)` — [`L320`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L320) — Helper to create segment IDs and per-segment indices.
- `_greedy_rejection_sample_with_segment(draft_token_ids: jax.Array, target_probs: jax.Array, num_draft_tokens: jax.Array, bonus_token_ids: jax.Array)` — [`L513`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L513) — Performs greedy speculative decoding validation in a vectorized, jittable manner.
- `_random_rejection_sample_with_segment(draft_token_ids: jax.Array, draft_probs: Optional[jax.Array], target_probs: jax.Array, num_draft_tokens: jax.Array, bonus_token_ids: jax.Array, key: jax.random.PRNGKey)` — [`L422`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L422) — Performs random speculative decoding validation in a vectorized, jittable manner.
- `_sample_recovered_tokens(draft_token_ids: jax.Array, draft_probs: Optional[jax.Array], target_probs: jax.Array, key: jax.random.PRNGKey)` — [`L341`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L341) — Sample recovered tokens using the Gumbel-Max trick.
- `rejection_sample(draft_token_ids: jnp.ndarray, num_draft_tokens: jnp.ndarray, draft_probs: Optional[jnp.ndarray], target_probs: jnp.ndarray, bonus_token_ids: jnp.ndarray, do_sampling: bool, key: Optional[jax.random.PRNGKey] = None)` — [`L371`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L371) — Perform rejection sampling on draft tokens with flattened inputs.

## Module values
- `GREEDY_TEMPERATURE` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L35)
- `PLACEHOLDER_TOKEN_ID` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/rejection_sampler.py#L34)

