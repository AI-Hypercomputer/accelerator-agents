---
title: 'Module: tpu_inference/layers/jax/sample/sampling.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/sample/sampling.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.sample.sampling`/
symbols:
  compute_prompt_logprobs: compute_prompt_logprobs().
  sample: sample().
  _apply_sampling_transforms: _apply_sampling_transforms().
  compute_and_gather_logprobs: compute_and_gather_logprobs().
  PromptLogprobsAsyncData: PromptLogprobsAsyncData#
  PromptLogprobsReqSnap.req_state: PromptLogprobsReqSnap#req_state.
  PromptLogprobsReqSnap: PromptLogprobsReqSnap#
  PromptLogprobsAsyncData.tensors: PromptLogprobsAsyncData#tensors.
  PromptLogprobsAsyncData.req_snaps: PromptLogprobsAsyncData#req_snaps.
  compute_and_gather_prompt_logprobs: compute_and_gather_prompt_logprobs().
  PromptLogprobsReqSnap.req_id: PromptLogprobsReqSnap#req_id.
  PromptLogprobsReqSnap.num_logits: PromptLogprobsReqSnap#num_logits.
  _jax_logprobs_copy_to_host_async: _jax_logprobs_copy_to_host_async().
  PromptLogprobsReqSnap.req_offset: PromptLogprobsReqSnap#req_offset.
  PromptLogprobsReqSnap.start_idx: PromptLogprobsReqSnap#start_idx.
  PromptLogprobsReqSnap.is_last_chunk: PromptLogprobsReqSnap#is_last_chunk.
  PromptLogprobsReqSnap.num_k: PromptLogprobsReqSnap#num_k.
  _SAMPLING_EPS: _SAMPLING_EPS.
  compute_logprobs: compute_logprobs().
  gather_logprobs: gather_logprobs().
---
# Module: [`tpu_inference/layers/jax/sample/sampling.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py)

## Classes
### `PromptLogprobsAsyncData`
- def: [`tpu_inference/layers/jax/sample/sampling.py:50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L50)
- doc: Holds async-copied prompt logprob tensors + per-request snapshots for get_output().
- signature: `class PromptLogprobsAsyncData:`
- members:
  - `req_snaps` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L53)
  - `tensors` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L52)
- uses (calls/refs, reference-scoped): [`PromptLogprobsReqSnap`](sampling.md#PromptLogprobsReqSnap)
- used by: [`compute_prompt_logprobs`](sampling.md#compute_prompt_logprobs), `_get_prompt_logprobs_dict`, `__init__`

### `PromptLogprobsReqSnap`
- def: [`tpu_inference/layers/jax/sample/sampling.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L38)
- doc: Per-request state snapshotted at step N for use in get_output().
- signature: `class PromptLogprobsReqSnap:`
- members:
  - `is_last_chunk` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L45)
  - `num_k` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L46)
  - `num_logits` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L44)
  - `req_id` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L40)
  - `req_offset` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L42)
  - `req_state` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L41)
  - `start_idx` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L43)
- uses (calls/refs, reference-scoped): [`CachedRequestState`](../../../runner/input_batch.md#CachedRequestState)
- used by: [`compute_prompt_logprobs`](sampling.md#compute_prompt_logprobs), `_get_prompt_logprobs_dict`, [`req_snaps`](sampling.md#PromptLogprobsAsyncData.req_snaps)

## Functions
- `_apply_sampling_transforms(logits: jax.Array, tpu_sampling_metadata: TPUSupportedSamplingMetadata)` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L68) — Apply temperature scaling, top-k, and top-p filtering to logits.
- `_jax_logprobs_copy_to_host_async(logprobs_tensors: LogprobsTensors)` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L56) — Initiate non-blocking TPU-to-host copies for all logprobs arrays.
- `compute_and_gather_logprobs(logits: jax.Array, next_tokens: jax.Array, max_logprobs: int)` — [`L155`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L155) — Compute logprobs from logits and gather the requested top-k.
- `compute_and_gather_prompt_logprobs(logits: jax.Array, input_ids: jax.Array, max_logprobs: int)` — [`L166`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L166) — Compute logprobs from full logits and gather the requested top-k for prompt tokens.
- `compute_logprobs(logits: jax.Array)` — [`L150`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L150)
- `compute_prompt_logprobs(full_logits: Optional[jax.Array], input_ids: Optional[jax.Array], num_prompt_logprobs: Dict[str, int], requests: Dict[str, CachedRequestState], scheduler_output: VllmSchedulerOutput, req_ids_dp: Optional[Dict[int, List[str]]], dp_size: int, max_logprobs: int)` — [`L176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L176) — Dispatches prompt logprob computation on TPU and snapshots per-request state.
- `gather_logprobs(logprobs: jax.Array, token_ids: jax.Array, num_logprobs: int)` — [`L238`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L238) — Gather logprobs for topk and sampled/prompt token.
- `sample(rng: jax.Array, mesh: Mesh, logits: jax.Array, tpu_sampling_metadata: TPUSupportedSamplingMetadata)` — [`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L106)

## Module values
- `_SAMPLING_EPS` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling.py#L34)

