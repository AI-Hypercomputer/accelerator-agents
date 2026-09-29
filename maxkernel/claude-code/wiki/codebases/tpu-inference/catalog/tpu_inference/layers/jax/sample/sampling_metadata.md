---
title: 'Module: tpu_inference/layers/jax/sample/sampling_metadata.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/sample/sampling_metadata.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.sample.sampling_metadata`/
symbols:
  TPUSupportedSamplingMetadata: TPUSupportedSamplingMetadata#
  TPUSupportedSamplingMetadata.from_input_batch: TPUSupportedSamplingMetadata#from_input_batch().
  TPUSupportedSamplingMetadata.do_sampling: TPUSupportedSamplingMetadata#do_sampling.
  TPUSupportedSamplingMetadata.temperature: TPUSupportedSamplingMetadata#temperature.
  TPUSupportedSamplingMetadata.top_k: TPUSupportedSamplingMetadata#top_k.
  TPUSupportedSamplingMetadata.top_p: TPUSupportedSamplingMetadata#top_p.
  TPUSupportedSamplingMetadata._cache_collision_dummy: TPUSupportedSamplingMetadata#_cache_collision_dummy.
  TPUSupportedSamplingMetadata.logprobs: TPUSupportedSamplingMetadata#logprobs.
  DEFAULT_SAMPLING_PARAMS: DEFAULT_SAMPLING_PARAMS.
  TPUSupportedSamplingMetadata.fill_slice: TPUSupportedSamplingMetadata#fill_slice().
---
# Module: [`tpu_inference/layers/jax/sample/sampling_metadata.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py)

## Classes
### `TPUSupportedSamplingMetadata`
- def: [`tpu_inference/layers/jax/sample/sampling_metadata.py:45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L45)
- signature: `class TPUSupportedSamplingMetadata:`
- members:
  - `fill_slice(cpu_tensor_np: np.ndarray, fill_val: float)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L78)
  - `from_input_batch(cls, mesh: Mesh, input_batch: InputBatch, padded_num_reqs: int, req_indices_dp: dict, sharding: Optional[jax.sharding.Sharding] = None)` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L54)
  - `do_sampling` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L50)
  - `logprobs` — [`L51`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L51)
  - `temperature` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L46)
  - `top_k` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L47)
  - `top_p` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L48)
- protocol/private: `_cache_collision_dummy`[`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L49)
- uses (calls/refs, reference-scoped): [`InputBatch`](../../../runner/input_batch.md#InputBatch), [`temperature_cpu`](../../../runner/input_batch.md#InputBatch.temperature_cpu), [`top_k_cpu`](../../../runner/input_batch.md#InputBatch.top_k_cpu), [`top_p_cpu`](../../../runner/input_batch.md#InputBatch.top_p_cpu), [`all_greedy`](../../../runner/input_batch.md#InputBatch.all_greedy), [`max_num_logprobs`](../../../runner/input_batch.md#InputBatch.max_num_logprobs), [`DEFAULT_SAMPLING_PARAMS`](sampling_metadata.md#DEFAULT_SAMPLING_PARAMS)
- used by: `_sample_from_logits`, [`_precompile_continue_decode`](../../../runner/compilation_manager.md#CompilationManager._precompile_continue_decode), `_prepare_inputs`, [`_precompile_process_and_extend_logits`](../../../runner/compilation_manager.md#CompilationManager._precompile_process_and_extend_logits), [`_precompile_sampling`](../../../runner/compilation_manager.md#CompilationManager._precompile_sampling), [`_precompile_rejection_sampler`](../../../runner/compilation_manager.md#CompilationManager._precompile_rejection_sampler), [`sample`](sampling.md#sample), [`forward`](rejection_sampler.md#RejectionSampler.forward), [`_forward`](rejection_sampler.md#RejectionSampler._forward), [`process_and_extend_logits`](../../../spec_decode/jax/utils.md#process_and_extend_logits), [`_compute_probs`](rejection_sampler.md#_compute_probs), [`local_fn`](../../../spec_decode/jax/utils.md#process_and_extend_logits.local_fn), [`_apply_sampling_transforms`](sampling.md#_apply_sampling_transforms), `sampling_metadata`, [`__call__`](rejection_sampler.md#RejectionSampler.__call__)

## Module values
- `DEFAULT_SAMPLING_PARAMS` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/sample/sampling_metadata.py#L27)

