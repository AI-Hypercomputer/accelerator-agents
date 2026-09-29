---
title: 'Module: easydel/inference/esurge/core/sampling_metadata.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/sampling_metadata.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.sampling_metadata`/SamplingMetadata#
symbols:
  SamplingMetadata: ''
  SamplingMetadata.temperatures: temperatures.
  SamplingMetadata.top_ps: top_ps.
  SamplingMetadata.top_ks: top_ks.
  SamplingMetadata.sampling_seeds: sampling_seeds.
  SamplingMetadata.linear_penalty: linear_penalty.
  SamplingMetadata.min_ps: min_ps.
  SamplingMetadata.is_all_greedy: is_all_greedy.
  SamplingMetadata.need_min_p_sampling: need_min_p_sampling.
  SamplingMetadata.do_penalties: do_penalties.
---
# Module: [`easydel/inference/esurge/core/sampling_metadata.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py)

## Classes
### `SamplingMetadata`
- def: [`easydel/inference/esurge/core/sampling_metadata.py:32`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L32)
- doc: Consolidated sampling parameters for JIT-compiled token sampling.
- signature: `class SamplingMetadata:`
- members:
  - `do_penalties` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L88)
  - `is_all_greedy` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L86)
  - `linear_penalty` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L91)
  - `min_ps` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L82)
  - `need_min_p_sampling` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L87)
  - `sampling_seeds` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L83)
  - `temperatures` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L79)
  - `top_ks` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L81)
  - `top_ps` — [`L80`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/sampling_metadata.py#L80)
- used by: [`_regular_sample`](sampler.md#_regular_sample), [`sample_tokens`](sampler.md#sample_tokens), [`_sampling_fn`](../runners/executors/sampler_executor.md#SamplerExecutor._sampling_fn), [`apply_min_p_mask`](binary_search.md#apply_min_p_mask), [`apply_penalties`](binary_search.md#apply_penalties), [`apply_topk`](sampler.md#_regular_sample.apply_topk), [`apply_topp`](sampler.md#_regular_sample.apply_topp), [`apply_temp`](sampler.md#_regular_sample.apply_temp)

