---
title: 'Module: python/sgl_jax/srt/layers/sampler.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/sampler.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.sampler`/
symbols:
  Sampler._process_logprob_results: Sampler#_process_logprob_results().
  Sampler.__call__: Sampler#__call__().
  top_k_top_p_min_p_sampling_from_probs_jax_with_sort: top_k_top_p_min_p_sampling_from_probs_jax_with_sort().
  top_k_top_p_min_p_sampling_from_probs_jax_with_mask: top_k_top_p_min_p_sampling_from_probs_jax_with_mask().
  Sampler.mesh: Sampler#mesh.
  Sampler._regular_sampling: Sampler#_regular_sampling().
  top_k_top_p_min_p_sampling_from_probs_jax: top_k_top_p_min_p_sampling_from_probs_jax().
  get_top_logprobs: get_top_logprobs().
  multinomial_with_seed: multinomial_with_seed().
  get_token_ids_logprobs: get_token_ids_logprobs().
  multinomial: multinomial().
  Sampler: Sampler#
  top_p_normalize_probs_jax: top_p_normalize_probs_jax().
  _apply_min_p_filter: _apply_min_p_filter().
  compute_logprobs: compute_logprobs().
  Sampler.rngs: Sampler#rngs.
  Sampler._greedy_sampling: Sampler#_greedy_sampling().
  Sampler._apply_linear_penalty: Sampler#_apply_linear_penalty().
  _get_sorted_indices_np: _get_sorted_indices_np().
  Sampler.__init__: Sampler#__init__().
  Sampler._apply_min_tokens_penalty: Sampler#_apply_min_tokens_penalty().
---
# Module: [`python/sgl_jax/srt/layers/sampler.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py)

## Classes
### `Sampler`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/sampler.py:18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L18)
- signature: `class Sampler(nnx.Module):`
- members:
  - `__call__(self, logits_output: LogitsProcessorOutput, sampling_metadata: SamplingMetadata, use_sort_for_toppk_minp: bool, rng_override: jax.Array | None = None)` — [`L166`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L166) — Run a sampler & compute logprobs and update logits_output accordingly.
  - `_apply_linear_penalty(self, operands)` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L123) — Args:
  - `_apply_min_tokens_penalty(self, operands)` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L140) — Apply min new tokens penalty to stop tokens
  - `_greedy_sampling(self, operands)` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L23) — Greedy sampling branch
  - `_process_logprob_results(self, operands)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L73) — Process logprob results when return_logprob=True
  - `_regular_sampling(self, operands)` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L30) — Regular sampling branch
  - `mesh` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L21)
  - `rngs` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L20)
- protocol/private: `__init__`[`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L19)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`LogitsProcessorOutput`](logits_processor.md#LogitsProcessorOutput), [`next_token_logits`](logits_processor.md#LogitsProcessorOutput.next_token_logits), [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`next_token_logprobs`](logits_processor.md#LogitsProcessorOutput.next_token_logprobs), [`next_token_top_logprobs_val`](logits_processor.md#LogitsProcessorOutput.next_token_top_logprobs_val), [`input_token_logprobs`](logits_processor.md#LogitsProcessorOutput.input_token_logprobs), [`next_token_token_ids_logprobs_val`](logits_processor.md#LogitsProcessorOutput.next_token_token_ids_logprobs_val), [`next_token_top_logprobs_idx`](logits_processor.md#LogitsProcessorOutput.next_token_top_logprobs_idx), [`input_token_ids_logprobs_val`](logits_processor.md#LogitsProcessorOutput.input_token_ids_logprobs_val), [`input_top_logprobs_val`](logits_processor.md#LogitsProcessorOutput.input_top_logprobs_val), [`input_top_logprobs_idx`](logits_processor.md#LogitsProcessorOutput.input_top_logprobs_idx), [`next_token_token_ids_logprobs_idx`](logits_processor.md#LogitsProcessorOutput.next_token_token_ids_logprobs_idx), [`input_token_ids_logprobs_idx`](logits_processor.md#LogitsProcessorOutput.input_token_ids_logprobs_idx), [`top_k_top_p_min_p_sampling_from_probs_jax`](sampler.md#top_k_top_p_min_p_sampling_from_probs_jax), [`apply_vocab_mask`](../sampling/sampling_batch_info.md#SamplingMetadata.apply_vocab_mask), [`get_top_logprobs`](sampler.md#get_top_logprobs), [`vocab_mask`](../sampling/sampling_batch_info.md#SamplingMetadata.vocab_mask), [`apply_token_bitmask`](../constrained/bitmask_ops.md#apply_token_bitmask), [`do_penalties`](../sampling/sampling_batch_info.md#SamplingMetadata.do_penalties), [`get_token_ids_logprobs`](sampler.md#get_token_ids_logprobs), [`is_all_greedy`](../sampling/sampling_batch_info.md#SamplingMetadata.is_all_greedy), [`return_logprob`](../sampling/sampling_batch_info.md#SamplingMetadata.return_logprob)
- used by: [`sampler`](../model_executor/model_runner.md#ModelRunner.sampler)

## Functions
- `_apply_min_p_filter(operands)` — [`L333`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L333) — Apply min_p filtering when need_min_p_sampling=True
- `_get_sorted_indices_np(probs_np: np.ndarray)` — [`L295`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L295) — CPU-side NumPy sorting index that is robust to NaNs/Infs.
- `compute_logprobs(mesh: jax.sharding.Mesh, logprobs: jax.Array, token_ids: jax.Array)` — [`L452`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L452)
- `get_token_ids_logprobs(logprobs: jax.Array, token_ids_logprobs: list[list[int]], mesh: Mesh)` — [`L232`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L232)
- `get_top_logprobs(logprobs: jax.Array, top_logprobs_nums: list[int], mesh: Mesh = None)` — [`L220`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L220)
- `multinomial(operands)` — [`L241`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L241)
- `multinomial_with_seed(operands)` — [`L250`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L250) — Note:
- `top_k_top_p_min_p_sampling_from_probs_jax(args, use_sort_for_toppk_minp)` — [`L342`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L342)
- `top_k_top_p_min_p_sampling_from_probs_jax_with_mask(args)` — [`L413`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L413)
- `top_k_top_p_min_p_sampling_from_probs_jax_with_sort(args)` — [`L351`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L351)
- `top_p_normalize_probs_jax(probs: jax.Array, top_ps: jax.Array, mesh: Mesh = None)` — [`L306`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/sampler.py#L306)

