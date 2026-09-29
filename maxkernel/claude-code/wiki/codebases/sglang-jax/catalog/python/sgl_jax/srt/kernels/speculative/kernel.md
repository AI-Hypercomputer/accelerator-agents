---
title: 'Module: python/sgl_jax/srt/kernels/speculative/kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/speculative/kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.speculative.kernel`/
symbols:
  top_p_renorm_prob: top_p_renorm_prob().
  top_k_renorm_prob: top_k_renorm_prob().
  tree_speculative_sampling_target_only: tree_speculative_sampling_target_only().
  create_extend_after_decode_spec_info: create_extend_after_decode_spec_info().
  create_extend_after_decode_spec_info.compute_position_updates: create_extend_after_decode_spec_info().compute_position_updates().
  top_p_renorm_prob.predicate: top_p_renorm_prob().predicate().
  _sampling_from_prob: _sampling_from_prob().
  verify_tree_greedy: verify_tree_greedy().
  align_evict_mask_to_page_size: align_evict_mask_to_page_size().
  get_target_cache_loc: get_target_cache_loc().
  filter_finished_cache_loc_kernel: filter_finished_cache_loc_kernel().
---
# Module: [`python/sgl_jax/srt/kernels/speculative/kernel.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py)

## Functions
- `_sampling_from_prob(probs: jax.Array, threshold: jax.Array)` — [`L156`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L156)
- `align_evict_mask_to_page_size(seq_lens, evict_mask, page_size, num_draft_tokens)` — [`L262`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L262)
- `compute_position_updates()` — [`L16`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L16)
- `create_extend_after_decode_spec_info(verified_id, seq_lens, accept_lens, positions, new_verified_id)` — [`L7`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L7)
- `filter_finished_cache_loc_kernel(tgt_cache_loc: jnp.array, accept_length: jnp.array, accept_length_filter: jnp.array)` — [`L322`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L322)
- `get_target_cache_loc(accept_length: jnp.array, to_free_num_slots: jnp.array, out_cache_loc: jnp.array, num_verify_tokens: int)` — [`L279`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L279)
- `predicate(threshold)` — [`L146`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L146)
- `top_k_renorm_prob(probs, top_k_values)` — [`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L103) — Renormalizing probabilities by top-k thresholding.
- `top_p_renorm_prob(probs, top_p_values)` — [`L124`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L124) — Renormalizing probabilities by top-p thresholding.
- `tree_speculative_sampling_target_only(predicts: jax.Array, accept_index: jax.Array, accept_token_num: jax.Array, candidates: jax.Array, retrive_index: jax.Array, retrive_next_token: jax.Array, retrive_next_sibling: jax.Array, uniform_samples: jax.Array, uniform_samples_for_final_sampling: jax.Array, target_probs: jax.Array, draft_probs: jax.Array, threshold_single: float = 1, threshold_acc: float = 1, deterministic: bool = True)` — [`L163`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L163) — Verify the draft tree with specific sample policy.
- `verify_tree_greedy(predicts: jax.Array, accept_index: jax.Array, accept_token_num: jax.Array, candidates: jax.Array, retrive_index: jax.Array, retrive_next_token: jax.Array, retrive_next_sibling: jax.Array, target_predict: jax.Array)` — [`L39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/kernel.py#L39) — Verify the draft tree with greedy sample policy.

