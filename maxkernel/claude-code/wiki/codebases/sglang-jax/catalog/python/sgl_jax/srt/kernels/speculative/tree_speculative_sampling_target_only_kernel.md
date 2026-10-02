---
title: 'Module: python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.speculative.tree_speculative_sampling_target_only_kernel`/
symbols:
  _tree_speculative_sampling_target_only_kernel: _tree_speculative_sampling_target_only_kernel().
  _tree_speculative_sampling_target_only_kernel.body.verify: _tree_speculative_sampling_target_only_kernel().body().verify().
  _tree_speculative_sampling_target_only_kernel.body.verify.while_body: _tree_speculative_sampling_target_only_kernel().body().verify().while_body().
  tree_speculative_sampling_target_only_pallas_call: tree_speculative_sampling_target_only_pallas_call().
  align_to: align_to().
  _tree_speculative_sampling_target_only_kernel.init_accept_index: _tree_speculative_sampling_target_only_kernel().init_accept_index().
  _tree_speculative_sampling_target_only_kernel.probs_cumsum: _tree_speculative_sampling_target_only_kernel().probs_cumsum().
  _tree_speculative_sampling_target_only_kernel._sampling_from_prob: _tree_speculative_sampling_target_only_kernel()._sampling_from_prob().
  _tree_speculative_sampling_target_only_kernel.body: _tree_speculative_sampling_target_only_kernel().body().
  prepare_for_verify: prepare_for_verify().
  _tree_speculative_sampling_target_only_kernel.init_accept_index.body: _tree_speculative_sampling_target_only_kernel().init_accept_index().body().
  _tree_speculative_sampling_target_only_kernel.probs_cumsum.body: _tree_speculative_sampling_target_only_kernel().probs_cumsum().body().
  _tree_speculative_sampling_target_only_kernel.body.verify.while_body.on_true: _tree_speculative_sampling_target_only_kernel().body().verify().while_body().on_true().
  _tree_speculative_sampling_target_only_kernel.body.verify.while_body.on_false: _tree_speculative_sampling_target_only_kernel().body().verify().while_body().on_false().
  _tree_speculative_sampling_target_only_kernel.body.verify.cond_fn: _tree_speculative_sampling_target_only_kernel().body().verify().cond_fn().
  _tree_speculative_sampling_target_only_kernel.on_true: _tree_speculative_sampling_target_only_kernel().on_true().
  _tree_speculative_sampling_target_only_kernel.on_false: _tree_speculative_sampling_target_only_kernel().on_false().
---
# Module: [`python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py)

## Functions
- `_sampling_from_prob(probs, threshold)` — [`L87`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L87)
- `_tree_speculative_sampling_target_only_kernel(candidates_ref, retrive_index_ref, retrive_next_token_ref, retrive_next_sibling_ref, uniform_samples_ref, uniform_samples_for_final_sampling_ref, target_probs_ref, draft_probs_ref, zeros_ref, o_accept_index_ref, o_accept_token_num_ref, o_predicts_ref, target_probs_buffer_ref, draft_probs_buffer_ref, q_vec_ref, p_vec_ref, *, num_draft_tokens: int, num_spec_tokens: int, threshold_single: float = 1, threshold_acc: float = 1, deterministic: bool = True)` — [`L12`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L12)
- `align_to(x, a)` — [`L345`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L345)
- `body(i, _)` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L54)
- `body(i, state)` — [`L73`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L73)
- `body(i, state)` — [`L93`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L93)
- `cond_fn(state)` — [`L234`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L234)
- `init_accept_index()` — [`L53`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L53)
- `on_false(state)` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L166)
- `on_false(cur_prob_offset)` — [`L327`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L327)
- `on_true(state)` — [`L135`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L135)
- `on_true(cur_prob_offset)` — [`L324`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L324)
- `prepare_for_verify(candidates, retrive_index, retrive_next_token, retrive_next_sibling, uniform_samples, target_probs, draft_probs)` — [`L349`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L349)
- `probs_cumsum(arr)` — [`L69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L69)
- `tree_speculative_sampling_target_only_pallas_call(predicts: jax.Array, accept_index: jax.Array, accept_token_num: jax.Array, candidates: jax.Array, retrive_index: jax.Array, retrive_next_token: jax.Array, retrive_next_sibling: jax.Array, uniform_samples: jax.Array, uniform_samples_for_final_sampling: jax.Array, target_probs: jax.Array, draft_probs: jax.Array, threshold_single: float = 1, threshold_acc: float = 1, deterministic: bool = True)` — [`L417`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L417) — Verify the tree greedy using a Pallas kernel
- `verify(state)` — [`L104`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L104)
- `while_body(state)` — [`L116`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L116)

