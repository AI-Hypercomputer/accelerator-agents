---
title: 'Module: python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.speculative.verify_tree_greedy_kernel`/
symbols:
  _verify_tree_greedy_kernel: _verify_tree_greedy_kernel().
  _verify_tree_greedy_kernel.loop_body.verify: _verify_tree_greedy_kernel().loop_body().verify().
  _verify_tree_greedy_kernel.loop_body.verify.while_loop_body: _verify_tree_greedy_kernel().loop_body().verify().while_loop_body().
  verify_tree_greedy_pallas_call: verify_tree_greedy_pallas_call().
  verify_tree_greedy: verify_tree_greedy().
  _verify_tree_greedy_kernel.init_accept_index: _verify_tree_greedy_kernel().init_accept_index().
  _verify_tree_greedy_kernel.loop_body: _verify_tree_greedy_kernel().loop_body().
  align_to: align_to().
  prepare_for_verify: prepare_for_verify().
  _verify_tree_greedy_kernel.init_accept_index.body: _verify_tree_greedy_kernel().init_accept_index().body().
  _verify_tree_greedy_kernel.loop_body.verify.while_loop_body.on_true: _verify_tree_greedy_kernel().loop_body().verify().while_loop_body().on_true().
  _verify_tree_greedy_kernel.loop_body.verify.while_loop_body.on_false: _verify_tree_greedy_kernel().loop_body().verify().while_loop_body().on_false().
  _verify_tree_greedy_kernel.loop_body.verify.cond_fn: _verify_tree_greedy_kernel().loop_body().verify().cond_fn().
---
# Module: [`python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py)

## Functions
- `_verify_tree_greedy_kernel(candidates_ref, retrive_index_ref, retrive_next_token_ref, retrive_next_sibling_ref, target_predict_ref, o_accept_index_ref, o_accept_token_num_ref, o_predicts_ref, *, draft_token_num: int, num_spec_tokens: int)` — [`L12`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L12)
- `align_to(x, a)` — [`L164`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L164)
- `body(i, _)` — [`L30`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L30)
- `cond_fn(state)` — [`L128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L128)
- `init_accept_index()` — [`L29`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L29)
- `loop_body(i, state)` — [`L49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L49)
- `on_false(state)` — [`L86`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L86)
- `on_true(state)` — [`L62`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L62)
- `prepare_for_verify(candidates, retrive_index, retrive_next_token, retrive_next_sibling)` — [`L168`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L168)
- `verify(state)` — [`L52`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L52)
- `verify_tree_greedy(speculative_num_steps: int, num_draft_tokens: int, draft_tokens: jax.Array, retrive_index: jax.Array, retrive_next_token: jax.Array, retrive_next_sibling: jax.Array, next_token_logits: jax.Array)` — [`L268`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L268)
- `verify_tree_greedy_pallas_call(predicts: jax.Array, accept_index: jax.Array, accept_token_num: jax.Array, candidates: jax.Array, retrive_index: jax.Array, retrive_next_token: jax.Array, retrive_next_sibling: jax.Array, target_predict: jax.Array, draft_token_num: int, num_spec_tokens: int)` — [`L201`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L201) — Verify the tree greedy using a Pallas kernel
- `while_loop_body(state)` — [`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L56)

