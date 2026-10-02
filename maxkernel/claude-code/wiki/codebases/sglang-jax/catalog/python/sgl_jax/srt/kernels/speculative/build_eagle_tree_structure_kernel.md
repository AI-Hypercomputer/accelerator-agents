---
title: 'Module: python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.speculative.build_eagle_tree_structure_kernel`/
symbols:
  _build_eagle_tree_structure_kernel.build_tree_body: _build_eagle_tree_structure_kernel().build_tree_body().
  _build_eagle_tree_structure_kernel.set_tree_mask: _build_eagle_tree_structure_kernel().set_tree_mask().
  _build_eagle_tree_structure_kernel: _build_eagle_tree_structure_kernel().
  _build_eagle_tree_structure_kernel.init_output: _build_eagle_tree_structure_kernel().init_output().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_tb_idx_over_0: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_tb_idx_over_0().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_position_lt_draft_token_num: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_position_lt_draft_token_num().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_not_equal_to_0: _build_eagle_tree_structure_kernel().build_tree_body().tid_not_equal_to_0().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_not_equal_to_0.body: _build_eagle_tree_structure_kernel().build_tree_body().tid_not_equal_to_0().body().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_not_equal_to_0.body.parent_tb_idx_not_equal_to_0: _build_eagle_tree_structure_kernel().build_tree_body().tid_not_equal_to_0().body().parent_tb_idx_not_equal_to_0().
  build_eagle_tree_structure: build_eagle_tree_structure().
  _build_eagle_tree_structure_kernel.init_output.init_tree_mask: _build_eagle_tree_structure_kernel().init_output().init_tree_mask().
  _build_eagle_tree_structure_kernel.init_output.init_positions_and_retrive: _build_eagle_tree_structure_kernel().init_output().init_positions_and_retrive().
  _build_eagle_tree_structure_kernel.on_true: _build_eagle_tree_structure_kernel().on_true().
  _build_eagle_tree_structure_kernel.build_tree_body.loop_body: _build_eagle_tree_structure_kernel().build_tree_body().loop_body().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_tb_idx_over_0.body: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_tb_idx_over_0().body().
  build_eagle_tree_structure_pallas_call: build_eagle_tree_structure_pallas_call().
  _build_eagle_tree_structure_kernel.set_tree_mask.set_true: _build_eagle_tree_structure_kernel().set_tree_mask().set_true().
  _build_eagle_tree_structure_kernel.set_tree_mask.set_false: _build_eagle_tree_structure_kernel().set_tree_mask().set_false().
  _build_eagle_tree_structure_kernel.init_output.init_positions_and_retrive.body: _build_eagle_tree_structure_kernel().init_output().init_positions_and_retrive().body().
  _build_eagle_tree_structure_kernel.on_true.body: _build_eagle_tree_structure_kernel().on_true().body().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_tb_idx_over_0.cond: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_tb_idx_over_0().cond().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_tb_idx_over_0.body.on_true: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_tb_idx_over_0().body().on_true().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_position_lt_draft_token_num.on_true: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_position_lt_draft_token_num().on_true().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_body.parent_position_lt_draft_token_num.on_false: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_body().parent_position_lt_draft_token_num().on_false().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_equal_to_0.while_cond: _build_eagle_tree_structure_kernel().build_tree_body().tid_equal_to_0().while_cond().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_not_equal_to_0.body.parent_tb_idx_not_equal_to_0.while_body: _build_eagle_tree_structure_kernel().build_tree_body().tid_not_equal_to_0().body().parent_tb_idx_not_equal_to_0().while_body().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_not_equal_to_0.body.parent_tb_idx_not_equal_to_0.while_cond: _build_eagle_tree_structure_kernel().build_tree_body().tid_not_equal_to_0().body().parent_tb_idx_not_equal_to_0().while_cond().
  _build_eagle_tree_structure_kernel.build_tree_body.tid_not_equal_to_0.cond: _build_eagle_tree_structure_kernel().build_tree_body().tid_not_equal_to_0().cond().
---
# Module: [`python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py)

## Functions
- `_build_eagle_tree_structure_kernel(parents_ref, selected_index_ref, verified_seq_len_ref, cu_full_mask_len_ref, tree_mask_size_ref, zeros_ref, ones_ref, o_tree_mask_ref, o_positions_ref, o_retrive_index_ref, o_retrive_next_token_ref, o_retrive_next_sibling_ref, *, draft_token_num: int, topk: int, tree_mask_mode: int = 0)` — [`L11`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L11)
- `body(i, _)` — [`L73`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L73)
- `body(i, seq_tree_idx)` — [`L101`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L101)
- `body(state)` — [`L163`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L163)
- `body(state)` — [`L235`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L235)
- `build_eagle_tree_structure(parent_list: jax.Array, selected_index: jax.Array, verified_seq_len: jax.Array, seq_lens_sum: jax.Array, draft_token_num: int, topk: int, max_context_len: int, tree_mask_mode: int = 0)` — [`L424`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L424) — Build eagle tree using a Pallas kernel.
- `build_eagle_tree_structure_pallas_call(parent_list: jax.Array, selected_index: jax.Array, verified_seq_len: jax.Array, seq_lens_sum: jax.Array, *, draft_token_num: int, topk: int, max_context_len: int, tree_mask_mode: int = 0)` — [`L310`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L310)
- `build_tree_body(tid, _)` — [`L118`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L118)
- `cond(state)` — [`L159`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L159)
- `cond(state)` — [`L278`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L278)
- `init_output()` — [`L58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L58)
- `init_positions_and_retrive()` — [`L72`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L72)
- `init_tree_mask()` — [`L59`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L59)
- `loop_body(i, _)` — [`L130`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L130)
- `on_false()` — [`L194`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L194)
- `on_true(seq_tree_idx)` — [`L100`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L100)
- `on_true()` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L166)
- `on_true()` — [`L191`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L191)
- `parent_position_lt_draft_token_num(parent_position)` — [`L190`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L190)
- `parent_tb_idx_not_equal_to_0(cur_position)` — [`L246`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L246)
- `parent_tb_idx_over_0()` — [`L155`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L155)
- `set_false()` — [`L46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L46)
- `set_tree_mask(start, offset, val)` — [`L39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L39)
- `set_true()` — [`L40`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L40)
- `tid_equal_to_0()` — [`L140`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L140)
- `tid_not_equal_to_0()` — [`L229`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L229)
- `while_body(state)` — [`L148`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L148)
- `while_body(state)` — [`L249`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L249)
- `while_cond(state)` — [`L214`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L214)
- `while_cond(state)` — [`L259`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L259)

