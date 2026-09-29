---
title: 'Module: src/maxtext/kernels/ragged/ragged_sort.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/ragged/ragged_sort.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.ragged.ragged_sort`/
symbols:
  ring_ragged_sort: ring_ragged_sort().
  ring_ragged_unsort: ring_ragged_unsort().
  a2a_ragged_sort: a2a_ragged_sort().
  a2a_ragged_unsort: a2a_ragged_unsort().
  ring_ragged_sort._ring_ragged_sort: ring_ragged_sort()._ring_ragged_sort().
  ring_ragged_sort._ring_ragged_sort_fwd: ring_ragged_sort()._ring_ragged_sort_fwd().
  ring_ragged_unsort._ring_ragged_unsort: ring_ragged_unsort()._ring_ragged_unsort().
  ring_ragged_unsort._ring_ragged_unsort_fwd: ring_ragged_unsort()._ring_ragged_unsort_fwd().
  a2a_ragged_sort._a2a_ragged_sort: a2a_ragged_sort()._a2a_ragged_sort().
  a2a_ragged_sort._a2a_ragged_sort_fwd: a2a_ragged_sort()._a2a_ragged_sort_fwd().
  a2a_ragged_unsort._a2a_ragged_unsort: a2a_ragged_unsort()._a2a_ragged_unsort().
  a2a_ragged_unsort._a2a_ragged_unsort_fwd: a2a_ragged_unsort()._a2a_ragged_unsort_fwd().
  ring_ragged_sort._ring_ragged_sort_bwd: ring_ragged_sort()._ring_ragged_sort_bwd().
  ring_ragged_unsort._ring_ragged_unsort_bwd: ring_ragged_unsort()._ring_ragged_unsort_bwd().
  a2a_ragged_sort._a2a_ragged_sort_bwd: a2a_ragged_sort()._a2a_ragged_sort_bwd().
  a2a_ragged_unsort._a2a_ragged_unsort_bwd: a2a_ragged_unsort()._a2a_ragged_unsort_bwd().
---
# Module: [`src/maxtext/kernels/ragged/ragged_sort.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py)

## Functions
- `_a2a_ragged_sort(inputs, sort_indices, valid_end)` — [`L451`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L451)
- `_a2a_ragged_sort_bwd(res, g_out)` — [`L466`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L466)
- `_a2a_ragged_sort_fwd(inputs, sort_indices, valid_end)` — [`L455`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L455)
- `_a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end)` — [`L519`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L519)
- `_a2a_ragged_unsort_bwd(res, g_out)` — [`L540`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L540)
- `_a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end)` — [`L523`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L523)
- `_ring_ragged_sort(hidden_states_local, topk_indices_local)` — [`L76`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L76) — Sort and gather activations to different EP shards.
- `_ring_ragged_sort_bwd(res, g_out)` — [`L152`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L152) — Backward pass for the gather: a Pallas SC ragged gather reduce.
- `_ring_ragged_sort_fwd(hidden_states_local, topk_indices_local)` — [`L81`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L81) — Sort and gather activations forward pass.
- `_ring_ragged_unsort(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk_weights_flat)` — [`L269`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L269) — Unsort and scatter activations.
- `_ring_ragged_unsort_bwd(res, g_out)` — [`L343`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L343) — Backward pass for the scatter with routing weights.
- `_ring_ragged_unsort_fwd(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk_weights_flat)` — [`L276`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L276) — Executes unsorting sending tokens back.
- `a2a_ragged_sort(inputs, sort_indices, valid_end, enforce_gather_fallback=False, enforce_gather_reduce_fallback=False)` — [`L418`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L418) — Ragged-gather variant for ``local_permute``. — documented in [maxtext-layers-moe](../../../../../concepts/maxtext-layers-moe.md)
- `a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end, enforce_gather_fallback=False, enforce_gather_reduce_fallback=False)` — [`L491`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L491) — Dual of :func:`a2a_ragged_sort`. — documented in [maxtext-layers-moe](../../../../../concepts/maxtext-layers-moe.md)
- `ring_ragged_sort(hidden_states_local, topk_indices_local, num_experts, topk, ep_name, ep_size, buffer_size=None, enforce_gather_fallback=False, enforce_gather_reduce_fallback=False, gather_flops_override=-1, gather_reduce_flops_override=-1, gather_bytes_accessed_override=-1, gather_reduce_bytes_accessed_override=-1)` — [`L23`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L23) — Ragged-gather variant for AG-RS Expert Parallelism token routing. — documented in [maxtext-layers-moe](../../../../../concepts/maxtext-layers-moe.md)
- `ring_ragged_unsort(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk, local_num_experts, ep_name, topk_weights, enforce_gather_fallback=False, enforce_gather_reduce_fallback=False, gather_flops_override=-1, gather_reduce_flops_override=-1, gather_bytes_accessed_override=-1, gather_reduce_bytes_accessed_override=-1)` — [`L224`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py#L224) — Dual of :func:`ring_ragged_sort`.

