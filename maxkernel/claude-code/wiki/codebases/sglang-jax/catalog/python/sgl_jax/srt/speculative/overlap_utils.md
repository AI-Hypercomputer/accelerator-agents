---
title: 'Module: python/sgl_jax/srt/speculative/overlap_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/overlap_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.overlap_utils`/
symbols:
  use_legacy_eagle3_non_overlap: use_legacy_eagle3_non_overlap().
  publish_spec_decode_new_seq_lens: publish_spec_decode_new_seq_lens().
  resolve_spec_prefill_token_ids: resolve_spec_prefill_token_ids().
  can_use_spec_decode_overlap: can_use_spec_decode_overlap().
  can_use_spec_prefill_overlap: can_use_spec_prefill_overlap().
  resolve_spec_decode_token_ids: resolve_spec_decode_token_ids().
---
# Module: [`python/sgl_jax/srt/speculative/overlap_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py)

## Functions
- `can_use_spec_decode_overlap(enable_overlap, spec_algorithm, batch)` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py#L24)
- `can_use_spec_prefill_overlap(enable_overlap, spec_algorithm, batch)` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py#L46)
- `publish_spec_decode_new_seq_lens(batch_output)` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py#L17)
- `resolve_spec_decode_token_ids(result, batch, draft_token_num: int)` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py#L71) — Resolve per-request accepted token ids from a speculative verify result.
- `resolve_spec_prefill_token_ids(result)` — [`L4`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py#L4) — Resolve prefill next-token ids prepared by the producer path.
- `use_legacy_eagle3_non_overlap(enable_overlap, spec_algorithm)` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/overlap_utils.py#L62)

