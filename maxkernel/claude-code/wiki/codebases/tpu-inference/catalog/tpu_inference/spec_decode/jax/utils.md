---
title: 'Module: tpu_inference/spec_decode/jax/utils.py'
type: catalog
provenance: extracted
module: tpu_inference/spec_decode/jax/utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.spec_decode.jax.utils`/
symbols:
  process_and_extend_logits: process_and_extend_logits().
  process_and_extend_logits.local_fn: process_and_extend_logits().local_fn().
  extract_last_sampled_tokens: extract_last_sampled_tokens().
  filter_speculative_logprobs: filter_speculative_logprobs().
  extend_logits_simple: extend_logits_simple().
  PLACEHOLDER_TOKEN_ID: PLACEHOLDER_TOKEN_ID.
  concat_last_sampled_tokens_and_draft_tokens: concat_last_sampled_tokens_and_draft_tokens().
  extract_last_sampled_tokens._body: extract_last_sampled_tokens()._body().
  _extract_last_sampled_tokens: _extract_last_sampled_tokens().
  extend_logits_simple.concat_fn: extend_logits_simple().concat_fn().
---
# Module: [`tpu_inference/spec_decode/jax/utils.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py)

## Functions
- `_body(draft_lengths, sampled_token_ids)` — [`L38`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L38)
- `_extract_last_sampled_tokens(draft_lengths: jnp.ndarray, sampled_token_ids: jnp.ndarray, num_speculative_tokens: int, vocab_size: int, max_num_seq: int)` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L54) — Extract the last sampled token and number rejected tokens per seq.
- `concat_fn(x, y)` — [`L257`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L257)
- `concat_last_sampled_tokens_and_draft_tokens(last_sampled_tokens, draft_tokens)` — [`L100`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L100)
- `extend_logits_simple(target_logits: jax.Array, bonus_logits: jax.Array, mesh: jax.sharding.Mesh)` — [`L250`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L250) — Concatenates target and bonus logits along the first axis.
- `extract_last_sampled_tokens(spec_decode_metadata: SpecDecodeMetadata, sampled_token_ids: jnp.ndarray, num_speculative_tokens: int, vocab_size: int, max_num_reqs_per_dp_rank: int, mesh: jax.sharding.Mesh)` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L32)
- `filter_speculative_logprobs(log_token_ids: np.ndarray, logprobs_arr: np.ndarray, selected_token_ranks: np.ndarray, spec_decode_metadata: SpecDecodeMetadata, vocab_size: int, dp_size: int, num_reqs: int)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L106) — Filters and reorganizes logprobs for speculative decoding.
- `local_fn(local_target, local_bonus, local_draft_lengths, local_temp, local_top_k, local_top_p)` — [`L283`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L283)
- `process_and_extend_logits(mesh: jax.sharding.Mesh, target_logits: jax.Array, processed_bonus_logits: jax.Array, spec_decode_metadata: SpecDecodeMetadata, tpu_sampling_metadata: TPUSupportedSamplingMetadata)` — [`L270`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L270) — Processes target logits and concatenates them with processed bonus logits.

## Module values
- `PLACEHOLDER_TOKEN_ID` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/utils.py#L26)

