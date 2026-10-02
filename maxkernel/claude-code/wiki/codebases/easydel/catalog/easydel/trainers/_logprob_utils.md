---
title: 'Module: easydel/trainers/_logprob_utils.py'
type: catalog
provenance: extracted
module: easydel/trainers/_logprob_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers._logprob_utils`/
symbols:
  compute_token_logps_and_entropies_chunked: compute_token_logps_and_entropies_chunked().
  compute_per_token_logps_and_entropies_from_hidden_states: compute_per_token_logps_and_entropies_from_hidden_states().
  compute_sequence_scores_from_hidden_states: compute_sequence_scores_from_hidden_states().
  resolve_lmhead_chunksize: resolve_lmhead_chunksize().
  compute_sequence_scores_from_hidden_states._chunk_contributions: compute_sequence_scores_from_hidden_states()._chunk_contributions().
  compute_per_token_logps_and_entropies_from_hidden_states._chunk_contributions: compute_per_token_logps_and_entropies_from_hidden_states()._chunk_contributions().
  compute_sequence_scores_from_hidden_states._accumulate_chunk: compute_sequence_scores_from_hidden_states()._accumulate_chunk().
  compute_per_token_logps_and_entropies_from_hidden_states._accumulate_chunk: compute_per_token_logps_and_entropies_from_hidden_states()._accumulate_chunk().
  compute_token_logps_and_entropies_chunked._max_step: compute_token_logps_and_entropies_chunked()._max_step().
  compute_token_logps_and_entropies_chunked._sum_step: compute_token_logps_and_entropies_chunked()._sum_step().
  compute_token_logps_and_entropies_chunked._entropy_step: compute_token_logps_and_entropies_chunked()._entropy_step().
  compute_token_logps_and_entropies_chunked.max_body: compute_token_logps_and_entropies_chunked().max_body().
  compute_token_logps_and_entropies_chunked.sum_body: compute_token_logps_and_entropies_chunked().sum_body().
  compute_token_logps_and_entropies_chunked.entropy_body: compute_token_logps_and_entropies_chunked().entropy_body().
  compute_sequence_scores_from_hidden_states._full_body: compute_sequence_scores_from_hidden_states()._full_body().
  compute_per_token_logps_and_entropies_from_hidden_states._full_body: compute_per_token_logps_and_entropies_from_hidden_states()._full_body().
  resolve_lmhead_projection_module: resolve_lmhead_projection_module().
  compute_sequence_scores_from_hidden_states._project_chunk: compute_sequence_scores_from_hidden_states()._project_chunk().
  compute_per_token_logps_and_entropies_from_hidden_states._project_chunk: compute_per_token_logps_and_entropies_from_hidden_states()._project_chunk().
---
# Module: [`easydel/trainers/_logprob_utils.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py)

## Functions
- `_accumulate_chunk(start: int, size: int, current: tuple[Array, Array, Array] | tuple[Array, Array, Array, Array])` — [`L363`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L363)
- `_accumulate_chunk(start: int, size: int, current: tuple[Array, Array])` — [`L505`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L505)
- `_chunk_contributions(chunk_hidden_states: Array, chunk_labels: Array, chunk_loss_mask: Array)` — [`L317`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L317)
- `_chunk_contributions(chunk_hidden_states: Array, chunk_targets: Array)` — [`L489`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L489)
- `_entropy_step(start: int, size: int, expected_logits: Array)` — [`L200`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L200)
- `_full_body(i: int, current: tuple[Array, Array, Array] | tuple[Array, Array, Array, Array])` — [`L391`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L391)
- `_full_body(i: int, current: tuple[Array, Array])` — [`L519`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L519)
- `_max_step(start: int, size: int, running_max: Array)` — [`L163`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L163)
- `_project_chunk(chunk_hidden_states: Array)` — [`L310`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L310)
- `_project_chunk(chunk_hidden_states: Array)` — [`L482`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L482)
- `_sum_step(start: int, size: int, running_sum: Array)` — [`L178`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L178)
- `compute_per_token_logps_and_entropies_from_hidden_states(model: tp.Any, hidden_states: Array, targets: Array, *, token_chunk_size: int, vocab_chunk_size: int | None, return_entropy: bool)` — [`L404`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L404) — Project hidden states through the LM head chunk-by-chunk and return per-token log-probabilities and optional entropies.
- `compute_sequence_scores_from_hidden_states(model: tp.Any, hidden_states: Array, labels: Array, loss_mask: Array, *, token_chunk_size: int, vocab_chunk_size: int | None, return_correct_counts: bool = False)` — [`L220`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L220) — Project hidden states through the LM head chunk-by-chunk and accumulate masked sequence-level scores.
- `compute_token_logps_and_entropies_chunked(logits: Array, targets: Array, *, return_entropy: bool, chunk_size: int | None)` — [`L90`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L90) — Compute per-token log-probabilities (and optional entropies) without materializing a full vocab-sized log-softmax.
- `entropy_body(i: int, expected_logits: Array)` — [`L207`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L207)
- `max_body(i: int, running_max: Array)` — [`L169`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L169)
- `resolve_lmhead_chunksize(model: tp.Any)` — [`L55`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L55) — Return the configured LM-head token chunk size when headless scoring is supported.
- `resolve_lmhead_projection_module(model: tp.Any)` — [`L24`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L24) — Resolve the module that owns the LM-head projection helpers.
- `sum_body(i: int, running_sum: Array)` — [`L184`](../../../../../../raw/code/EasyDeL/easydel/trainers/_logprob_utils.py#L184)

