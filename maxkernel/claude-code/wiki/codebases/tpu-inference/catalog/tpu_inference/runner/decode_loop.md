---
title: 'Module: tpu_inference/runner/decode_loop.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/decode_loop.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.decode_loop`/
symbols:
  continue_decode: continue_decode().
  _decode_core._run_one_step: _decode_core()._run_one_step().
  continue_decode._model_experts_only: continue_decode()._model_experts_only().
  _decode_core: _decode_core().
  TpuSamplingState: TpuSamplingState#
  TpuSamplingState.attn_metadata: TpuSamplingState#attn_metadata.
  _decode_core.body_fn: _decode_core().body_fn().
  TpuSamplingState.current_tokens: TpuSamplingState#current_tokens.
  TpuSamplingState.step_counter: TpuSamplingState#step_counter.
  TpuSamplingState.active_mask: TpuSamplingState#active_mask.
  _decode_core._pack: _decode_core()._pack().
  _decode_core._unpack: _decode_core()._unpack().
  _update_loop_state: _update_loop_state().
  _split_rngs: _split_rngs().
  _decode_core.cond_fn: _decode_core().cond_fn().
---
# Module: [`tpu_inference/runner/decode_loop.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py)

## Classes
### `TpuSamplingState`
- def: [`tpu_inference/runner/decode_loop.py:36`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L36)
- signature: `class TpuSamplingState:`
- members:
  - `active_mask` — [`L40`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L40)
  - `attn_metadata` — [`L42`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L42)
  - `current_tokens` — [`L38`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L38)
  - `step_counter` — [`L44`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L44)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../layers/common/attention_metadata.md#AttentionMetadata)
- used by: [`_precompile_continue_decode`](compilation_manager.md#CompilationManager._precompile_continue_decode), `_execute_continue_decode`, [`continue_decode`](decode_loop.md#continue_decode)

## Functions
- `_decode_core(*, state, kv_caches, step_rngs, sampling_metadata, inputs_embeds, lora_metadata, intermediate_tensors, block_tables, query_start_loc, request_distribution, mamba_state_indices, current_tokens, active_mask, input_positions, seq_lens, model_fn, compute_logits_fn, sample_fn, mesh, max_decode_steps, static_max_decode_steps, eos_token_id, padding_token_id, dp_size, pad_len, has_experts, expert_shape, expert_dtype, layer_name_to_kvcache_index, is_first_rank, is_last_rank)` — [`L122`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L122)
- `_model_experts_only(current_tokens, input_positions, seq_lens, kv_caches)` — [`L334`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L334)
- `_pack(i, ct, am, pos, sl, kvc, tb, eb, eos)` — [`L206`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L206)
- `_run_one_step(step_idx, ct, am, pos, sl, kvc)` — [`L157`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L157)
- `_split_rngs(rng, static_size, dynamic_size)` — [`L85`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L85)
- `_unpack(carry)` — [`L210`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L210)
- `_update_loop_state(next_tokens: jax.Array, active_mask: jax.Array, input_positions: jax.Array, seq_lens: jax.Array, eos_token_id: tuple[int, ...], padding_token_id: int, dp_size: int, pad_len: int)` — [`L50`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L50)
- `body_fn(carry)` — [`L222`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L222)
- `cond_fn(carry)` — [`L216`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L216)
- `continue_decode(state: dict, model_fn: Callable, compute_logits_fn: Callable, sample_fn: Callable, init_state: TpuSamplingState, kv_caches: Any, max_decode_steps: int, static_max_decode_steps: int, eos_token_id: tuple[int, ...], padding_token_id: int, rng: jax.Array, *, mesh: Any, sampling_metadata: Any, inputs_embeds: jax.Array | None = None, layer_name_to_kvcache_index: tuple[tuple[str, int], ...] = (), lora_metadata: dict | None = None, intermediate_tensors: dict[str, jax.Array] | None = None, is_first_rank: bool = True, is_last_rank: bool = True, dp_size: int = 1, collect_expert_indices: bool = False)` — [`L251`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/decode_loop.py#L251) — Run the TPU decode loop as one fused, kv-cache-donating program. — documented in [root](../../../concepts/root.md)

