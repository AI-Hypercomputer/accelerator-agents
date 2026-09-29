---
title: 'Module: python/sgl_jax/srt/model_executor/compilation_manager.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/model_executor/compilation_manager.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.model_executor.compilation_manager`/
symbols:
  CompilationManager._make_dummy_batch: CompilationManager#_make_dummy_batch().
  CompilationManager._precompile_extend: CompilationManager#_precompile_extend().
  CompilationManager._precompile_decode: CompilationManager#_precompile_decode().
  CompilationManager._compute_bs_buckets: CompilationManager#_compute_bs_buckets().
  CompilationManager._compute_token_buckets: CompilationManager#_compute_token_buckets().
  CompilationManager.bs_buckets: CompilationManager#bs_buckets.
  CompilationManager._compute_cache_loc_buckets: CompilationManager#_compute_cache_loc_buckets().
  CompilationManager.precompile_all: CompilationManager#precompile_all().
  CompilationManager.token_buckets: CompilationManager#token_buckets.
  CompilationManager.cache_loc_buckets: CompilationManager#cache_loc_buckets.
  CompilationManager.dp_size: CompilationManager#dp_size.
  logger: logger.
  CompilationManager.max_padded_batch_size: CompilationManager#max_padded_batch_size.
  CompilationManager.vocab_size: CompilationManager#vocab_size.
  CompilationManager._compiled_variants: CompilationManager#_compiled_variants.
  CompilationManager.moe_backend: CompilationManager#moe_backend.
  CompilationManager.page_size: CompilationManager#page_size.
  CompilationManager.max_padded_num_tokens: CompilationManager#max_padded_num_tokens.
  CompilationManager.__init__: CompilationManager#__init__().
  CompilationManager.enable_static_lora: CompilationManager#enable_static_lora.
  CompilationManager.register_variant_if_new: CompilationManager#register_variant_if_new().
  CompilationManager: CompilationManager#
  CompilationManager.has_recurrent_state: CompilationManager#has_recurrent_state.
  CompilationManager.tp_size: CompilationManager#tp_size.
  CompilationManager.max_req_len: CompilationManager#max_req_len.
  CompilationManager.multimodal: CompilationManager#multimodal.
---
# Module: [`python/sgl_jax/srt/model_executor/compilation_manager.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py)

## Classes
### `CompilationManager`
- def: [`python/sgl_jax/srt/model_executor/compilation_manager.py:24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L24)
- doc: Owns bucket computation, dummy batch construction, and pre-compilation.
- signature: `class CompilationManager:`
- members:
  - `precompile_all(self, forward_fn: Callable, model_runner: ModelRunner, mesh, prepare_lora_fn: Callable | None = None, future_token_ids_map=None)` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L106)
  - `register_variant_if_new(self, variant_key: tuple)` — [`L343`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L343) — Register a compilation variant and return True if it was not seen before.
  - `bs_buckets` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L58)
  - `cache_loc_buckets` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L59)
  - `dp_size` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L41)
  - `enable_static_lora` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L55)
  - `has_recurrent_state` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L49)
  - `max_padded_batch_size` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L45)
  - `max_padded_num_tokens` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L46)
  - `max_req_len` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L44)
  - `moe_backend` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L54)
  - `multimodal` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L48)
  - `page_size` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L43)
  - `token_buckets` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L57)
  - `tp_size` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L42)
  - `vocab_size` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L47)
- protocol/private: `__init__`[`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L27), `_compiled_variants`[`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L61), `_compute_bs_buckets`[`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L85), `_compute_cache_loc_buckets`[`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L100), `_compute_token_buckets`[`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L63), `_make_dummy_batch`[`L246`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L246), `_precompile_decode`[`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L180), `_precompile_extend`[`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L121)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`ForwardMode`](forward_batch_info.md#ForwardMode), [`init_new`](forward_batch_info.md#ForwardBatch.init_new), `ModelWorkerBatch`, `seq_lens`, [`from_model_worker_batch`](../sampling/sampling_batch_info.md#SamplingMetadata.from_model_worker_batch), [`CaptureHiddenMode`](forward_batch_info.md#CaptureHiddenMode), [`ForwardBatch`](forward_batch_info.md#ForwardBatch), `forward_mode`, `dp_size`, `sampling_info`, [`SpeculativeAlgorithm`](../speculative/spec_info.md#SpeculativeAlgorithm), `per_dp_bs_size`, [`ModelRunner`](model_runner.md#ModelRunner), [`EXTEND`](forward_batch_info.md#ForwardMode.EXTEND), `extend_seq_lens`, [`DECODE`](forward_batch_info.md#ForwardMode.DECODE), [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`FULL`](forward_batch_info.md#CaptureHiddenMode.FULL), `cache_loc`, `capture_hidden_mode`, `input_ids`, `logits_indices_selector`, `real_bs`, [`input_ids`](forward_batch_info.md#ForwardBatch.input_ids), `positions`, `vocab_mask`, `spec_algorithm`, `bid`, `logits_indices`, `lora_ids`, `token_ids_logprobs`, `top_logprobs_nums`, `ModelWorkerSamplingInfo`, `forward_batch`, `return_logprob`, [`enable_static_lora`](../server_args.md#ServerArgs.enable_static_lora), `extend_logprob_start_lens`, `real_bs_per_dp`, `req_pool_indices`  (+20 more)
- used by: [`compilation_manager`](../managers/tp_worker.md#ModelWorker.compilation_manager), [`run_precompile`](../managers/tp_worker.md#ModelWorker.run_precompile), [`get_precompile_paddings`](../managers/tp_worker.md#ModelWorker.get_precompile_paddings), [`max_padded_num_tokens`](../managers/tp_worker.md#ModelWorker.max_padded_num_tokens)

## Module values
- `logger` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/compilation_manager.py#L21)

