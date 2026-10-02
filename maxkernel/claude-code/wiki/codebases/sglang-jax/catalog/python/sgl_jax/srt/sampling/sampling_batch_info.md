---
title: 'Module: python/sgl_jax/srt/sampling/sampling_batch_info.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/sampling/sampling_batch_info.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.sampling.sampling_batch_info`/
symbols:
  SamplingMetadata.from_model_worker_batch: SamplingMetadata#from_model_worker_batch().
  SamplingMetadata.tree_flatten: SamplingMetadata#tree_flatten().
  SamplingMetadata.tree_unflatten: SamplingMetadata#tree_unflatten().
  SamplingBatchInfo.from_schedule_batch: SamplingBatchInfo#from_schedule_batch().
  SamplingBatchInfo.penalizer_orchestrator: SamplingBatchInfo#penalizer_orchestrator.
  SamplingMetadata: SamplingMetadata#
  SamplingBatchInfo.merge_batch: SamplingBatchInfo#merge_batch().
  SamplingBatchInfo.linear_penalty: SamplingBatchInfo#linear_penalty.
  SamplingBatchInfo: SamplingBatchInfo#
  SamplingBatchInfo.update_penalties: SamplingBatchInfo#update_penalties().
  _get_or_create_zero_penalty_device: _get_or_create_zero_penalty_device().
  SamplingMetadata.vocab_mask: SamplingMetadata#vocab_mask.
  SamplingMetadata.apply_vocab_mask: SamplingMetadata#apply_vocab_mask.
  SamplingBatchInfo.generate_for_precompile: SamplingBatchInfo#generate_for_precompile().
  SamplingBatchInfo.generate_for_precompile_all_greedy: SamplingBatchInfo#generate_for_precompile_all_greedy().
  SamplingBatchInfo.cumulate_output_tokens: SamplingBatchInfo#cumulate_output_tokens().
  SamplingBatchInfo.filter_batch: SamplingBatchInfo#filter_batch().
  SamplingBatchInfo.is_all_greedy: SamplingBatchInfo#is_all_greedy.
  SamplingBatchInfo.sampling_seeds: SamplingBatchInfo#sampling_seeds.
  SamplingBatchInfo._get_global_server_args_dict: SamplingBatchInfo#_get_global_server_args_dict().
  SamplingMetadata.return_logprob: SamplingMetadata#return_logprob.
  SamplingMetadata.is_all_greedy: SamplingMetadata#is_all_greedy.
  SamplingMetadata.do_penalties: SamplingMetadata#do_penalties.
  SamplingBatchInfo.temperatures: SamplingBatchInfo#temperatures.
  _zero_linear_penalty_cache._zero_linear_penalty_cache: _zero_linear_penalty_cache._zero_linear_penalty_cache.
  SamplingBatchInfo.__len__: SamplingBatchInfo#__len__().
  _zero_linear_penalty_lock: _zero_linear_penalty_lock.
  SamplingMetadata.top_logprobs_nums: SamplingMetadata#top_logprobs_nums.
  SamplingMetadata.token_ids_logprobs: SamplingMetadata#token_ids_logprobs.
  SamplingMetadata.temperatures: SamplingMetadata#temperatures.
  SamplingMetadata.top_ps: SamplingMetadata#top_ps.
  SamplingMetadata.top_ks: SamplingMetadata#top_ks.
  SamplingMetadata.min_ps: SamplingMetadata#min_ps.
  SamplingMetadata.sampling_seeds: SamplingMetadata#sampling_seeds.
  SamplingMetadata.positions: SamplingMetadata#positions.
  SamplingMetadata.need_min_p_sampling: SamplingMetadata#need_min_p_sampling.
  SamplingMetadata.linear_penalty: SamplingMetadata#linear_penalty.
  SamplingBatchInfo.top_ps: SamplingBatchInfo#top_ps.
  SamplingBatchInfo.top_ks: SamplingBatchInfo#top_ks.
  SamplingBatchInfo.min_ps: SamplingBatchInfo#min_ps.
  SamplingBatchInfo.vocab_size: SamplingBatchInfo#vocab_size.
  SamplingBatchInfo.need_top_p_sampling: SamplingBatchInfo#need_top_p_sampling.
  SamplingBatchInfo.need_top_k_sampling: SamplingBatchInfo#need_top_k_sampling.
  SamplingBatchInfo.need_min_p_sampling: SamplingBatchInfo#need_min_p_sampling.
  logger: logger.
  SamplingBatchInfo.sampling_info_done: SamplingBatchInfo#sampling_info_done.
---
# Module: [`python/sgl_jax/srt/sampling/sampling_batch_info.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py)

## Classes
### `SamplingBatchInfo`
- def: [`python/sgl_jax/srt/sampling/sampling_batch_info.py:244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L244)
- doc: Batched sampling information for a generation batch.
- signature: `class SamplingBatchInfo:`
- members:
  - `cumulate_output_tokens(self, output_ids: jax.Array)` — [`L445`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L445) — Feed the output tokens to the penalty orchestrator.
  - `filter_batch(self, keep_indices: np.ndarray)` — [`L407`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L407)
  - `from_schedule_batch(cls, info: ScheduleReqsInfo, vocab_size: int, batch: ScheduleBatch = None)` — [`L345`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L345)
  - `generate_for_precompile(cls, bs: int, vocab_size: int = 32000)` — [`L283`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L283)
  - `generate_for_precompile_all_greedy(cls, bs: int, vocab_size: int = 32000, do_penalties: bool = False)` — [`L315`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L315)
  - `merge_batch(self, other: SamplingBatchInfo)` — [`L422`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L422)
  - `update_penalties(self)` — [`L400`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L400)
  - `is_all_greedy` — [`L256`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L256)
  - `linear_penalty` — [`L274`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L274)
  - `min_ps` — [`L251`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L251)
  - `need_min_p_sampling` — [`L265`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L265)
  - `need_top_k_sampling` — [`L262`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L262)
  - `need_top_p_sampling` — [`L259`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L259)
  - `penalizer_orchestrator` — [`L273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L273)
  - `sampling_info_done` — [`L268`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L268)
  - `sampling_seeds` — [`L270`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L270)
  - `temperatures` — [`L248`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L248)
  - `top_ks` — [`L250`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L250)
  - `top_ps` — [`L249`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L249)
  - `vocab_size` — [`L253`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L253)
- protocol/private: `__len__`[`L397`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L397), `_get_global_server_args_dict`[`L277`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L277)
- uses (calls/refs, reference-scoped): `reqs`, `sampling_params`, `ScheduleBatch`, [`get_bool_env_var`](../utils/common_utils.md#get_bool_env_var), `global_server_args_dict`, [`DEFAULT_SAMPLING_SEED`](sampling_params.md#DEFAULT_SAMPLING_SEED), [`sampling_seed`](sampling_params.md#SamplingParams.sampling_seed), `ScheduleReqsInfo`, [`top_k`](sampling_params.md#SamplingParams.top_k), [`TOP_K_ALL`](sampling_params.md#TOP_K_ALL), [`temperature`](sampling_params.md#SamplingParams.temperature), `is_required`, [`min_p`](sampling_params.md#SamplingParams.min_p), [`top_p`](sampling_params.md#SamplingParams.top_p)
- used by: `prepare_for_extend`, `prepare_for_decode`, `filter_batch`, `_merge_sampling_info`, `merge_batch`, `sampling_info`, `prepare_for_idle`

### `SamplingMetadata`
- def: [`python/sgl_jax/srt/sampling/sampling_batch_info.py:35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L35)
- doc: SamplingMetadata is used as input parameter for jitted sample function.
- signature: `class SamplingMetadata:`
- members:
  - `from_model_worker_batch(cls, batch: ModelWorkerBatch, pad_size: int = 0, mesh: Mesh = None, vocab_size: int = 32000)` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L110)
  - `tree_flatten(self)` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L63)
  - `tree_unflatten(cls, aux_data, children)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L87)
  - `apply_vocab_mask` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L61)
  - `do_penalties` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L56)
  - `is_all_greedy` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L52)
  - `linear_penalty` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L57)
  - `min_ps` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L49)
  - `need_min_p_sampling` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L53)
  - `positions` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L51)
  - `return_logprob` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L41)
  - `sampling_seeds` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L50)
  - `temperatures` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L46)
  - `token_ids_logprobs` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L43)
  - `top_ks` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L48)
  - `top_logprobs_nums` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L42)
  - `top_ps` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L47)
  - `vocab_mask` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L60)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, `seq_lens`, `forward_mode`, [`device_array`](../utils/jax_utils.md#device_array), `sampling_info`, [`is_decode`](../model_executor/forward_batch_info.md#ForwardMode.is_decode), `temperatures`, `positions`, `vocab_mask`, `token_ids_logprobs`, `top_logprobs_nums`, `return_logprob`, `top_ps`, [`_get_or_create_zero_penalty_device`](sampling_batch_info.md#_get_or_create_zero_penalty_device), `top_ks`, `min_ps`, `is_all_greedy`, `linear_penalty`, `sampling_seeds`, `penalizer_orchestrator`, `need_min_p_sampling`
- used by: [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`cur_sampling_info`](../managers/tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`_precompile_decode`](../model_executor/compilation_manager.md#CompilationManager._precompile_decode), [`_precompile_extend`](../model_executor/compilation_manager.md#CompilationManager._precompile_extend), [`__call__`](../layers/sampler.md#Sampler.__call__), [`_update_grammar_vocab_mask`](../managers/tp_worker.md#ModelWorker._update_grammar_vocab_mask), [`forward_batch_generation`](../managers/tp_worker_overlap_thread.md#ModelWorkerClient.forward_batch_generation), [`sample`](../model_executor/model_runner.md#ModelRunner.sample), [`forward_batch_generation`](../managers/tp_worker.md#MockModelWorker.forward_batch_generation)

## Functions
- `_get_or_create_zero_penalty_device(shape: tuple[int, int], sharding: NamedSharding | None)` — [`L214`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L214) — Reuse cached zero penalties per shape to avoid reallocation.

## Module values
- `_zero_linear_penalty_cache` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L29)
- `_zero_linear_penalty_lock` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L30)
- `logger` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py#L27)

