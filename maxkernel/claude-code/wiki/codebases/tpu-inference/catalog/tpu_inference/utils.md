---
title: 'Module: tpu_inference/utils.py'
type: catalog
provenance: extracted
module: tpu_inference/utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.utils`/
symbols:
  PhasedBasedProfiler._start_profiling: PhasedBasedProfiler#_start_profiling().
  determine_phase_from_batch_composition_stats: determine_phase_from_batch_composition_stats().
  SpecDecodeMetadata: SpecDecodeMetadata#
  logger: logger.
  AggregatedStatsLogger.flush: AggregatedStatsLogger#flush().
  PhasedBasedProfiler.inference_phase_seen: PhasedBasedProfiler#inference_phase_seen.
  host_extract_sampled_tokens: host_extract_sampled_tokens().
  PhasedBasedProfiler._merge_profile_directories: PhasedBasedProfiler#_merge_profile_directories().
  extract_request_ids_for_tracing: extract_request_ids_for_tracing().
  AggregatedStatsLogger.log: AggregatedStatsLogger#log().
  PhasedBasedProfiler.step: PhasedBasedProfiler#step().
  InferencePhase: InferencePhase#
  AggregatedStatsLogger.close: AggregatedStatsLogger#close().
  PhasedBasedProfiler._canonical_dst_ts: PhasedBasedProfiler#_canonical_dst_ts.
  PhasedBasedProfiler._step_or_stop_profiling: PhasedBasedProfiler#_step_or_stop_profiling().
  get_attn_req_paddings: get_attn_req_paddings().
  get_padded_token_len: get_padded_token_len().
  get_batch_composition_stats: get_batch_composition_stats().
  PhasedBasedProfiler.default_profiling_options: PhasedBasedProfiler#default_profiling_options.
  PhasedBasedProfiler._resolve_canonical_dst_ts: PhasedBasedProfiler#_resolve_canonical_dst_ts().
  get_req_paddings: get_req_paddings().
  AggregatedStatsLogger._f_tmp: AggregatedStatsLogger#_f_tmp.
  LatencyTracker.__exit__: LatencyTracker#__exit__().
  PhasedBasedProfiler.aggregated_stats_logger: PhasedBasedProfiler#aggregated_stats_logger.
  AggregatedStatsLogger.target_file: AggregatedStatsLogger#target_file.
  SpecDecodeMetadata.final_logits_indices: SpecDecodeMetadata#final_logits_indices.
  PhasedBasedProfiler.decode_kv_len_threshold: PhasedBasedProfiler#decode_kv_len_threshold.
  PhasedBasedProfiler.current_phase: PhasedBasedProfiler#current_phase.
  SpecDecodeMetadata.draft_lengths: SpecDecodeMetadata#draft_lengths.
  SpecDecodeMetadata.target_logits_indices: SpecDecodeMetadata#target_logits_indices.
  get_token_paddings: get_token_paddings().
  PhasedBasedProfiler.num_decode_steps_to_skip: PhasedBasedProfiler#num_decode_steps_to_skip.
  PhasedBasedProfiler.profile_dir_with_phase_suffix: PhasedBasedProfiler#profile_dir_with_phase_suffix.
  get_padded_num_reqs_with_upper_limit: get_padded_num_reqs_with_upper_limit().
  AggregatedStatsLogger._f_local: AggregatedStatsLogger#_f_local.
  ForbidCompile.__enter__: ForbidCompile#__enter__().
  AggregatedStatsLogger.local_temp_file: AggregatedStatsLogger#local_temp_file.
  SpecDecodeMetadata.bonus_logits_indices: SpecDecodeMetadata#bonus_logits_indices.
  SpecDecodeMetadata.draft_lengths_cpu: SpecDecodeMetadata#draft_lengths_cpu.
  SpecDecodeMetadata.req_indices_dp: SpecDecodeMetadata#req_indices_dp.
  AggregatedStatsLogger._upload: AggregatedStatsLogger#_upload().
  PhasedBasedProfiler._write_batch_composition_stats_to_file_helper: PhasedBasedProfiler#_write_batch_composition_stats_to_file_helper().
  ForbidCompile._original_func: ForbidCompile#_original_func.
  PhasedBasedProfiler.profiling_n_steps_left: PhasedBasedProfiler#profiling_n_steps_left.
  PhasedBasedProfiler.worker_rank: PhasedBasedProfiler#worker_rank.
  ForbidCompile.wrapper: ForbidCompile#wrapper().
  AggregatedStatsLogger._sync_to_gcs: AggregatedStatsLogger#_sync_to_gcs().
  PhasedBasedProfiler.num_steps_to_profile_for: PhasedBasedProfiler#num_steps_to_profile_for.
  MIN_NUM_SEQS: MIN_NUM_SEQS.
  InferencePhase.DECODE_HEAVY: InferencePhase#DECODE_HEAVY.
  InferencePhase.DECODE_ONLY: InferencePhase#DECODE_ONLY.
  LatencyTracker: LatencyTracker#
  AggregatedStatsLogger.step_count: AggregatedStatsLogger#step_count.
  PhasedBasedProfiler.decode_steps_skipped: PhasedBasedProfiler#decode_steps_skipped.
  SpecDecodeMetadata.req_ids_dp: SpecDecodeMetadata#req_ids_dp.
  ForbidCompile.__exit__: ForbidCompile#__exit__().
  BALANCED_RATIO_THRESHOLD: BALANCED_RATIO_THRESHOLD.
  InferencePhase.PREFILL_HEAVY: InferencePhase#PREFILL_HEAVY.
  InferencePhase.BALANCED: InferencePhase#BALANCED.
  InferencePhase.PREFILL_ONLY: InferencePhase#PREFILL_ONLY.
  ForbidCompile: ForbidCompile#
  AggregatedStatsLogger: AggregatedStatsLogger#
  AggregatedStatsLogger.profile_dir: AggregatedStatsLogger#profile_dir.
  PhasedBasedProfiler: PhasedBasedProfiler#
  PhasedBasedProfiler.profile_dir: PhasedBasedProfiler#profile_dir.
  PhasedBasedProfiler._CANONICAL_TS_POLL_TIMEOUT_S: PhasedBasedProfiler#_CANONICAL_TS_POLL_TIMEOUT_S.
  get_eos_token_id: get_eos_token_id().
  get_pad_token_id: get_pad_token_id().
  trim_request_id_suffix: trim_request_id_suffix().
  PREFILL_HEAVY_RATIO_THRESHOLD: PREFILL_HEAVY_RATIO_THRESHOLD.
  DECODE_HEAVY_RATIO_THRESHOLD: DECODE_HEAVY_RATIO_THRESHOLD.
  PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR: PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR.
  PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP: PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP.
  PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD: PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD.
  InferencePhase.AMBIGUOUS: InferencePhase#AMBIGUOUS.
  _inject_dp_rank_into_filename: _inject_dp_rank_into_filename().
  LatencyTracker.name: LatencyTracker#name.
  LatencyTracker.start_time: LatencyTracker#start_time.
  LatencyTracker.end_time: LatencyTracker#end_time.
  ForbidCompile.message: ForbidCompile#message.
  AggregatedStatsLogger.flush_interval: AggregatedStatsLogger#flush_interval.
  AggregatedStatsLogger._get_local_and_target_paths: AggregatedStatsLogger#_get_local_and_target_paths().
  PhasedBasedProfiler._CANONICAL_TS_POLL_INTERVAL_S: PhasedBasedProfiler#_CANONICAL_TS_POLL_INTERVAL_S.
  get_kv_transfer_metadata: get_kv_transfer_metadata().
  LatencyTracker.__init__: LatencyTracker#__init__().
  LatencyTracker.__enter__: LatencyTracker#__enter__().
  ForbidCompile.__init__: ForbidCompile#__init__().
  AggregatedStatsLogger.__init__: AggregatedStatsLogger#__init__().
  PhasedBasedProfiler.__init__: PhasedBasedProfiler#__init__().
---
# Module: [`tpu_inference/utils.py`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py)

## Classes
### `AggregatedStatsLogger`
- def: [`tpu_inference/utils.py:412`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L412)
- doc: Logs batch composition stats continuously for all steps to a file and
- signature: `class AggregatedStatsLogger:`
- members:
  - `_get_local_and_target_paths(self, base_dir: str, filename: str)` — [`L443`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L443) — Helper to resolve local temp path vs final target path (e.g. for GCS).
  - `_sync_to_gcs(self, local_file: str, target_file: str, blocking: bool = False)` — [`L452`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L452) — Helper to sync local file to GCS using the Python SDK.
  - `close(self)` — [`L513`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L513) — Closes the file handles, ensuring a final blocking flush to storage.
  - `flush(self, blocking: bool = False)` — [`L494`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L494) — Flushes the current buffered logs to local disk and syncs to Google Cloud Storage (GCS)
  - `log(self, batch_composition_stats: dict)` — [`L477`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L477) — Logs a single batch's composition statistics to local temporary files.
  - `flush_interval` — [`L424`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L424)
  - `local_temp_file` — [`L430`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L430)
  - `profile_dir` — [`L423`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L423)
  - `step_count` — [`L425`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L425)
  - `target_file` — [`L430`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L430)
- protocol/private: `__init__`[`L422`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L422), `_f_local`[`L433`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L433), `_f_tmp`[`L434`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L434), `_upload`[`L459`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L459)
- uses (calls/refs, reference-scoped): [`logger`](utils.md#logger)
- used by: `_prepare_inputs`, `aggregated_stats_logger`

### `ForbidCompile`
- def: [`tpu_inference/utils.py:237`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L237)
- doc: A context manager to forbid JAX compilation in a specific block of code.
- signature: `class ForbidCompile:`
- members:
  - `wrapper(*args, **kwargs)` — [`L272`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L272)
  - `message` — [`L262`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L262)
- protocol/private: `__enter__`[`L265`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L265), `__exit__`[`L292`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L292), `__init__`[`L258`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L258), `_original_func`[`L263`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L263)
- used by: `maybe_forbid_compile`

### `InferencePhase`  ·  implements/extends Enum
- def: [`tpu_inference/utils.py:121`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L121)
- signature: `class InferencePhase(Enum):`
- members:
  - `AMBIGUOUS` — [`L125`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L125)
  - `BALANCED` — [`L124`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L124)
  - `DECODE_HEAVY` — [`L123`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L123)
  - `DECODE_ONLY` — [`L127`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L127)
  - `PREFILL_HEAVY` — [`L122`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L122)
  - `PREFILL_ONLY` — [`L126`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L126)
- used by: [`_start_profiling`](utils.md#PhasedBasedProfiler._start_profiling), [`determine_phase_from_batch_composition_stats`](utils.md#determine_phase_from_batch_composition_stats), [`inference_phase_seen`](utils.md#PhasedBasedProfiler.inference_phase_seen)

### `LatencyTracker`
- def: [`tpu_inference/utils.py:222`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L222)
- signature: `class LatencyTracker:`
- members:
  - `end_time` — [`L232`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L232)
  - `name` — [`L225`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L225)
  - `start_time` — [`L228`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L228)
- protocol/private: `__enter__`[`L227`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L227), `__exit__`[`L231`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L231), `__init__`[`L224`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L224)
- uses (calls/refs, reference-scoped): [`logger`](utils.md#logger)
- used by: [`insert_request_with_kv_cache`](runner/kv_cache_manager.md#KVCacheManager.insert_request_with_kv_cache), [`_jitted_insert_kv_cache`](runner/kv_cache_manager.md#KVCacheManager._jitted_insert_kv_cache)

### `PhasedBasedProfiler`
- def: [`tpu_inference/utils.py:529`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L529)
- doc: Implements a phased-based profiler, which will profile three phases:
- signature: `class PhasedBasedProfiler:`
- members:
  - `_merge_profile_directories(self)` — [`L774`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L774) — Consolidates phase trace artifacts so downstream tools (c2xprof,
  - `_resolve_canonical_dst_ts(self, phase_dir: str)` — [`L732`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L732) — Resolve the canonical destination timestamp for this phase.
  - `_start_profiling(self, batch_composition_stats: dict)` — [`L630`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L630) — Potentially starts profiling for a given unseen phase.
  - `_step_or_stop_profiling(self, batch_composition_stats: dict)` — [`L699`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L699) — Steps the profiler or stops it if we have profiled enough steps for the
  - `_write_batch_composition_stats_to_file_helper(self, batch_composition_stats: dict)` — [`L614`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L614) — Writes the batch composition stats to a file at the given time,
  - `step(self, batch_composition_stats: dict)` — [`L835`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L835) — Steps the profiler and logs batch composition stats.
  - `aggregated_stats_logger` — [`L601`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L601)
  - `current_phase` — [`L598`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L598)
  - `decode_kv_len_threshold` — [`L572`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L572)
  - `decode_steps_skipped` — [`L571`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L571)
  - `default_profiling_options` — [`L584`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L584)
  - `inference_phase_seen` — [`L577`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L577)
  - `num_decode_steps_to_skip` — [`L568`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L568)
  - `num_steps_to_profile_for` — [`L565`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L565)
  - `profile_dir` — [`L575`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L575)
  - `profile_dir_with_phase_suffix` — [`L564`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L564)
  - `profiling_n_steps_left` — [`L563`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L563)
  - `worker_rank` — [`L600`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L600)
- protocol/private: `_CANONICAL_TS_POLL_INTERVAL_S`[`L730`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L730), `_CANONICAL_TS_POLL_TIMEOUT_S`[`L729`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L729), `__init__`[`L559`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L559), `_canonical_dst_ts`[`L683`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L683)
- uses (calls/refs, reference-scoped): [`determine_phase_from_batch_composition_stats`](utils.md#determine_phase_from_batch_composition_stats), [`logger`](utils.md#logger), [`InferencePhase`](utils.md#InferencePhase), [`TPU_MULTIPROCESS_DP`](envs.md#TPU_MULTIPROCESS_DP), [`DECODE_HEAVY`](utils.md#InferencePhase.DECODE_HEAVY), [`DECODE_ONLY`](utils.md#InferencePhase.DECODE_ONLY), [`BALANCED`](utils.md#InferencePhase.BALANCED), [`PREFILL_HEAVY`](utils.md#InferencePhase.PREFILL_HEAVY), [`PREFILL_ONLY`](utils.md#InferencePhase.PREFILL_ONLY), [`PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD`](utils.md#PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD), [`PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP`](utils.md#PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP), [`PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR`](utils.md#PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR), [`PROFILE_SINGLE_DEVICE`](envs.md#PROFILE_SINGLE_DEVICE), [`PYTHON_TRACER_LEVEL`](envs.md#PYTHON_TRACER_LEVEL), [`_inject_dp_rank_into_filename`](utils.md#_inject_dp_rank_into_filename)
- used by: `_prepare_inputs`, `phase_based_profiler`

### `SpecDecodeMetadata`
- def: [`tpu_inference/utils.py:877`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L877)
- doc: Metadata for speculative decoding on JAX/TPU, containing all necessary indices.
- signature: `class SpecDecodeMetadata:`
- members:
  - `bonus_logits_indices` — [`L881`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L881)
  - `draft_lengths` — [`L879`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L879)
  - `draft_lengths_cpu` — [`L884`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L884)
  - `final_logits_indices` — [`L882`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L882)
  - `req_ids_dp` — [`L886`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L886)
  - `req_indices_dp` — [`L885`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L885)
  - `target_logits_indices` — [`L880`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L880)
- used by: `_sample_from_logits`, `_prepare_inputs`, [`_precompile_process_and_extend_logits`](runner/compilation_manager.md#CompilationManager._precompile_process_and_extend_logits), [`_precompile_extract_last_sampled_tokens`](runner/compilation_manager.md#CompilationManager._precompile_extract_last_sampled_tokens), [`propose_eagle3_draft_token_ids`](runner/speculative_decoding_manager.md#SpeculativeDecodingManager.propose_eagle3_draft_token_ids), [`propose_draft_token_ids`](runner/speculative_decoding_manager.md#SpeculativeDecodingManager.propose_draft_token_ids), [`get_spec_decode_metadata`](runner/speculative_decoding_manager.md#SpeculativeDecodingManager.get_spec_decode_metadata), [`process_and_extend_logits`](spec_decode/jax/utils.md#process_and_extend_logits), `_update_placeholder`, [`host_extract_sampled_tokens`](utils.md#host_extract_sampled_tokens), [`extract_last_sampled_tokens`](spec_decode/jax/utils.md#extract_last_sampled_tokens), [`filter_speculative_logprobs`](spec_decode/jax/utils.md#filter_speculative_logprobs), `_jax_logprobs_materialize`, `spec_decode_metadata`, `spec_decode_metadata`, `__init__`

## Functions
- `_inject_dp_rank_into_filename(fname: str, dp_rank: int)` — [`L130`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L130) — Prefix `dp<N>_` to an xplane or trace filename, e.g.
- `determine_phase_from_batch_composition_stats(batch_composition_stats: dict[str, Any])` — [`L377`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L377) — Determines the inference phase based on the batch composition stats.
- `extract_request_ids_for_tracing(input_batch: InputBatch, scheduler_output: Optional[Any] = None)` — [`L65`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L65) — Extracts request IDs from an InputBatch and formats them for XProf tracing.
- `get_attn_req_paddings(min_req_size: int, max_req_size: int)` — [`L160`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L160) — Get num reqs paddings with custom override to reduce compilation time
- `get_batch_composition_stats(batch_id: int, input_batch: InputBatch, total_num_scheduled_tokens: int, num_reqs: int, padded_total_num_scheduled_tokens: int, scheduler_output: VllmSchedulerOutput)` — [`L300`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L300) — Logs the total number of tokens scheduled for the batch, the number of
- `get_eos_token_id(model_config: Any)` — [`L915`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L915) — Extract EOS token ID from the model configuration with fallback.
- `get_kv_transfer_metadata(kv: list[Any])` — [`L42`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L42) — Returns the dimensions string and size in bytes for a list of KV cache tensors.
- `get_pad_token_id(model_config: Any)` — [`L934`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L934) — Extract padding token ID from the model configuration with fallback.
- `get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int)` — [`L143`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L143)
- `get_padded_token_len(paddings: list[int], x: int)` — [`L214`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L214) — Return the first element in paddings list greater or equal to x.
- `get_req_paddings(min_req_size: int, max_req_size: int)` — [`L148`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L148)
- `get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int)` — [`L180`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L180) — Generate a list of padding size, starting from min_token_size,
- `host_extract_sampled_tokens(runner, spec_decode_metadata: Optional[SpecDecodeMetadata], sampled_output: jnp.ndarray, logits_indices_selector: np.ndarray, discard_sampled_tokens_req_indices: list, num_reqs: int)` — [`L889`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L889) — host retrieve the sampled tokens for the current step.
- `trim_request_id_suffix(request_id: str)` — [`L30`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L30) — Trims the suffix from a request ID, keeping only the base ID.

## Module values
- `BALANCED_RATIO_THRESHOLD` — [`L111`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L111)
- `DECODE_HEAVY_RATIO_THRESHOLD` — [`L108`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L108)
- `MIN_NUM_SEQS` — [`L99`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L99)
- `PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD` — [`L116`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L116)
- `PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP` — [`L113`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L113)
- `PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR` — [`L112`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L112)
- `PREFILL_HEAVY_RATIO_THRESHOLD` — [`L105`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L105)
- `logger` — [`L118`](../../../../../raw/code/tpu-inference/tpu_inference/utils.py#L118)

