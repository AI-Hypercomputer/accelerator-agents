---
title: 'Module: python/sgl_jax/srt/layers/routed_experts_capturer.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/routed_experts_capturer.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.routed_experts_capturer`/
symbols:
  _ExpertBalanceAnalyzer._flush_segment: _ExpertBalanceAnalyzer#_flush_segment().
  _RoutedExpertsCapturerReal.on_forward_end: _RoutedExpertsCapturerReal#on_forward_end().
  _RoutedExpertsCapturerReal._sync_fwd_experts_buffer_DtoH: _RoutedExpertsCapturerReal#_sync_fwd_experts_buffer_DtoH().
  _RoutedExpertsCapturerReal.bid: _RoutedExpertsCapturerReal#bid.
  _RoutedExpertsCapturerReal._balance_analyzer: _RoutedExpertsCapturerReal#_balance_analyzer.
  _ExpertBalanceAnalyzer.add_decode_step: _ExpertBalanceAnalyzer#add_decode_step().
  _ExpertBalanceAnalyzer._flush_and_reset_segment: _ExpertBalanceAnalyzer#_flush_and_reset_segment().
  _RoutedExpertsCapturerReal._dist_recorder: _RoutedExpertsCapturerReal#_dist_recorder.
  _ExpertDistributionRecorder.add_topk_ids: _ExpertDistributionRecorder#add_topk_ids().
  _ExpertBalanceAnalyzer._open_writer: _ExpertBalanceAnalyzer#_open_writer().
  _ExpertBalanceAnalyzer._experts_per_device: _ExpertBalanceAnalyzer#_experts_per_device.
  _ExpertDistributionRecorder.dump: _ExpertDistributionRecorder#dump().
  _RoutedExpertsCapturerReal.host_buffer: _RoutedExpertsCapturerReal#host_buffer.
  _RoutedExpertsCapturerReal.get_routed_experts: _RoutedExpertsCapturerReal#get_routed_experts().
  logger: logger.
  _RoutedExpertsCapturerReal.dummy_experts_ids: _RoutedExpertsCapturerReal#dummy_experts_ids.
  RoutedExpertsCapturer.create: RoutedExpertsCapturer#create().
  _ExpertBalanceAnalyzer._ensure_writer: _ExpertBalanceAnalyzer#_ensure_writer().
  _ExpertDistributionRecorder._physical_counts: _ExpertDistributionRecorder#_physical_counts.
  _RoutedExpertsCapturerReal.reset: _RoutedExpertsCapturerReal#reset().
  _ExpertBalanceAnalyzer._file: _ExpertBalanceAnalyzer#_file.
  _ExpertBalanceAnalyzer.reset: _ExpertBalanceAnalyzer#reset().
  _ExpertDistributionRecorder.reset: _ExpertDistributionRecorder#reset().
  get_routed_expert_count: get_routed_expert_count().
  _global_expert_capturer._global_expert_capturer: _global_expert_capturer._global_expert_capturer.
  RoutedExpertsCapturer._sync_fwd_experts_buffer_DtoH: RoutedExpertsCapturer#_sync_fwd_experts_buffer_DtoH().
  RoutedExpertsCapturer.get_routed_experts: RoutedExpertsCapturer#get_routed_experts().
  RoutedExpertsCapturer.on_forward_end: RoutedExpertsCapturer#on_forward_end().
  _ExpertBalanceAnalyzer.output_file: _ExpertBalanceAnalyzer#output_file.
  set_global_experts_capturer: set_global_experts_capturer().
  _ExpertBalanceAnalyzer.num_experts: _ExpertBalanceAnalyzer#num_experts.
  _RoutedExpertsCapturerReal.num_experts_per_tok: _RoutedExpertsCapturerReal#num_experts_per_tok.
  get_global_experts_capturer: get_global_experts_capturer().
  _ExpertBalanceAnalyzer._writer: _ExpertBalanceAnalyzer#_writer.
  get_routed_experts_per_token: get_routed_experts_per_token().
  RoutedExpertsCapturer: RoutedExpertsCapturer#
  _RoutedExpertsCapturerReal.enable_host_buffer: _RoutedExpertsCapturerReal#enable_host_buffer.
  _RoutedExpertsCapturerReal.get_buffer_size_bytes: _RoutedExpertsCapturerReal#get_buffer_size_bytes().
  _ExpertBalanceAnalyzer.ep_size: _ExpertBalanceAnalyzer#ep_size.
  _RoutedExpertsCapturerReal.num_hidden_layers: _RoutedExpertsCapturerReal#num_hidden_layers.
  RoutedExpertsCapturer.reset: RoutedExpertsCapturer#reset().
  _ExpertBalanceAnalyzer._segment_decode_steps: _ExpertBalanceAnalyzer#_segment_decode_steps.
  _RoutedExpertsCapturerNoop: _RoutedExpertsCapturerNoop#
  _ExpertBalanceAnalyzer._segment_progress: _ExpertBalanceAnalyzer#_segment_progress.
  _ExpertDistributionRecorder._steps_accumulated: _ExpertDistributionRecorder#_steps_accumulated.
  _RoutedExpertsCapturerReal.max_padding: _RoutedExpertsCapturerReal#max_padding.
  _RoutedExpertsCapturerNoop.get_routed_experts: _RoutedExpertsCapturerNoop#get_routed_experts().
  _RoutedExpertsCapturerNoop.on_forward_end: _RoutedExpertsCapturerNoop#on_forward_end().
  _RoutedExpertsCapturerReal: _RoutedExpertsCapturerReal#
  _ExpertBalanceAnalyzer.segment_counter: _ExpertBalanceAnalyzer#segment_counter.
  _ExpertBalanceAnalyzer._counts: _ExpertBalanceAnalyzer#_counts.
  _ExpertBalanceAnalyzer._segment_padding_tokens_sum: _ExpertBalanceAnalyzer#_segment_padding_tokens_sum.
  _ExpertDistributionRecorder.num_layers: _ExpertDistributionRecorder#num_layers.
  _RoutedExpertsCapturerReal.__init__: _RoutedExpertsCapturerReal#__init__().
  _RoutedExpertsCapturerNoop._sync_fwd_experts_buffer_DtoH: _RoutedExpertsCapturerNoop#_sync_fwd_experts_buffer_DtoH().
  _first_present_attr: _first_present_attr().
  _RoutedExpertsCapturerReal.num_tokens: _RoutedExpertsCapturerReal#num_tokens.
  _RoutedExpertsCapturerReal._balance_missing_topk_warned: _RoutedExpertsCapturerReal#_balance_missing_topk_warned.
  _ExpertBalanceAnalyzer._segment_idx: _ExpertBalanceAnalyzer#_segment_idx.
  _ExpertBalanceAnalyzer._lock: _ExpertBalanceAnalyzer#_lock.
  _ExpertDistributionRecorder.physical_expert_counts: _ExpertDistributionRecorder#physical_expert_counts.
  _ExpertDistributionRecorder._lock: _ExpertDistributionRecorder#_lock.
  _GB: _GB.
  get_array_size_bytes: get_array_size_bytes().
  _TOTAL_EXPERT_FIELDS: _TOTAL_EXPERT_FIELDS.
  _TOPK_EXPERT_FIELDS: _TOPK_EXPERT_FIELDS.
  _ExpertBalanceAnalyzer.num_layers: _ExpertBalanceAnalyzer#num_layers.
  _ExpertBalanceAnalyzer.topk: _ExpertBalanceAnalyzer#topk.
  _ExpertBalanceAnalyzer._device_metrics_enabled: _ExpertBalanceAnalyzer#_device_metrics_enabled.
  _ExpertBalanceAnalyzer._header: _ExpertBalanceAnalyzer#_header.
  _ExpertDistributionRecorder.buffer_size: _ExpertDistributionRecorder#buffer_size.
  _ExpertDistributionRecorder.output_file: _ExpertDistributionRecorder#output_file.
  _ExpertBalanceAnalyzer: _ExpertBalanceAnalyzer#
  _ExpertDistributionRecorder: _ExpertDistributionRecorder#
  _MB: _MB.
  _RoutedExpertsCapturerReal.mesh: _RoutedExpertsCapturerReal#mesh.
  _RoutedExpertsCapturerNoop.__init__: _RoutedExpertsCapturerNoop#__init__().
  _RoutedExpertsCapturerNoop.reset: _RoutedExpertsCapturerNoop#reset().
  extract_routed_experts_from_meta_info: extract_routed_experts_from_meta_info().
  _ExpertBalanceAnalyzer.__init__: _ExpertBalanceAnalyzer#__init__().
  _ExpertDistributionRecorder.__init__: _ExpertDistributionRecorder#__init__().
---
# Module: [`python/sgl_jax/srt/layers/routed_experts_capturer.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py)

## Classes
### `RoutedExpertsCapturer`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/layers/routed_experts_capturer.py:72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L72)
- signature: `class RoutedExpertsCapturer(ABC):`
- members:
  - `create(mesh: jax.sharding.Mesh, enable: bool, model_config: ModelConfig, num_tokens: int, max_padding: int, ep_size: int, *, enable_balance_debug: bool = False, balance_segment_counter: int = 100, balance_output_file: str | None = None, enable_dist_recorder: bool = False, dist_recorder_buffer_size: int = 100, dist_recorder_output_file: str | None = None, physical_expert_counts: int = 256)` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L74)
  - `get_routed_experts(self, req_pool_idx: int, seqlen: int, req_to_token_pool: ReqToTokenPool, bid: int)` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L128)
  - `on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch)` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L138)
  - `reset(self)` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L142)
- protocol/private: `_sync_fwd_experts_buffer_DtoH`[`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L120)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, [`ModelConfig`](../configs/model_config.md#ModelConfig), [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`on_forward_end`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.on_forward_end), [`_sync_fwd_experts_buffer_DtoH`](routed_experts_capturer.md#_RoutedExpertsCapturerReal._sync_fwd_experts_buffer_DtoH), [`get_routed_experts`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.get_routed_experts), [`reset`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.reset), [`get_routed_expert_count`](routed_experts_capturer.md#get_routed_expert_count), [`_RoutedExpertsCapturerNoop`](routed_experts_capturer.md#_RoutedExpertsCapturerNoop), [`_RoutedExpertsCapturerReal`](routed_experts_capturer.md#_RoutedExpertsCapturerReal), [`get_routed_experts`](routed_experts_capturer.md#_RoutedExpertsCapturerNoop.get_routed_experts), [`on_forward_end`](routed_experts_capturer.md#_RoutedExpertsCapturerNoop.on_forward_end), [`_sync_fwd_experts_buffer_DtoH`](routed_experts_capturer.md#_RoutedExpertsCapturerNoop._sync_fwd_experts_buffer_DtoH), [`reset`](routed_experts_capturer.md#_RoutedExpertsCapturerNoop.reset)
- used by: [`init_routed_experts_capturer`](../model_executor/model_runner.md#ModelRunner.init_routed_experts_capturer), [`_global_expert_capturer`](routed_experts_capturer.md#_global_expert_capturer._global_expert_capturer), [`set_global_experts_capturer`](routed_experts_capturer.md#set_global_experts_capturer), [`_RoutedExpertsCapturerNoop`](routed_experts_capturer.md#_RoutedExpertsCapturerNoop), [`_RoutedExpertsCapturerReal`](routed_experts_capturer.md#_RoutedExpertsCapturerReal)

### `_ExpertBalanceAnalyzer`
- def: [`python/sgl_jax/srt/layers/routed_experts_capturer.py:374`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L374)
- members:
  - `add_decode_step(self, topk_ids_cpu: list[np.ndarray], real_bs_per_dp: list[int], per_dp_bs_size: int)` — [`L455`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L455)
  - `reset(self)` — [`L544`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L544)
  - `ep_size` — [`L388`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L388)
  - `num_experts` — [`L386`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L386)
  - `num_layers` — [`L385`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L385)
  - `output_file` — [`L390`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L390)
  - `segment_counter` — [`L389`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L389)
  - `topk` — [`L387`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L387)
- protocol/private: `__init__`[`L375`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L375), `_counts`[`L392`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L392), `_device_metrics_enabled`[`L399`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L399), `_ensure_writer`[`L444`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L444), `_experts_per_device`[`L400`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L400), `_file`[`L427`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L427), `_flush_and_reset_segment`[`L536`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L536), `_flush_segment`[`L485`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L485), `_header`[`L414`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L414), `_lock`[`L395`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L395), `_open_writer`[`L431`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L431), `_segment_decode_steps`[`L396`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L396), `_segment_idx`[`L394`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L394), `_segment_padding_tokens_sum`[`L397`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L397), `_segment_progress`[`L393`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L393), `_writer`[`L428`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L428)
- uses (calls/refs, reference-scoped): [`logger`](routed_experts_capturer.md#logger)
- used by: [`on_forward_end`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.on_forward_end), [`bid`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.bid), [`_balance_analyzer`](routed_experts_capturer.md#_RoutedExpertsCapturerReal._balance_analyzer), [`reset`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.reset)

### `_ExpertDistributionRecorder`
- def: [`python/sgl_jax/srt/layers/routed_experts_capturer.py:550`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L550)
- members:
  - `add_topk_ids(self, topk_ids_cpu: list[np.ndarray], real_bs_per_dp: list[int], per_dp_bs_size: int)` — [`L570`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L570)
  - `dump(self)` — [`L608`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L608)
  - `reset(self)` — [`L603`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L603)
  - `buffer_size` — [`L560`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L560)
  - `num_layers` — [`L559`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L559)
  - `output_file` — [`L561`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L561)
  - `physical_expert_counts` — [`L562`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L562)
- protocol/private: `__init__`[`L551`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L551), `_lock`[`L568`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L568), `_physical_counts`[`L564`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L564), `_steps_accumulated`[`L567`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L567)
- uses (calls/refs, reference-scoped): [`get_global_expert_location_metadata`](../eplb/expert_location.md#get_global_expert_location_metadata), [`logger`](routed_experts_capturer.md#logger)
- used by: [`on_forward_end`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.on_forward_end), [`_dist_recorder`](routed_experts_capturer.md#_RoutedExpertsCapturerReal._dist_recorder), [`reset`](routed_experts_capturer.md#_RoutedExpertsCapturerReal.reset)

### `_RoutedExpertsCapturerNoop`
- def: [`python/sgl_jax/srt/layers/routed_experts_capturer.py:322`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L322)
- members:
  - `get_routed_experts(self, req_pool_idx: int, seqlen: int, req_to_token_pool: ReqToTokenPool, bid: int)` — [`L333`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L333)
  - `on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch)` — [`L342`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L342)
  - `reset(self)` — [`L345`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L345)
- protocol/private: `__init__`[`L323`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L323), `_sync_fwd_experts_buffer_DtoH`[`L326`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L326)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`RoutedExpertsCapturer`](routed_experts_capturer.md#RoutedExpertsCapturer)
- used by: [`maybe_collect_routed_experts`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.maybe_collect_routed_experts), [`create`](routed_experts_capturer.md#RoutedExpertsCapturer.create), [`_sync_experts_ids_d2h`](../managers/tp_worker.md#ModelWorker._sync_experts_ids_d2h), [`_global_expert_capturer`](routed_experts_capturer.md#_global_expert_capturer._global_expert_capturer), [`_sync_fwd_experts_buffer_DtoH`](routed_experts_capturer.md#RoutedExpertsCapturer._sync_fwd_experts_buffer_DtoH), [`get_routed_experts`](routed_experts_capturer.md#RoutedExpertsCapturer.get_routed_experts), [`on_forward_end`](routed_experts_capturer.md#RoutedExpertsCapturer.on_forward_end), [`reset`](routed_experts_capturer.md#RoutedExpertsCapturer.reset)

### `_RoutedExpertsCapturerReal`
- def: [`python/sgl_jax/srt/layers/routed_experts_capturer.py:146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L146)
- members:
  - `get_buffer_size_bytes(self)` — [`L240`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L240)
  - `get_routed_experts(self, req_pool_idx: int, seqlen: int, req_to_token_pool: ReqToTokenPool, bid: int)` — [`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L263)
  - `on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch)` — [`L279`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L279)
  - `reset(self)` — [`L315`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L315)
  - `bid` — [`L223`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L223) — ---
  - `dummy_experts_ids` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L183)
  - `enable_host_buffer` — [`L167`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L167)
  - `host_buffer` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L174)
  - `max_padding` — [`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L171)
  - `mesh` — [`L166`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L166)
  - `num_experts_per_tok` — [`L169`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L169)
  - `num_hidden_layers` — [`L168`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L168)
  - `num_tokens` — [`L170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L170)
- protocol/private: `__init__`[`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L149), `_balance_analyzer`[`L190`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L190), `_balance_missing_topk_warned`[`L222`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L222), `_dist_recorder`[`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L207), `_sync_fwd_experts_buffer_DtoH`[`L245`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L245)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, [`ModelConfig`](../configs/model_config.md#ModelConfig), `forward_mode`, [`hf_text_config`](../configs/model_config.md#ModelConfig.hf_text_config), `per_dp_bs_size`, [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`req_to_token`](../mem_cache/memory_pool.md#ReqToTokenPool.req_to_token), [`is_decode`](../model_executor/forward_batch_info.md#ForwardMode.is_decode), [`add_decode_step`](routed_experts_capturer.md#_ExpertBalanceAnalyzer.add_decode_step), [`add_topk_ids`](routed_experts_capturer.md#_ExpertDistributionRecorder.add_topk_ids), `get_original_input_len`, [`logger`](routed_experts_capturer.md#logger), `bid`, [`reset`](routed_experts_capturer.md#_ExpertBalanceAnalyzer.reset), [`reset`](routed_experts_capturer.md#_ExpertDistributionRecorder.reset), `real_bs_per_dp`, `out_cache_loc`, [`output_file`](routed_experts_capturer.md#_ExpertBalanceAnalyzer.output_file), [`RoutedExpertsCapturer`](routed_experts_capturer.md#RoutedExpertsCapturer), [`get_routed_experts_per_token`](routed_experts_capturer.md#get_routed_experts_per_token), [`segment_counter`](routed_experts_capturer.md#_ExpertBalanceAnalyzer.segment_counter), [`_ExpertBalanceAnalyzer`](routed_experts_capturer.md#_ExpertBalanceAnalyzer), [`_ExpertDistributionRecorder`](routed_experts_capturer.md#_ExpertDistributionRecorder), [`_GB`](routed_experts_capturer.md#_GB), [`get_array_size_bytes`](routed_experts_capturer.md#get_array_size_bytes)
- used by: [`create`](routed_experts_capturer.md#RoutedExpertsCapturer.create), [`_sync_fwd_experts_buffer_DtoH`](routed_experts_capturer.md#RoutedExpertsCapturer._sync_fwd_experts_buffer_DtoH), [`get_routed_experts`](routed_experts_capturer.md#RoutedExpertsCapturer.get_routed_experts), [`on_forward_end`](routed_experts_capturer.md#RoutedExpertsCapturer.on_forward_end), [`reset`](routed_experts_capturer.md#RoutedExpertsCapturer.reset)

## Functions
- `_first_present_attr(configs, names)` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L44)
- `extract_routed_experts_from_meta_info(data)` — [`L361`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L361)
- `get_array_size_bytes(t: np.ndarray)` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L30)
- `get_global_experts_capturer()` — [`L352`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L352)
- `get_routed_expert_count(model_config)` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L55) — Total routed experts across name aliases and root/text config; ``None`` if
- `get_routed_experts_per_token(model_config)` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L64) — Routing top-k across name aliases and root/text config; ``None`` if router-less.
- `set_global_experts_capturer(capturer: RoutedExpertsCapturer)` — [`L356`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L356)

## Module values
- `_GB` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L26)
- `_MB` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L27)
- `_TOPK_EXPERT_FIELDS` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L41)
- `_TOTAL_EXPERT_FIELDS` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L40)
- `_global_expert_capturer` — [`L349`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L349)
- `logger` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/routed_experts_capturer.py#L24)

