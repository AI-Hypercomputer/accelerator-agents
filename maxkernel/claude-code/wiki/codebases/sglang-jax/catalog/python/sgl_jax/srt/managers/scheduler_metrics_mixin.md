---
title: 'Module: python/sgl_jax/srt/managers/scheduler_metrics_mixin.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/scheduler_metrics_mixin.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.scheduler_metrics_mixin`/
symbols:
  SchedulerMetricsMixin.log_decode_stats: SchedulerMetricsMixin#log_decode_stats().
  SchedulerMetricsMixin.log_prefill_stats: SchedulerMetricsMixin#log_prefill_stats().
  SchedulerMetricsMixin: SchedulerMetricsMixin#
  RECORD_STEP_TIME: RECORD_STEP_TIME.
  SchedulerMetricsMixin.init_metrics: SchedulerMetricsMixin#init_metrics().
  logger: logger.
---
# Module: [`python/sgl_jax/srt/managers/scheduler_metrics_mixin.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py)

## Classes
### `SchedulerMetricsMixin`
- def: [`python/sgl_jax/srt/managers/scheduler_metrics_mixin.py:20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py#L20)
- signature: `class SchedulerMetricsMixin:`
- members:
  - `init_metrics(self: Scheduler)` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py#L21)
  - `log_decode_stats(self: Scheduler, running_batch: ScheduleBatch = None)` — [`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py#L89)
  - `log_prefill_stats(self: Scheduler, adder: PrefillAdder, can_run_list: list[Req], running_bs: int)` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py#L31)
- uses (calls/refs, reference-scoped): `reqs`, `reqs_info`, `Req`, `dp_size`, [`running_batch`](scheduler.md#Scheduler.running_batch), [`Scheduler`](scheduler.md#Scheduler), [`dp_size`](scheduler.md#Scheduler.dp_size), [`spec_algorithm`](scheduler.md#Scheduler.spec_algorithm), `ScheduleBatch`, [`log_input_tokens`](schedule_policy.md#PrefillAdder.log_input_tokens), [`is_none`](../speculative/spec_info.md#SpeculativeAlgorithm.is_none), [`_get_swa_token_info`](scheduler.md#Scheduler._get_swa_token_info), [`waiting_queue`](scheduler.md#Scheduler.waiting_queue), [`server_args`](scheduler.md#Scheduler.server_args), [`_get_token_info`](scheduler.md#Scheduler._get_token_info), [`can_run_list`](schedule_policy.md#PrefillAdder.can_run_list), `batch_size`, [`is_hybrid`](scheduler.md#Scheduler.is_hybrid), `cache_miss_count`, [`num_generated_tokens`](scheduler.md#Scheduler.num_generated_tokens), [`PrefillAdder`](schedule_policy.md#PrefillAdder), [`accept_token`](scheduler.md#Scheduler.accept_token), [`draft_token`](scheduler.md#Scheduler.draft_token), [`spec_num_forward_ct`](scheduler.md#Scheduler.spec_num_forward_ct), [`RECORD_STEP_TIME`](scheduler_metrics_mixin.md#RECORD_STEP_TIME), [`decode_log_interval`](../server_args.md#ServerArgs.decode_log_interval), [`last_decode_stats_tic`](scheduler.md#Scheduler.last_decode_stats_tic), [`last_prefill_stats_tic`](scheduler.md#Scheduler.last_prefill_stats_tic), [`last_prefill_tokens`](scheduler.md#Scheduler.last_prefill_tokens), [`log_hit_tokens`](schedule_policy.md#PrefillAdder.log_hit_tokens), [`logger`](scheduler_metrics_mixin.md#logger)
- used by: [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`get_new_batch_prefill`](scheduler.md#Scheduler.get_new_batch_prefill), [`Scheduler`](scheduler.md#Scheduler), [`parent_process`](scheduler.md#Scheduler.parent_process)

## Module values
- `RECORD_STEP_TIME` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py#L17)
- `logger` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_metrics_mixin.py#L15)

