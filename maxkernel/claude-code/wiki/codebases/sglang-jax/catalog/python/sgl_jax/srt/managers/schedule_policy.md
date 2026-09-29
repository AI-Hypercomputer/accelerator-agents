---
title: 'Module: python/sgl_jax/srt/managers/schedule_policy.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/schedule_policy.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.schedule_policy`/
symbols:
  PrefillAdder.add_one_req: PrefillAdder#add_one_req().
  SchedulePolicy._compute_prefix_matches: SchedulePolicy#_compute_prefix_matches().
  PrefillAdder.add_one_req_ignore_eos: PrefillAdder#add_one_req_ignore_eos().
  SchedulePolicy.calc_priority: SchedulePolicy#calc_priority().
  PrefillAdder.add_chunked_req: PrefillAdder#add_chunked_req().
  PrefillAdder._update_prefill_budget: PrefillAdder#_update_prefill_budget().
  PrefillAdder.budget_state: PrefillAdder#budget_state().
  PrefillAdder.log_input_tokens: PrefillAdder#log_input_tokens.
  PrefillAdder.rem_total_tokens_for_dp: PrefillAdder#rem_total_tokens_for_dp().
  PrefillAdder._budget_state_after_add: PrefillAdder#_budget_state_after_add().
  PrefillAdder.rem_chunk_tokens_list: PrefillAdder#rem_chunk_tokens_list.
  PrefillAdder.cur_rem_tokens_for_dp: PrefillAdder#cur_rem_tokens_for_dp().
  AddReqResult: AddReqResult#
  SchedulePolicy._determine_active_policy: SchedulePolicy#_determine_active_policy().
  SchedulePolicy._sort_by_dfs_weight: SchedulePolicy#_sort_by_dfs_weight().
  PrefillAdder.is_hybrid: PrefillAdder#is_hybrid.
  PrefillAdder.rem_swa_tokens_for_dp: PrefillAdder#rem_swa_tokens_for_dp().
  PrefillAdder.can_run_list: PrefillAdder#can_run_list.
  SchedulePolicy._get_dfs_priority: SchedulePolicy#_get_dfs_priority().
  PrefillAdder._swa_budget_for_req: PrefillAdder#_swa_budget_for_req().
  SchedulePolicy._validate_and_adjust_policy: SchedulePolicy#_validate_and_adjust_policy().
  PrefillAdder._lock_node: PrefillAdder#_lock_node().
  SchedulePolicy._calc_weight: SchedulePolicy#_calc_weight().
  PrefillAdder.add_req_state: PrefillAdder#add_req_state().
  PrefillAdder.tree_cache: PrefillAdder#tree_cache.
  SchedulePolicy: SchedulePolicy#
  AddReqResult.NO_TOKEN: AddReqResult#NO_TOKEN.
  PrefillAdder.page_size: PrefillAdder#page_size.
  PrefillAdder.req_states: PrefillAdder#req_states.
  SchedulePolicy._sort_by_longest_prefix: SchedulePolicy#_sort_by_longest_prefix().
  SchedulePolicy._sort_by_longest_output: SchedulePolicy#_sort_by_longest_output().
  PrefillAdder.rem_chunk_tokens: PrefillAdder#rem_chunk_tokens.
  PrefillAdder.ceil_paged_tokens: PrefillAdder#ceil_paged_tokens().
  CLIP_MAX_NEW_TOKENS_ESTIMATION: CLIP_MAX_NEW_TOKENS_ESTIMATION.
  CacheAgnosticPolicy: CacheAgnosticPolicy#
  PrefillAdder.__init__: PrefillAdder#__init__().
  CacheAwarePolicy: CacheAwarePolicy#
  SchedulePolicy.Policy: SchedulePolicy#Policy.
  AddReqResult.OTHER: AddReqResult#OTHER.
  PrefillAdder.token_to_kv_pool_allocator: PrefillAdder#token_to_kv_pool_allocator.
  PrefillAdder.rem_total_tokens: PrefillAdder#rem_total_tokens().
  PrefillAdder.cur_rem_tokens: PrefillAdder#cur_rem_tokens().
  SchedulePolicy.policy: SchedulePolicy#policy.
  SchedulePolicy.waiting_queue_radix_tree: SchedulePolicy#waiting_queue_radix_tree.
  PrefillAdder.rem_input_tokens: PrefillAdder#rem_input_tokens.
  CacheAgnosticPolicy.FCFS: CacheAgnosticPolicy#FCFS.
  PrefillAdder: PrefillAdder#
  PrefillAdder.new_chunked_reqs: PrefillAdder#new_chunked_reqs.
  SchedulePolicy._sort_randomly: SchedulePolicy#_sort_randomly().
  PrefillAdder.align_page_size: PrefillAdder#align_page_size().
  IGNORE_EOS_RESERVE_TOKENS: IGNORE_EOS_RESERVE_TOKENS.
  AddReqResult.CONTINUE: AddReqResult#CONTINUE.
  PrefillAdder.dp_size: PrefillAdder#dp_size.
  PrefillAdder.rem_total_token_offset: PrefillAdder#rem_total_token_offset.
  SchedulePolicy.__init__: SchedulePolicy#__init__().
  CacheAwarePolicy.LPM: CacheAwarePolicy#LPM.
  SchedulePolicy.tree_cache: SchedulePolicy#tree_cache.
  PrefillAdder.running_batch: PrefillAdder#running_batch.
  PrefillAdder.new_token_ratio: PrefillAdder#new_token_ratio.
  PrefillAdder.cur_rem_token_offset: PrefillAdder#cur_rem_token_offset.
  PrefillAdder.log_hit_tokens: PrefillAdder#log_hit_tokens.
  PrefillAdder.rem_swa_token_offset: PrefillAdder#rem_swa_token_offset.
  IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD: IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD.
  IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD: IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD.
  CacheAwarePolicy.DFS_WEIGHT: CacheAwarePolicy#DFS_WEIGHT.
  CacheAgnosticPolicy.LOF: CacheAgnosticPolicy#LOF.
  CacheAgnosticPolicy.RANDOM: CacheAgnosticPolicy#RANDOM.
  logger: logger.
---
# Module: [`python/sgl_jax/srt/managers/schedule_policy.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py)

## Classes
### `AddReqResult`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/managers/schedule_policy.py:256`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L256)
- signature: `class AddReqResult(Enum):`
- members:
  - `CONTINUE` — [`L257`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L257)
  - `NO_TOKEN` — [`L258`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L258)
  - `OTHER` — [`L259`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L259)
- used by: [`get_new_batch_prefill`](scheduler.md#Scheduler.get_new_batch_prefill), [`add_one_req`](schedule_policy.md#PrefillAdder.add_one_req), [`add_one_req_ignore_eos`](schedule_policy.md#PrefillAdder.add_one_req_ignore_eos), [`budget_state`](schedule_policy.md#PrefillAdder.budget_state), [`_budget_state_after_add`](schedule_policy.md#PrefillAdder._budget_state_after_add)

### `CacheAgnosticPolicy`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/managers/schedule_policy.py:62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L62)
- doc: Scheduling policies that are not aware of the tree cache.
- signature: `class CacheAgnosticPolicy(Enum):`
- members:
  - `FCFS` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L65)
  - `LOF` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L66)
  - `RANDOM` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L67)
- used by: [`calc_priority`](schedule_policy.md#SchedulePolicy.calc_priority), [`_determine_active_policy`](schedule_policy.md#SchedulePolicy._determine_active_policy), [`_validate_and_adjust_policy`](schedule_policy.md#SchedulePolicy._validate_and_adjust_policy), [`Policy`](schedule_policy.md#SchedulePolicy.Policy)

### `CacheAwarePolicy`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/managers/schedule_policy.py:55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L55)
- doc: Scheduling policies that are aware of the tree cache.
- signature: `class CacheAwarePolicy(Enum):`
- members:
  - `DFS_WEIGHT` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L59)
  - `LPM` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L58)
- used by: [`_compute_prefix_matches`](schedule_policy.md#SchedulePolicy._compute_prefix_matches), [`calc_priority`](schedule_policy.md#SchedulePolicy.calc_priority), [`_determine_active_policy`](schedule_policy.md#SchedulePolicy._determine_active_policy), [`_validate_and_adjust_policy`](schedule_policy.md#SchedulePolicy._validate_and_adjust_policy), [`Policy`](schedule_policy.md#SchedulePolicy.Policy)

### `PrefillAdder`
- def: [`python/sgl_jax/srt/managers/schedule_policy.py:262`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L262)
- signature: `class PrefillAdder:`
- members:
  - `_swa_budget_for_req(self, extend_input_len: int, dp_rank: int)` — [`L372`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L372) — SWA pool budget per request.
  - `add_chunked_req(self, req: Req)` — [`L434`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L434)
  - `add_one_req(self, req: Req)` — [`L588`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L588) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `add_one_req_ignore_eos(self, req: Req)` — [`L501`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L501) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `add_req_state(r, insert_sort=False)` — [`L513`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L513)
  - `align_page_size(self, size: int)` — [`L406`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L406)
  - `budget_state(self)` — [`L409`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L409)
  - `ceil_paged_tokens(self, tokens: int)` — [`L403`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L403)
  - `cur_rem_tokens(self)` — [`L396`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L396) — Global current remaining tokens (minimum across all DP ranks).
  - `cur_rem_tokens_for_dp(self, dp_rank: int)` — [`L353`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L353) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `rem_swa_tokens_for_dp(self, dp_rank: int)` — [`L365`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L365)
  - `rem_total_tokens(self)` — [`L388`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L388) — Global remaining total tokens (minimum across all DP ranks).
  - `rem_total_tokens_for_dp(self, dp_rank: int)` — [`L330`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L330) — Calculate remaining total tokens for a specific DP rank.
  - `can_run_list` — [`L306`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L306) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `cur_rem_token_offset` — [`L303`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L303)
  - `dp_size` — [`L280`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L280)
  - `is_hybrid` — [`L327`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L327) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `log_hit_tokens` — [`L308`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L308)
  - `log_input_tokens` — [`L309`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L309)
  - `new_chunked_reqs` — [`L307`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L307)
  - `new_token_ratio` — [`L279`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L279)
  - `page_size` — [`L275`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L275)
  - `rem_chunk_tokens` — [`L293`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L293)
  - `rem_chunk_tokens_list` — [`L295`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L295) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `rem_input_tokens` — [`L292`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L292)
  - `rem_swa_token_offset` — [`L328`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L328)
  - `rem_total_token_offset` — [`L302`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L302)
  - `req_states` — [`L305`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L305)
  - `running_batch` — [`L278`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L278)
  - `token_to_kv_pool_allocator` — [`L277`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L277)
  - `tree_cache` — [`L276`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L276)
- protocol/private: `__init__`[`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L263), `_budget_state_after_add`[`L423`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L423), `_lock_node`[`L494`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L494), `_update_prefill_budget`[`L474`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L474)
- uses (calls/refs, reference-scoped): `reqs`, `reqs_info`, `Req`, `host_hit_length`, `sampling_params`, `output_ids`, `prefix_indices`, `ScheduleBatch`, [`BasePrefixCache`](../mem_cache/base_prefix_cache.md#BasePrefixCache), `dp_rank`, `extend_input_len`, `fill_ids`, [`BaseTokenToKVPoolAllocator`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator), [`available_size`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator.available_size), `last_node`, [`TreeNode`](../mem_cache/radix_cache.md#TreeNode), [`max_new_tokens`](../sampling/sampling_params.md#SamplingParams.max_new_tokens), [`AddReqResult`](schedule_policy.md#AddReqResult), [`SWATokenToKVPoolAllocator`](../mem_cache/allocator.md#SWATokenToKVPoolAllocator), `last_matched_prefix_len`, [`inc_lock_ref`](../mem_cache/base_prefix_cache.md#BasePrefixCache.inc_lock_ref), [`dec_lock_ref`](../mem_cache/base_prefix_cache.md#BasePrefixCache.dec_lock_ref), [`evictable_size`](../mem_cache/base_prefix_cache.md#BasePrefixCache.evictable_size), [`full_evictable_size`](../mem_cache/base_prefix_cache.md#BasePrefixCache.full_evictable_size), [`to_dec_params`](../mem_cache/base_prefix_cache.md#IncLockRefResult.to_dec_params), [`swa_evictable_size`](../mem_cache/base_prefix_cache.md#BasePrefixCache.swa_evictable_size), [`NO_TOKEN`](schedule_policy.md#AddReqResult.NO_TOKEN), [`CLIP_MAX_NEW_TOKENS_ESTIMATION`](schedule_policy.md#CLIP_MAX_NEW_TOKENS_ESTIMATION), `swa_uuid_for_lock`, [`OTHER`](schedule_policy.md#AddReqResult.OTHER), [`swa_uuid_for_lock`](../mem_cache/base_prefix_cache.md#IncLockRefResult.swa_uuid_for_lock), [`CONTINUE`](schedule_policy.md#AddReqResult.CONTINUE), [`IGNORE_EOS_RESERVE_TOKENS`](schedule_policy.md#IGNORE_EOS_RESERVE_TOKENS), [`ignore_eos`](../sampling/sampling_params.md#SamplingParams.ignore_eos)
- used by: [`get_new_batch_prefill`](scheduler.md#Scheduler.get_new_batch_prefill), [`log_prefill_stats`](scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_prefill_stats)

### `SchedulePolicy`
- def: [`python/sgl_jax/srt/managers/schedule_policy.py:70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L70)
- signature: `class SchedulePolicy:`
- members:
  - `_compute_prefix_matches(self, waiting_queue: list[Req], policy: CacheAwarePolicy)` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L140) — Computes and caches the matching prefixes for requests in the waiting queue, — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `_sort_by_dfs_weight(waiting_queue: list[Req], tree_cache: BasePrefixCache)` — [`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L207) — Sorts the waiting queue based on a depth-first search weighting.
  - `_sort_by_longest_output(waiting_queue: list[Req])` — [`L227`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L227) — Sorts the waiting queue based on the longest output (max_new_tokens).
  - `_sort_by_longest_prefix(waiting_queue: list[Req], temporary_deprioritized: set[int])` — [`L196`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L196) — Sorts the waiting queue based on the longest prefix match.
  - `_sort_randomly(waiting_queue: list[Req])` — [`L232`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L232) — Shuffles the waiting queue randomly.
  - `_validate_and_adjust_policy(self, policy: str, tree_cache: BasePrefixCache)` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L124) — Validates the policy and adjusts it if necessary based on tree cache settings.
  - `calc_priority(self, waiting_queue: list[Req])` — [`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L89) — documented in [python-sgl_jax-srt-managers-schedule_policy](../../../../../concepts/python-sgl_jax-srt-managers-schedule_policy.md)
  - `Policy` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L71)
  - `policy` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L78)
  - `tree_cache` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L79)
  - `waiting_queue_radix_tree` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L82)
- protocol/private: `__init__`[`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L73), `_calc_weight`[`L237`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L237), `_determine_active_policy`[`L118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L118), `_get_dfs_priority`[`L243`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L243)
- uses (calls/refs, reference-scoped): `Req`, `host_hit_length`, `sampling_params`, [`RadixKey`](../mem_cache/radix_cache.md#RadixKey), `prefix_indices`, `rid`, [`match_prefix`](../mem_cache/radix_cache.md#RadixCache.match_prefix), [`BasePrefixCache`](../mem_cache/base_prefix_cache.md#BasePrefixCache), `dp_rank`, [`insert`](../mem_cache/radix_cache.md#RadixCache.insert), [`InsertParams`](../mem_cache/base_prefix_cache.md#InsertParams), [`MatchPrefixParams`](../mem_cache/base_prefix_cache.md#MatchPrefixParams), `last_node`, [`TreeNode`](../mem_cache/radix_cache.md#TreeNode), [`max_new_tokens`](../sampling/sampling_params.md#SamplingParams.max_new_tokens), `adjust_max_prefix_ids`, [`match_prefix`](../mem_cache/base_prefix_cache.md#BasePrefixCache.match_prefix), [`last_device_node`](../mem_cache/base_prefix_cache.md#MatchResult.last_device_node), [`device_indices`](../mem_cache/base_prefix_cache.md#MatchResult.device_indices), [`children`](../mem_cache/radix_cache.md#TreeNode.children), [`key`](../mem_cache/base_prefix_cache.md#MatchPrefixParams.key), [`key`](../mem_cache/base_prefix_cache.md#InsertParams.key), [`last_host_node`](../mem_cache/base_prefix_cache.md#MatchResult.last_host_node), `extra_key`, [`value`](../mem_cache/base_prefix_cache.md#InsertParams.value), `last_host_node`, [`CacheAgnosticPolicy`](schedule_policy.md#CacheAgnosticPolicy), [`CacheAwarePolicy`](schedule_policy.md#CacheAwarePolicy), [`host_hit_length`](../mem_cache/base_prefix_cache.md#MatchResult.host_hit_length), [`RadixCache`](../mem_cache/radix_cache.md#RadixCache), [`FCFS`](schedule_policy.md#CacheAgnosticPolicy.FCFS), [`LPM`](schedule_policy.md#CacheAwarePolicy.LPM), [`reset`](../mem_cache/radix_cache.md#RadixCache.reset), [`DFS_WEIGHT`](schedule_policy.md#CacheAwarePolicy.DFS_WEIGHT), [`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`](schedule_policy.md#IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD), [`IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`](schedule_policy.md#IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD), [`LOF`](schedule_policy.md#CacheAgnosticPolicy.LOF), [`RANDOM`](schedule_policy.md#CacheAgnosticPolicy.RANDOM)
- used by: [`get_new_batch_prefill`](scheduler.md#Scheduler.get_new_batch_prefill), [`policy`](scheduler.md#Scheduler.policy)

## Module values
- `CLIP_MAX_NEW_TOKENS_ESTIMATION` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L32)
- `IGNORE_EOS_RESERVE_TOKENS` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L52)
- `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L40)
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L47)
- `logger` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/schedule_policy.py#L26)

