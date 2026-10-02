---
title: 'Module: easydel/inference/esurge/scheduler/token_budget.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/scheduler/token_budget.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.scheduler.token_budget`/TokenBudgetManager#
symbols:
  TokenBudgetManager.begin_cycle: begin_cycle().
  TokenBudgetManager._remaining: _remaining.
  TokenBudgetManager.max_batch_tokens: max_batch_tokens.
  TokenBudgetManager.remaining: remaining().
  TokenBudgetManager.consume: consume().
  TokenBudgetManager: ''
  TokenBudgetManager.page_size: page_size.
  TokenBudgetManager.safety_margin_tokens: safety_margin_tokens.
  TokenBudgetManager.__post_init__: __post_init__().
---
# Module: [`easydel/inference/esurge/scheduler/token_budget.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py)

## Classes
### `TokenBudgetManager`
- def: [`easydel/inference/esurge/scheduler/token_budget.py:43`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L43)
- doc: Utility that keeps batch-level token usage in sync with KV cache capacity.
- signature: `class TokenBudgetManager:`
- members:
  - `__post_init__(self)` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L84) — Initialize the remaining budget to max_batch_tokens.
  - `begin_cycle(self, cache_manager: CacheManager, num_running_requests: int)` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L100) — Refresh the budget using latest KV cache statistics.
  - `consume(self, requested_tokens: int)` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L133) — Consume tokens from the budget, clamping to the remaining capacity.
  - `remaining(self)` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L92) — Get the remaining token budget for the current cycle.
  - `max_batch_tokens` — [`L75`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L75) — ---
  - `page_size` — [`L78`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L78) — ---
  - `safety_margin_tokens` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L81) — ---
- protocol/private: `_remaining`[`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/token_budget.py#L89)
- uses (calls/refs, reference-scoped): [`page_pool`](../core/manager.md#CacheManager.page_pool), [`get_num_free_pages`](../core/page_pool.md#PagePool.get_num_free_pages), [`CacheManager`](../core/manager.md#CacheManager)
- used by: [`schedule`](scheduler.md#Scheduler.schedule), [`_token_budget_manager`](scheduler.md#Scheduler._token_budget_manager)

