---
title: 'Module: src/maxtext/utils/rampup_batch.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/rampup_batch.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.rampup_batch`/
symbols:
  RampupBatchManager.update: RampupBatchManager#update().
  RampupBatchManager.global_batch_size_current: RampupBatchManager#global_batch_size_current.
  RampupBatchManager.num_accum_samples: RampupBatchManager#num_accum_samples.
  RampupBatchManager.__init__: RampupBatchManager#__init__().
  RampupBatchManager._recover_states: RampupBatchManager#_recover_states().
  RampupBatchManager.total_rampup_steps: RampupBatchManager#total_rampup_steps.
  create_rampup_manager: create_rampup_manager().
  RampupBatchManager.global_batch_size_end: RampupBatchManager#global_batch_size_end.
  RampupBatchManager: RampupBatchManager#
  RampupBatchManager._verify_inputs: RampupBatchManager#_verify_inputs().
  RampupBatchManager._init_values: RampupBatchManager#_init_values().
  RampupBatchManager.samples_per_increment: RampupBatchManager#samples_per_increment.
  RampupBatchManager.global_batch_size_start: RampupBatchManager#global_batch_size_start.
  RampupBatchManager.increment: RampupBatchManager#increment.
  RampupBatchManager.total_used_samples: RampupBatchManager#total_used_samples.
  RampupBatchManager._compute_total_rampup_steps: RampupBatchManager#_compute_total_rampup_steps().
  RampupBatchManager.global_rampup_samples: RampupBatchManager#global_rampup_samples.
---
# Module: [`src/maxtext/utils/rampup_batch.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py)

## Classes
### `RampupBatchManager`
- def: [`src/maxtext/utils/rampup_batch.py:21`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L21)
- doc: A stateful class tracking current batch size given train step
- signature: `class RampupBatchManager:`
- members:
  - `_compute_total_rampup_steps(self, config)` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L64) — Compute total number of rampup steps
  - `_init_values(self, config)` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L50) — Initialize rampup batch related parameters
  - `_recover_states(self, step_num)` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L82) — Recover the number of samples already used
  - `_verify_inputs(self, config)` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L34) — Verify the rampup batch related inputs.
  - `update(self)` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L90) — Update values when load_batch is called
  - `global_batch_size_current` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L60)
  - `global_batch_size_end` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L56)
  - `global_batch_size_start` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L57)
  - `global_rampup_samples` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L59)
  - `increment` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L58)
  - `num_accum_samples` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L29)
  - `samples_per_increment` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L54)
  - `total_rampup_steps` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L61)
  - `total_used_samples` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L62)
- protocol/private: `__init__`[`L26`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L26)
- used by: [`create_rampup_manager`](rampup_batch.md#create_rampup_manager)

## Functions
- `create_rampup_manager(config, checkpoint_manager)` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/utils/rampup_batch.py#L103)

