---
title: 'Module: easydel/trainers/trainer/modeling_output.py'
type: catalog
provenance: extracted
module: easydel/trainers/trainer/modeling_output.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.trainer.modeling_output`/
symbols:
  TrainerOutput.state: TrainerOutput#state.
  TrainerOutput.shard_fns: TrainerOutput#shard_fns.
  CallFN.CallFN: CallFN.CallFN.
  TrainerOutput: TrainerOutput#
  TrainerOutput.mesh: TrainerOutput#mesh.
  TrainerOutput.checkpoint_manager: TrainerOutput#checkpoint_manager.
  TrainerOutput.last_save_file_name: TrainerOutput#last_save_file_name.
  TrainerOutput.checkpoint_path: TrainerOutput#checkpoint_path.
---
# Module: [`easydel/trainers/trainer/modeling_output.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py)

## Classes
### `TrainerOutput`
- def: [`easydel/trainers/trainer/modeling_output.py:37`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L37)
- doc: Output from a training run.
- signature: `class TrainerOutput:`
- members:
  - `checkpoint_manager` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L54)
  - `checkpoint_path` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L57)
  - `last_save_file_name` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L56)
  - `mesh` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L53)
  - `shard_fns` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L55)
  - `state` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L52)
- uses (calls/refs, reference-scoped): [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`CallFN`](modeling_output.md#CallFN.CallFN)

## Module values
- `CallFN` — [`L33`](../../../../../../../raw/code/EasyDeL/easydel/trainers/trainer/modeling_output.py#L33)

