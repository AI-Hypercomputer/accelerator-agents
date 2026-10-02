---
title: 'Module: src/maxdiffusion/trainers/wan_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/wan_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.wan_trainer`/
symbols:
  WanTrainer.load_dataset: WanTrainer#load_dataset().
  train_step: train_step().
  step_optimizer: step_optimizer().
  eval_step: eval_step().
  WanTrainer.get_train_step: WanTrainer#get_train_step().
  WanTrainer.get_eval_step: WanTrainer#get_eval_step().
  WanTrainer.prepare_sample_train: WanTrainer#prepare_sample_train().
  WanTrainer.prepare_sample_eval: WanTrainer#prepare_sample_eval().
  step_optimizer.loss_fn: step_optimizer().loss_fn().
  eval_step.loss_fn: eval_step().loss_fn().
  WanTrainer: WanTrainer#
  WanTrainer._get_checkpointer: WanTrainer#_get_checkpointer().
  WanTrainer.get_data_shardings: WanTrainer#get_data_shardings().
  WanTrainer.get_eval_data_shardings: WanTrainer#get_eval_data_shardings().
---
# Module: [`src/maxdiffusion/trainers/wan_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py)

## Classes
### `WanTrainer`  ·  implements/extends BaseWanTrainer
- def: [`src/maxdiffusion/trainers/wan_trainer.py:30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L30)
- signature: `class WanTrainer(BaseWanTrainer):`
- members:
  - `get_data_shardings(self, mesh)` — [`L35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L35)
  - `get_eval_data_shardings(self, mesh)` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L40)
  - `get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings)` — [`L124`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L124)
  - `get_train_step(self, pipeline, mesh, state_shardings, data_shardings)` — [`L116`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L116)
  - `load_dataset(self, mesh, pipeline=None, is_training=True)` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L45) — Load dataset - supports both real tfrecord and synthetic data.
  - `prepare_sample_eval(features)` — [`L98`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L98)
  - `prepare_sample_train(features)` — [`L93`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L93)
- protocol/private: `_get_checkpointer`[`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L32)
- uses (calls/refs, reference-scoped): [`eval_step`](wan_trainer.md#eval_step), [`train_step`](wan_trainer.md#train_step)

## Functions
- `eval_step(state, data, rng, scheduler_state, scheduler, config)` — [`L192`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L192) — Computes the evaluation loss for a single batch without updating model weights.
- `loss_fn(params)` — [`L142`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L142)
- `loss_fn(params, latents, encoder_hidden_states, timesteps, rng)` — [`L199`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L199)
- `step_optimizer(state, data, rng, scheduler_state, scheduler, config)` — [`L136`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L136)
- `train_step(state, data, rng, scheduler_state, scheduler, config)` — [`L132`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py#L132)

