---
title: 'Module: src/maxdiffusion/trainers/wan_vace_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/wan_vace_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.wan_vace_trainer`/
symbols:
  WanVaceTrainer.load_dataset: WanVaceTrainer#load_dataset().
  train_step: train_step().
  step_optimizer: step_optimizer().
  eval_step: eval_step().
  WanVaceTrainer.get_train_step: WanVaceTrainer#get_train_step().
  WanVaceTrainer.get_eval_step: WanVaceTrainer#get_eval_step().
  WanVaceTrainer.prepare_sample_train: WanVaceTrainer#prepare_sample_train().
  WanVaceTrainer.prepare_sample_eval: WanVaceTrainer#prepare_sample_eval().
  step_optimizer.loss_fn: step_optimizer().loss_fn().
  eval_step.loss_fn: eval_step().loss_fn().
  WanVaceTrainer: WanVaceTrainer#
  WanVaceTrainer._get_checkpointer: WanVaceTrainer#_get_checkpointer().
  WanVaceTrainer.post_training_steps: WanVaceTrainer#post_training_steps().
  WanVaceTrainer.get_data_shardings: WanVaceTrainer#get_data_shardings().
  WanVaceTrainer.get_eval_data_shardings: WanVaceTrainer#get_eval_data_shardings().
---
# Module: [`src/maxdiffusion/trainers/wan_vace_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py)

## Classes
### `WanVaceTrainer`  ·  implements/extends BaseWanTrainer
- def: [`src/maxdiffusion/trainers/wan_vace_trainer.py:30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L30)
- signature: `class WanVaceTrainer(BaseWanTrainer):`
- members:
  - `get_data_shardings(self, mesh)` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L38)
  - `get_eval_data_shardings(self, mesh)` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L47)
  - `get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings)` — [`L128`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L128)
  - `get_train_step(self, pipeline, mesh, state_shardings, data_shardings)` — [`L120`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L120)
  - `load_dataset(self, mesh, pipeline=None, is_training=True)` — [`L57`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L57)
  - `post_training_steps(self, pipeline, params, train_states, msg="")` — [`L35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L35)
  - `prepare_sample_eval(features)` — [`L96`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L96)
  - `prepare_sample_train(features)` — [`L86`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L86)
- protocol/private: `_get_checkpointer`[`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L32)
- uses (calls/refs, reference-scoped): [`eval_step`](wan_vace_trainer.md#eval_step), [`train_step`](wan_vace_trainer.md#train_step)

## Functions
- `eval_step(state, data, rng, scheduler_state, scheduler, config)` — [`L200`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L200) — Computes the evaluation loss for a single batch without updating model weights.
- `loss_fn(params)` — [`L146`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L146)
- `loss_fn(params, latents, encoder_hidden_states, timesteps, rng, conditioning_latents)` — [`L207`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L207)
- `step_optimizer(state, data, rng, scheduler_state, scheduler, config)` — [`L140`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L140)
- `train_step(state, data, rng, scheduler_state, scheduler, config)` — [`L136`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/wan_vace_trainer.py#L136)

