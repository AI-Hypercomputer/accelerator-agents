---
title: 'Module: src/maxdiffusion/trainers/stable_diffusion_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/stable_diffusion_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.stable_diffusion_trainer`/
symbols:
  StableDiffusionTrainer.compile_train_step: StableDiffusionTrainer#compile_train_step().
  StableDiffusionTrainer.training_loop: StableDiffusionTrainer#training_loop().
  StableDiffusionTrainer._profiler: StableDiffusionTrainer#_profiler.
  _train_step: _train_step().
  StableDiffusionTrainer.checkpoint_manager: StableDiffusionTrainer#checkpoint_manager.
  StableDiffusionTrainer.rng: StableDiffusionTrainer#rng.
  StableDiffusionTrainer.load_dataset: StableDiffusionTrainer#load_dataset().
  StableDiffusionTrainer.get_shaped_batch: StableDiffusionTrainer#get_shaped_batch().
  _train_step.compute_loss: _train_step().compute_loss().
  StableDiffusionTrainer: StableDiffusionTrainer#
  StableDiffusionTrainer.__init__: StableDiffusionTrainer#__init__().
  StableDiffusionTrainer.pre_training_steps: StableDiffusionTrainer#pre_training_steps().
  StableDiffusionTrainer.post_training_steps: StableDiffusionTrainer#post_training_steps().
  StableDiffusionTrainer.create_scheduler: StableDiffusionTrainer#create_scheduler().
  StableDiffusionTrainer.get_data_shardings: StableDiffusionTrainer#get_data_shardings().
---
# Module: [`src/maxdiffusion/trainers/stable_diffusion_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py)

## Classes
### `StableDiffusionTrainer`  ·  implements/extends BaseStableDiffusionTrainer
- def: [`src/maxdiffusion/trainers/stable_diffusion_trainer.py:38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L38)
- signature: `class StableDiffusionTrainer(BaseStableDiffusionTrainer):`
- members:
  - `compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings)` — [`L148`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L148)
  - `create_scheduler(self, pipeline, params)` — [`L94`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L94)
  - `get_data_shardings(self)` — [`L100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L100)
  - `get_shaped_batch(self, config, pipeline)` — [`L51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L51) — Return the shape of the batch - this is what eval_shape would return for the
  - `load_dataset(self, pipeline, params, train_states)` — [`L106`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L106)
  - `post_training_steps(self, pipeline, params, train_states)` — [`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L48)
  - `pre_training_steps(self)` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L45)
  - `training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler)` — [`L173`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L173)
  - `checkpoint_manager` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L39)
  - `rng` — [`L149`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L149)
- protocol/private: `__init__`[`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L42), `_profiler`[`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L40)
- uses (calls/refs, reference-scoped): [`_train_step`](stable_diffusion_trainer.md#_train_step)

## Functions
- `_train_step(unet_state, vae_state, text_encoder_state, batch, train_rng, pipeline, params, config)` — [`L253`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L253)
- `compute_loss(state_params)` — [`L262`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/stable_diffusion_trainer.py#L262)

