---
title: 'Module: src/maxdiffusion/trainers/sdxl_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/sdxl_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.sdxl_trainer`/
symbols:
  StableDiffusionXLTrainer.compile_train_step: StableDiffusionXLTrainer#compile_train_step().
  StableDiffusionXLTrainer.load_dataset: StableDiffusionXLTrainer#load_dataset().
  _train_step: _train_step().
  StableDiffusionXLTrainer.rng: StableDiffusionXLTrainer#rng.
  StableDiffusionXLTrainer.training_loop: StableDiffusionXLTrainer#training_loop().
  StableDiffusionXLTrainer.get_shaped_batch: StableDiffusionXLTrainer#get_shaped_batch().
  StableDiffusionXLTrainer.prepare_sample: StableDiffusionXLTrainer#prepare_sample().
  _train_step.compute_loss: _train_step().compute_loss().
  StableDiffusionXLTrainer: StableDiffusionXLTrainer#
  StableDiffusionXLTrainer.__init__: StableDiffusionXLTrainer#__init__().
  StableDiffusionXLTrainer.text_encoder_2_learning_rate_scheduler: StableDiffusionXLTrainer#text_encoder_2_learning_rate_scheduler.
  StableDiffusionXLTrainer.post_training_steps: StableDiffusionXLTrainer#post_training_steps().
  StableDiffusionXLTrainer.get_data_shardings: StableDiffusionXLTrainer#get_data_shardings().
---
# Module: [`src/maxdiffusion/trainers/sdxl_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py)

## Classes
### `StableDiffusionXLTrainer`  ·  implements/extends StableDiffusionTrainer
- def: [`src/maxdiffusion/trainers/sdxl_trainer.py:49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L49)
- signature: `class StableDiffusionXLTrainer(StableDiffusionTrainer):`
- members:
  - `compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings)` — [`L178`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L178)
  - `get_data_shardings(self)` — [`L93`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L93)
  - `get_shaped_batch(self, config, pipeline)` — [`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L62) — Return the shape of the batch - this is what eval_shape would return for the
  - `load_dataset(self, pipeline, params, train_states)` — [`L109`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L109)
  - `post_training_steps(self, pipeline, params, train_states)` — [`L59`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L59)
  - `prepare_sample(features)` — [`L151`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L151)
  - `training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler)` — [`L209`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L209)
  - `rng` — [`L179`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L179)
  - `text_encoder_2_learning_rate_scheduler` — [`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L54)
- protocol/private: `__init__`[`L51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L51)
- uses (calls/refs, reference-scoped): [`_train_step`](sdxl_trainer.md#_train_step)

## Functions
- `_train_step(unet_state, vae_state, text_encoder_state, text_encoder_2_state, batch, train_rng, pipeline, params, config)` — [`L294`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L294)
- `compute_loss(state_params)` — [`L303`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/sdxl_trainer.py#L303)

