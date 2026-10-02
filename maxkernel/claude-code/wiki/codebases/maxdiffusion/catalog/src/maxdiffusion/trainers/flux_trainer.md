---
title: 'Module: src/maxdiffusion/trainers/flux_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/flux_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.flux_trainer`/
symbols:
  FluxTrainer.per_device_tflops: FluxTrainer#per_device_tflops.
  FluxTrainer.load_dataset: FluxTrainer#load_dataset().
  FluxTrainer.compile_train_step: FluxTrainer#compile_train_step().
  FluxTrainer.training_loop: FluxTrainer#training_loop().
  FluxTrainer.start_training: FluxTrainer#start_training().
  FluxTrainer.create_scheduler: FluxTrainer#create_scheduler().
  FluxTrainer._profiler: FluxTrainer#_profiler.
  FluxTrainer.calculate_tflops: FluxTrainer#calculate_tflops().
  _train_step: _train_step().
  FluxTrainer.rng: FluxTrainer#rng.
  FluxTrainer: FluxTrainer#
  FluxTrainer.post_training_steps: FluxTrainer#post_training_steps().
  FluxTrainer.get_shaped_batch: FluxTrainer#get_shaped_batch().
  FluxTrainer.get_data_shardings: FluxTrainer#get_data_shardings().
  FluxTrainer.tokenize_captions: FluxTrainer#tokenize_captions().
  FluxTrainer.transform_images: FluxTrainer#transform_images().
  _train_step.compute_loss: _train_step().compute_loss().
  FluxTrainer.__init__: FluxTrainer#__init__().
  FluxTrainer.text_encoder_2_learning_rate_scheduler: FluxTrainer#text_encoder_2_learning_rate_scheduler.
---
# Module: [`src/maxdiffusion/trainers/flux_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py)

## Classes
### `FluxTrainer`  ·  implements/extends FluxCheckpointer
- def: [`src/maxdiffusion/trainers/flux_trainer.py:51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L51)
- signature: `class FluxTrainer(FluxCheckpointer):`
- members:
  - `calculate_tflops(self, pipeline)` — [`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L74)
  - `compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings)` — [`L298`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L298)
  - `create_scheduler(self, pipeline, params)` — [`L65`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L65)
  - `get_data_shardings(self)` — [`L185`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L185)
  - `get_shaped_batch(self, config, pipeline=None)` — [`L149`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L149) — Return the shape of the batch - this is what eval_shape would return for the
  - `load_dataset(self, pipeline, params, train_states)` — [`L245`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L245)
  - `post_training_steps(self, pipeline, params, train_states, msg="")` — [`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L62)
  - `start_training(self)` — [`L79`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L79)
  - `tokenize_captions(examples, caption_column, encoder)` — [`L199`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L199)
  - `training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler)` — [`L328`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L328)
  - `transform_images(examples, image_column, image_resolution, vae_encode, pack_latents, prepare_latent_imgage_ids)` — [`L211`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L211) — Preprocess images to latents.
  - `per_device_tflops` — [`L136`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L136)
  - `rng` — [`L299`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L299)
  - `text_encoder_2_learning_rate_scheduler` — [`L57`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L57)
- protocol/private: `__init__`[`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L54), `_profiler`[`L52`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L52)
- uses (calls/refs, reference-scoped): [`FlaxEulerDiscreteScheduler`](../schedulers/scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler), [`_train_step`](flux_trainer.md#_train_step), [`from_pretrained`](../schedulers/scheduling_utils_flax.md#FlaxSchedulerMixin.from_pretrained)

## Functions
- `_train_step(flux_state, batch, train_rng, guidance_vec, pipeline, scheduler, config)` — [`L403`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L403)
- `compute_loss(state_params)` — [`L408`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/flux_trainer.py#L408)

