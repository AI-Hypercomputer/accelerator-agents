---
title: 'Module: src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.stable_diffusion.safety_checker_flax`/
symbols:
  FlaxStableDiffusionSafetyCheckerModule.__call__: FlaxStableDiffusionSafetyCheckerModule#__call__().
  FlaxStableDiffusionSafetyCheckerModule.visual_projection: FlaxStableDiffusionSafetyCheckerModule#visual_projection.
  FlaxStableDiffusionSafetyChecker.__init__: FlaxStableDiffusionSafetyChecker#__init__().
  FlaxStableDiffusionSafetyCheckerModule.vision_model: FlaxStableDiffusionSafetyCheckerModule#vision_model.
  FlaxStableDiffusionSafetyCheckerModule.concept_embeds: FlaxStableDiffusionSafetyCheckerModule#concept_embeds.
  FlaxStableDiffusionSafetyCheckerModule.special_care_embeds: FlaxStableDiffusionSafetyCheckerModule#special_care_embeds.
  FlaxStableDiffusionSafetyCheckerModule.config: FlaxStableDiffusionSafetyCheckerModule#config.
  FlaxStableDiffusionSafetyChecker.module_class: FlaxStableDiffusionSafetyChecker#module_class.
  jax_cosine_distance: jax_cosine_distance().
  FlaxStableDiffusionSafetyCheckerModule.dtype: FlaxStableDiffusionSafetyCheckerModule#dtype.
  FlaxStableDiffusionSafetyCheckerModule: FlaxStableDiffusionSafetyCheckerModule#
  FlaxStableDiffusionSafetyCheckerModule.concept_embeds_weights: FlaxStableDiffusionSafetyCheckerModule#concept_embeds_weights.
  FlaxStableDiffusionSafetyCheckerModule.special_care_embeds_weights: FlaxStableDiffusionSafetyCheckerModule#special_care_embeds_weights.
  FlaxStableDiffusionSafetyChecker: FlaxStableDiffusionSafetyChecker#
  FlaxStableDiffusionSafetyCheckerModule.setup: FlaxStableDiffusionSafetyCheckerModule#setup().
  FlaxStableDiffusionSafetyChecker.config_class: FlaxStableDiffusionSafetyChecker#config_class.
  FlaxStableDiffusionSafetyChecker.main_input_name: FlaxStableDiffusionSafetyChecker#main_input_name.
  FlaxStableDiffusionSafetyChecker.init_weights: FlaxStableDiffusionSafetyChecker#init_weights().
  FlaxStableDiffusionSafetyChecker.__call__: FlaxStableDiffusionSafetyChecker#__call__().
---
# Module: [`src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py)

## Classes
### `FlaxStableDiffusionSafetyChecker`  ·  implements/extends FlaxPreTrainedModel
- def: [`src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py:77`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L77)
- signature: `class FlaxStableDiffusionSafetyChecker(FlaxPreTrainedModel):`
- members:
  - `init_weights(self, rng: jax.Array, input_shape: Tuple, params: FrozenDict = None)` — [`L96`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L96)
  - `config_class` — [`L78`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L78)
  - `main_input_name` — [`L79`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L79)
  - `module_class` — [`L80`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L80)
- protocol/private: `__call__`[`L107`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L107), `__init__`[`L82`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L82)
- uses (calls/refs, reference-scoped): [`config`](safety_checker_flax.md#FlaxStableDiffusionSafetyCheckerModule.config), [`dtype`](safety_checker_flax.md#FlaxStableDiffusionSafetyCheckerModule.dtype), [`FlaxStableDiffusionSafetyCheckerModule`](safety_checker_flax.md#FlaxStableDiffusionSafetyCheckerModule)
- used by: [`_import_structure`](__init__.md#_import_structure)

### `FlaxStableDiffusionSafetyCheckerModule`  ·  implements/extends Module
- def: [`src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py:39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L39)
- signature: `class FlaxStableDiffusionSafetyCheckerModule(nn.Module):`
- members:
  - `setup(self)` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L43)
  - `concept_embeds` — [`L47`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L47)
  - `concept_embeds_weights` — [`L50`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L50)
  - `config` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L40)
  - `dtype` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L41)
  - `special_care_embeds` — [`L48`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L48)
  - `special_care_embeds_weights` — [`L51`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L51)
  - `vision_model` — [`L44`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L44)
  - `visual_projection` — [`L45`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L45)
- protocol/private: `__call__`[`L53`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L53)
- uses (calls/refs, reference-scoped): `projection_dim`, [`jax_cosine_distance`](safety_checker_flax.md#jax_cosine_distance), `vision_config`
- used by: [`__init__`](safety_checker_flax.md#FlaxStableDiffusionSafetyChecker.__init__), [`module_class`](safety_checker_flax.md#FlaxStableDiffusionSafetyChecker.module_class)

## Functions
- `jax_cosine_distance(emb_1, emb_2, eps=1e-12)` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/safety_checker_flax.py#L33)

