---
title: 'Module: src/alphafold3/model/network/diffusion_head.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/diffusion_head.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.diffusion_head`/
symbols:
  DiffusionHead.__call__: DiffusionHead#__call__().
  DiffusionHead._conditioning: DiffusionHead#_conditioning().
  sample: sample().
  sample.apply_denoising_step: sample().apply_denoising_step().
  SIGMA_DATA: SIGMA_DATA.
  DiffusionHead.Config: DiffusionHead#Config#
  random_augmentation: random_augmentation().
  DiffusionHead.Config.transformer: DiffusionHead#Config#transformer.
  DiffusionHead.config: DiffusionHead#config.
  DiffusionHead.global_config: DiffusionHead#global_config.
  DiffusionHead.Config.conditioning: DiffusionHead#Config#conditioning.
  DiffusionHead.__init__: DiffusionHead#__init__().
  noise_schedule: noise_schedule().
  DiffusionHead.Config.eval: DiffusionHead#Config#eval.
  SampleConfig: SampleConfig#
  ConditioningConfig.seq_channel: ConditioningConfig#seq_channel.
  DiffusionHead: DiffusionHead#
  random_rotation: random_rotation().
  ConditioningConfig: ConditioningConfig#
  ConditioningConfig.pair_channel: ConditioningConfig#pair_channel.
  SampleConfig.steps: SampleConfig#steps.
  SampleConfig.gamma_0: SampleConfig#gamma_0.
  SampleConfig.gamma_min: SampleConfig#gamma_min.
  SampleConfig.noise_scale: SampleConfig#noise_scale.
  SampleConfig.step_scale: SampleConfig#step_scale.
  SampleConfig.num_samples: SampleConfig#num_samples.
  ConditioningConfig.prob: ConditioningConfig#prob.
  DiffusionHead.Config.eval_batch_size: DiffusionHead#Config#eval_batch_size.
  DiffusionHead.Config.eval_batch_dim_shard_size: DiffusionHead#Config#eval_batch_dim_shard_size.
---
# Module: [`src/alphafold3/model/network/diffusion_head.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py)

## Classes
### `ConditioningConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/diffusion_head.py:95`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L95)
- signature: `class ConditioningConfig(base_config.BaseConfig):`
- members:
  - `pair_channel` — [`L96`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L96)
  - `prob` — [`L98`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L98)
  - `seq_channel` — [`L97`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L97)
- used by: [`_conditioning`](diffusion_head.md#DiffusionHead._conditioning), [`conditioning`](diffusion_head.md#DiffusionHead.Config.conditioning)

### `Config`  ·  implements/extends AtomCrossAttDecoderConfig, AtomCrossAttEncoderConfig
- def: [`src/alphafold3/model/network/diffusion_head.py:113`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L113)
- doc: Configuration for DiffusionHead.
- signature: `class Config(atom_cross_attention.AtomCrossAttEncoderConfig, atom_cross_attention.AtomCrossAttDecoderConfig):`
- members:
  - `conditioning` — [`L121`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L121) — documented in [alphafold3-model-network-diffusion_head](../../../../../concepts/alphafold3-model-network-diffusion_head.md)
  - `eval` — [`L124`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L124)
  - `eval_batch_dim_shard_size` — [`L120`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L120)
  - `eval_batch_size` — [`L119`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L119)
  - `transformer` — [`L128`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L128) — documented in [alphafold3-model-network-diffusion_head](../../../../../concepts/alphafold3-model-network-diffusion_head.md)
- uses (calls/refs, reference-scoped): [`AtomCrossAttDecoderConfig`](atom_cross_attention.md#AtomCrossAttDecoderConfig), [`AtomCrossAttEncoderConfig`](atom_cross_attention.md#AtomCrossAttEncoderConfig), [`SampleConfig`](diffusion_head.md#SampleConfig), [`Config`](diffusion_transformer.md#Transformer.Config), [`Transformer`](diffusion_transformer.md#Transformer), [`ConditioningConfig`](diffusion_head.md#ConditioningConfig)
- used by: [`__call__`](../model.md#Model.__call__), [`__call__`](diffusion_head.md#DiffusionHead.__call__), [`_conditioning`](diffusion_head.md#DiffusionHead._conditioning), [`AtomCrossAttDecoderConfig`](atom_cross_attention.md#AtomCrossAttDecoderConfig), [`diffusion`](../model.md#Model.HeadsConfig.diffusion), [`AtomCrossAttEncoderConfig`](atom_cross_attention.md#AtomCrossAttEncoderConfig), [`__init__`](diffusion_head.md#DiffusionHead.__init__)

### `DiffusionHead`
- def: [`src/alphafold3/model/network/diffusion_head.py:110`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L110)
- doc: Denoising Diffusion Head.
- signature: `class DiffusionHead(hk.Module):`
- members:
  - `config` — [`L138`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L138)
  - `global_config` — [`L139`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L139)
- protocol/private: `__call__`[`L211`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L211), `__init__`[`L132`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L132), `_conditioning`[`L143`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L143)
- uses (calls/refs, reference-scoped): [`atom_cross_att_encoder`](atom_cross_attention.md#atom_cross_att_encoder), [`Linear`](../components/haiku_modules.md#Linear), [`atom_cross_att_decoder`](atom_cross_attention.md#atom_cross_att_decoder), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`Batch`](../feat_batch.md#Batch), [`final_init`](../model_config.md#GlobalConfig.final_init), [`create_relative_encoding`](featurization.md#create_relative_encoding), [`transition_block`](diffusion_transformer.md#transition_block), [`token_features`](../feat_batch.md#Batch.token_features), [`mask`](../features.md#TokenFeatures.mask), [`predicted_structure_info`](../feat_batch.md#Batch.predicted_structure_info), [`Config`](diffusion_head.md#DiffusionHead.Config), [`SIGMA_DATA`](diffusion_head.md#SIGMA_DATA), [`atom_mask`](../features.md#PredictedStructureInfo.atom_mask), [`bfloat16_context`](../components/utils.md#bfloat16_context), [`AtomCrossAttEncoderOutput`](atom_cross_attention.md#AtomCrossAttEncoderOutput), [`noise_embeddings`](noise_level_embeddings.md#noise_embeddings), [`transformer`](diffusion_head.md#DiffusionHead.Config.transformer), [`conditioning`](diffusion_head.md#DiffusionHead.Config.conditioning), [`token_act`](atom_cross_attention.md#AtomCrossAttEncoderOutput.token_act), [`Transformer`](diffusion_transformer.md#Transformer), [`seq_channel`](diffusion_head.md#ConditioningConfig.seq_channel), [`pair_channel`](diffusion_head.md#ConditioningConfig.pair_channel)
- used by: [`diffusion_module`](../model.md#Model.diffusion_module), [`diffusion`](../model.md#Model.HeadsConfig.diffusion)

### `SampleConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/diffusion_head.py:101`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L101)
- signature: `class SampleConfig(base_config.BaseConfig):`
- members:
  - `gamma_0` — [`L103`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L103)
  - `gamma_min` — [`L104`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L104)
  - `noise_scale` — [`L105`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L105)
  - `num_samples` — [`L107`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L107)
  - `step_scale` — [`L106`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L106)
  - `steps` — [`L102`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L102)
- used by: [`sample`](diffusion_head.md#sample), [`apply_denoising_step`](diffusion_head.md#sample.apply_denoising_step), [`_sample_diffusion`](../model.md#Model._sample_diffusion), [`eval`](diffusion_head.md#DiffusionHead.Config.eval)

## Functions
- `apply_denoising_step(carry, noise_level)` — [`L333`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L333)
- `noise_schedule(t, smin=0.0004, smax=160, p=7)` — [`L88`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L88)
- `random_augmentation(rng_key: jnp.ndarray, positions: jnp.ndarray, mask: jnp.ndarray)` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L53) — Apply random rigid augmentation.
- `random_rotation(key)` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L42)
- `sample(denoising_step: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], batch: feat_batch.Batch, key: jnp.ndarray, config: SampleConfig)` — [`L306`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L306) — Sample using denoiser on batch. — documented in [alphafold3-model-network-diffusion_head](../../../../../concepts/alphafold3-model-network-diffusion_head.md)

## Module values
- `SIGMA_DATA` — [`L39`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_head.py#L39) — documented in [alphafold3-model-network-diffusion_head](../../../../../concepts/alphafold3-model-network-diffusion_head.md)

