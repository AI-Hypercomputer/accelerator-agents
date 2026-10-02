---
title: 'Module: src/alphafold3/model/model.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/model.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.model`/
symbols:
  Model.__call__: Model#__call__().
  Model.get_inference_result: Model#get_inference_result().
  get_predicted_structure: get_predicted_structure().
  create_target_feat_embedding: create_target_feat_embedding().
  Model.diffusion_module: Model#diffusion_module.
  InferenceResult.metadata: InferenceResult#metadata.
  Model._sample_diffusion: Model#_sample_diffusion().
  Model.config: Model#config.
  Model.Config.heads: Model#Config#heads.
  Model.Config.evoformer: Model#Config#evoformer.
  Model.global_config: Model#global_config.
  InferenceResult: InferenceResult#
  _compute_ptm: _compute_ptm().
  Model.HeadsConfig.diffusion: Model#HeadsConfig#diffusion.
  InferenceResult.predicted_structure: InferenceResult#predicted_structure.
  Model.HeadsConfig.confidence: Model#HeadsConfig#confidence.
  Model.HeadsConfig.distogram: Model#HeadsConfig#distogram.
  ModelResult.ModelResult: ModelResult.ModelResult.
  _compute_chain_pair_iptm: _compute_chain_pair_iptm().
  Model.Config.global_config: Model#Config#global_config.
  InferenceResult.numerical_data: InferenceResult#numerical_data.
  InferenceResult.model_id: InferenceResult#model_id.
  Model.__init__: Model#__init__().
  Model.recycle_body: Model#recycle_body().
  InferenceResult.debug_outputs: InferenceResult#debug_outputs.
  Model: Model#
  Model.HeadsConfig: Model#HeadsConfig#
  Model.Config: Model#Config#
  Model.Config.num_recycles: Model#Config#num_recycles.
  Model.Config.return_embeddings: Model#Config#return_embeddings.
  Model.Config.return_distogram: Model#Config#return_distogram.
---
# Module: [`src/alphafold3/model/model.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/model.py:233`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L233)
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `evoformer` — [`L234`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L234)
  - `global_config` — [`L235`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L235) — documented in [alphafold3-model-model_config](../../../../concepts/alphafold3-model-model_config.md)
  - `heads` — [`L236`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L236)
  - `num_recycles` — [`L237`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L237)
  - `return_distogram` — [`L239`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L239)
  - `return_embeddings` — [`L238`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L238)
- uses (calls/refs, reference-scoped): [`GlobalConfig`](model_config.md#GlobalConfig), [`Evoformer`](network/evoformer.md#Evoformer), [`Config`](network/evoformer.md#Evoformer.Config), [`HeadsConfig`](model.md#Model.HeadsConfig), [`Model`](model.md#Model)
- used by: [`__call__`](model.md#Model.__call__), [`diffusion_module`](model.md#Model.diffusion_module), [`global_config`](model.md#Model.global_config), [`__init__`](model.md#Model.__init__)

### `HeadsConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/model.py:228`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L228)
- signature: `class HeadsConfig(base_config.BaseConfig):`
- members:
  - `confidence` — [`L230`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L230)
  - `diffusion` — [`L229`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L229)
  - `distogram` — [`L231`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L231)
- uses (calls/refs, reference-scoped): [`Config`](network/diffusion_head.md#DiffusionHead.Config), [`ConfidenceHead`](network/confidence_head.md#ConfidenceHead), [`Config`](network/confidence_head.md#ConfidenceHead.Config), [`Config`](network/distogram_head.md#DistogramHead.Config), [`DiffusionHead`](network/diffusion_head.md#DiffusionHead), [`DistogramHead`](network/distogram_head.md#DistogramHead)
- used by: [`__call__`](model.md#Model.__call__), [`diffusion_module`](model.md#Model.diffusion_module), [`heads`](model.md#Model.Config.heads)

### `InferenceResult`
- def: [`src/alphafold3/model/model.py:54`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L54) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
- doc: Postprocessed model result.
- signature: `class InferenceResult:`
- members:
  - `debug_outputs` — [`L75`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L75)
  - `metadata` — [`L72`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L72) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `model_id` — [`L76`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L76)
  - `numerical_data` — [`L69`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L69) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `predicted_structure` — [`L68`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L68) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
- uses (calls/refs, reference-scoped): [`Structure`](../structure/structure.md#Structure)
- used by: [`get_inference_result`](model.md#Model.get_inference_result), [`post_process_inference_result`](post_processing.md#post_process_inference_result), [`from_inference_result`](confidence_types.md#AtomConfidence.from_inference_result), [`from_inference_result`](confidence_types.md#StructureConfidenceFull.from_inference_result), [`write_output`](post_processing.md#write_output), [`from_inference_result`](confidence_types.md#StructureConfidenceSummary.from_inference_result)

### `Model`
- def: [`src/alphafold3/model/model.py:225`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L225)
- doc: Full model. Takes in data batch and returns model outputs.
- signature: `class Model(hk.Module):`
- members:
  - `get_inference_result(cls, batch: features.BatchDict, result: ModelResult, target_name: str = '')` — [`L356`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L356) — Get the predicted structure, scalars, and arrays for inference. — documented in [alphafold3-model](../../../../concepts/alphafold3-model.md)
  - `recycle_body(_, args)` — [`L289`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L289)
  - `config` — [`L243`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L243)
  - `diffusion_module` — [`L245`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L245)
  - `global_config` — [`L244`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L244)
- protocol/private: `__call__`[`L272`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L272), `__init__`[`L241`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L241), `_sample_diffusion`[`L250`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L250)
- uses (calls/refs, reference-scoped): [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`get_predicted_structure`](model.md#get_predicted_structure), [`BatchDict`](features.md#BatchDict.BatchDict), [`create_target_feat_embedding`](model.md#create_target_feat_embedding), [`Batch`](feat_batch.md#Batch), [`chains`](../structure/structure.md#Structure.chains), [`sample`](network/diffusion_head.md#sample), [`config`](network/evoformer.md#Evoformer.config), [`fraction_disordered`](confidences.md#fraction_disordered), [`has_clash`](confidences.md#has_clash), [`metadata`](model.md#InferenceResult.metadata), [`unstack`](../structure/structure.md#Structure.unstack), [`heads`](model.md#Model.Config.heads), [`get_ranking_score`](confidences.md#get_ranking_score), [`pae_metrics`](confidences.md#pae_metrics), [`sharded_map`](components/mapping.md#sharded_map), [`evoformer`](model.md#Model.Config.evoformer), [`InferenceResult`](model.md#InferenceResult), [`_compute_ptm`](model.md#_compute_ptm), [`diffusion`](model.md#Model.HeadsConfig.diffusion), [`pair_channel`](network/evoformer.md#Evoformer.Config.pair_channel), [`predicted_structure`](model.md#InferenceResult.predicted_structure), [`confidence`](model.md#Model.HeadsConfig.confidence), [`distogram`](model.md#Model.HeadsConfig.distogram), [`chain_pair_pae`](confidences.md#chain_pair_pae), [`Evoformer`](network/evoformer.md#Evoformer), [`ModelResult`](model.md#ModelResult.ModelResult), [`_compute_chain_pair_iptm`](model.md#_compute_chain_pair_iptm), [`eval`](network/diffusion_head.md#DiffusionHead.Config.eval), [`get_iptm_xchain`](confidences.md#get_iptm_xchain), [`global_config`](model.md#Model.Config.global_config), [`pde_single`](confidences.md#pde_single), [`ConfidenceHead`](network/confidence_head.md#ConfidenceHead), [`SampleConfig`](network/diffusion_head.md#SampleConfig), [`model_id`](model.md#InferenceResult.model_id), [`numerical_data`](model.md#InferenceResult.numerical_data), [`seq_channel`](network/evoformer.md#Evoformer.Config.seq_channel), [`DiffusionHead`](network/diffusion_head.md#DiffusionHead), [`DistogramHead`](network/distogram_head.md#DistogramHead), [`Config`](model.md#Model.Config)  (+6 more)
- used by: [`heads`](model.md#Model.Config.heads)

## Functions
- `_compute_chain_pair_iptm(num_tokens: int, asym_ids: np.ndarray, mask: np.ndarray, tm_adjusted_pae: np.ndarray)` — [`L205`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L205) — Computes the chain pair ipTM metrics from PAE.
- `_compute_ptm(result: ModelResult, num_tokens: int, asym_id: np.ndarray, pae_single_mask: np.ndarray, interface: bool)` — [`L183`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L183) — Computes the pTM metrics from PAE.
- `create_target_feat_embedding(batch: feat_batch.Batch, config: evoformer_network.Evoformer.Config, global_config: model_config.GlobalConfig)` — [`L152`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L152) — Create target feature embedding. — documented in [alphafold3-model](../../../../concepts/alphafold3-model.md)
- `get_predicted_structure(result: ModelResult, batch: feat_batch.Batch)` — [`L79`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L79) — Creates the predicted structure and ion preditions. — documented in [alphafold3-model](../../../../concepts/alphafold3-model.md)

## Module values
- `ModelResult` — [`L50`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model.py#L50)

