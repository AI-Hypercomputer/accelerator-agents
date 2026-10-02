---
title: 'Module: tpu_inference/layers/common/sharding.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/sharding.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.sharding`/
symbols:
  ShardingAxisName: ShardingAxisName.
  ShardingAxisNameBase: ShardingAxisNameBase#
  ShardingConfigManager.from_vllm_config: ShardingConfigManager#from_vllm_config().
  Sharding.make_sharding_config: Sharding#make_sharding_config().
  ShardingConfigManager.sharding_strategy: ShardingConfigManager#sharding_strategy.
  ShardingConfigManager.apply_vision_sharding: ShardingConfigManager#apply_vision_sharding().
  ShardingConfigManager.tp_size: ShardingConfigManager#tp_size().
  ShardingAxisNameBase.ATTN_HEAD: ShardingAxisNameBase#ATTN_HEAD.
  LazyShardingAxisName._initialize: LazyShardingAxisName#_initialize().
  ShardingConfigManager.model_dp_size: ShardingConfigManager#model_dp_size().
  ShardingConfigManager.total_dp_size: ShardingConfigManager#total_dp_size().
  ShardingConfigManager.__str__: ShardingConfigManager#__str__().
  ShardingConfigManager: ShardingConfigManager#
  ShardingConfigManager.attn_dp_size: ShardingConfigManager#attn_dp_size().
  ShardingConfigManager.attn_dp_expert_size: ShardingConfigManager#attn_dp_expert_size().
  ShardingConfigManager.expert_size: ShardingConfigManager#expert_size().
  ShardingConfigManager.decode_cp_size: ShardingConfigManager#decode_cp_size().
  LazyShardingAxisName.__getattr__: LazyShardingAxisName#__getattr__().
  ShardingAxisNameBase.ATTN_DATA: ShardingAxisNameBase#ATTN_DATA.
  ShardingAxisNameBase.MLP_DATA: ShardingAxisNameBase#MLP_DATA.
  Sharding._get_overrides: Sharding#_get_overrides().
  MESH_AXIS_NAMES_2D: MESH_AXIS_NAMES_2D.
  ShardingAxisNameBase.ATTN_DATA_EXPERT: ShardingAxisNameBase#ATTN_DATA_EXPERT.
  ShardingAxisNameBase.MLP_TENSOR: ShardingAxisNameBase#MLP_TENSOR.
  ShardingAxisNameBase.MOE_TENSOR: ShardingAxisNameBase#MOE_TENSOR.
  LazyShardingAxisName._cls: LazyShardingAxisName#_cls.
  ShardingConfigManager.sequence_size: ShardingConfigManager#sequence_size().
  Sharding.__str__: Sharding#__str__().
  Sharding.get_sharding_cfg: Sharding#get_sharding_cfg().
  ShardingConfigManager._total_devices: ShardingConfigManager#_total_devices.
  Sharding._apply_overrides: Sharding#_apply_overrides().
  MESH_AXIS_NAMES: MESH_AXIS_NAMES.
  LazyShardingAxisName._overrides: LazyShardingAxisName#_overrides.
  ShardingRulesConfig: ShardingRulesConfig#
  LazyShardingAxisName.override: LazyShardingAxisName#override().
  LazyShardingAxisName.reset: LazyShardingAxisName#reset().
  ShardingConfigManager.validate: ShardingConfigManager#validate().
  ShardingConfigManager.total_devices: ShardingConfigManager#total_devices().
  Sharding.sharding_cfg: Sharding#sharding_cfg.
  Sharding._make_default_sharding_config: Sharding#_make_default_sharding_config().
  ShardingStrategy: ShardingStrategy#
  ShardingStrategy.data_parallelism: ShardingStrategy#data_parallelism.
  ShardingStrategy.attention_data_parallelism: ShardingStrategy#attention_data_parallelism.
  ShardingStrategy.attention_data_expert_parallelism: ShardingStrategy#attention_data_expert_parallelism.
  ShardingConfig: ShardingConfig#
  logger: logger.
  ShardingConfigManager.__init__: ShardingConfigManager#__init__().
  ShardingConfig.__init__: ShardingConfig#__init__().
  Sharding.__init__: Sharding#__init__().
  ShardingAxisName2D: ShardingAxisName2D#
  ShardingStrategy.tensor_parallelism: ShardingStrategy#tensor_parallelism.
  ShardingStrategy.expert_parallelism: ShardingStrategy#expert_parallelism.
  ShardingStrategy.sequence_parallelism: ShardingStrategy#sequence_parallelism.
  ShardingStrategy.decode_context_parallelism: ShardingStrategy#decode_context_parallelism.
  ShardingConfigManager.mm_encoder_tp_mode: ShardingConfigManager#mm_encoder_tp_mode.
  Sharding.vllm_config: Sharding#vllm_config.
  ShardingAxisNameBase.BATCH: ShardingAxisNameBase#BATCH.
  LazyShardingAxisName: LazyShardingAxisName#
  ShardingConfigManager.device_indexes: ShardingConfigManager#device_indexes.
  ShardingConfig.prefill_rules: ShardingConfig#prefill_rules.
  ShardingConfig.generate_rules: ShardingConfig#generate_rules.
  Sharding.default_rules_cls: Sharding#default_rules_cls.
  ShardingAxisNameBase.SEQUENCE: ShardingAxisNameBase#SEQUENCE.
  ShardingAxisNameBase.ATTN_TENSOR: ShardingAxisNameBase#ATTN_TENSOR.
  ShardingAxisNameBase.EXPERT: ShardingAxisNameBase#EXPERT.
  ShardingAxisNameBase.EXPERT_DATA: ShardingAxisNameBase#EXPERT_DATA.
  ShardingAxisNameBase.VOCAB: ShardingAxisNameBase#VOCAB.
  ShardingAxisNameBase.MODEL_1: ShardingAxisNameBase#MODEL_1.
  ShardingAxisNameBase.MODEL_2: ShardingAxisNameBase#MODEL_2.
  ShardingAxisNameBase.MODEL: ShardingAxisNameBase#MODEL.
  ShardingAxisNameBase.CONTEXT: ShardingAxisNameBase#CONTEXT.
  ShardingAxisNameBase.KV_CACHE_HEAD: ShardingAxisNameBase#KV_CACHE_HEAD.
  ShardingAxisNameBase.VIT_BATCH: ShardingAxisNameBase#VIT_BATCH.
  ShardingAxisNameBase.VIT_MODEL: ShardingAxisNameBase#VIT_MODEL.
  ShardingAxisName2D.SEQUENCE: ShardingAxisName2D#SEQUENCE.
  ShardingAxisName2D.ATTN_DATA: ShardingAxisName2D#ATTN_DATA.
  ShardingAxisName2D.MLP_DATA: ShardingAxisName2D#MLP_DATA.
  ShardingAxisName2D.ATTN_HEAD: ShardingAxisName2D#ATTN_HEAD.
  ShardingAxisName2D.ATTN_TENSOR: ShardingAxisName2D#ATTN_TENSOR.
  ShardingAxisName2D.MLP_TENSOR: ShardingAxisName2D#MLP_TENSOR.
  ShardingAxisName2D.MOE_TENSOR: ShardingAxisName2D#MOE_TENSOR.
  ShardingAxisName2D.EXPERT: ShardingAxisName2D#EXPERT.
  ShardingAxisName2D.EXPERT_DATA: ShardingAxisName2D#EXPERT_DATA.
  ShardingAxisName2D.VOCAB: ShardingAxisName2D#VOCAB.
  ShardingAxisName2D.BATCH: ShardingAxisName2D#BATCH.
  ShardingAxisName2D.CONTEXT: ShardingAxisName2D#CONTEXT.
  ShardingAxisName2D.KV_CACHE_HEAD: ShardingAxisName2D#KV_CACHE_HEAD.
  ShardingAxisName2D.MODEL: ShardingAxisName2D#MODEL.
  ShardingAxisName2D.VIT_BATCH: ShardingAxisName2D#VIT_BATCH.
  ShardingAxisName2D.VIT_MODEL: ShardingAxisName2D#VIT_MODEL.
  LazyShardingAxisName.__init__: LazyShardingAxisName#__init__().
  ShardingRulesConfig.activation_attention_td: ShardingRulesConfig#activation_attention_td.
  ShardingRulesConfig.activation_attention_out_td: ShardingRulesConfig#activation_attention_out_td.
  ShardingRulesConfig.activation_q_td: ShardingRulesConfig#activation_q_td.
  ShardingRulesConfig.attn_o_tnh: ShardingRulesConfig#attn_o_tnh.
  ShardingRulesConfig.query_tnh: ShardingRulesConfig#query_tnh.
  ShardingRulesConfig.keyvalue_skh: ShardingRulesConfig#keyvalue_skh.
  ShardingRulesConfig.attn_q_weight_dnh: ShardingRulesConfig#attn_q_weight_dnh.
  ShardingRulesConfig.attn_k_weight_dkh: ShardingRulesConfig#attn_k_weight_dkh.
  ShardingRulesConfig.attn_v_weight_dkh: ShardingRulesConfig#attn_v_weight_dkh.
  ShardingRulesConfig.attn_o_weight_nhd: ShardingRulesConfig#attn_o_weight_nhd.
  ShardingRulesConfig.activation_ffw_td: ShardingRulesConfig#activation_ffw_td.
  ShardingRulesConfig.activation_ffw_ted: ShardingRulesConfig#activation_ffw_ted.
  ShardingRulesConfig.ffw_hidden_tf: ShardingRulesConfig#ffw_hidden_tf.
  ShardingRulesConfig.ffw_weight_df: ShardingRulesConfig#ffw_weight_df.
  ShardingRulesConfig.ffw_weight_fd: ShardingRulesConfig#ffw_weight_fd.
  ShardingRulesConfig.moe_weights_edf: ShardingRulesConfig#moe_weights_edf.
  ShardingRulesConfig.moe_weights_efd: ShardingRulesConfig#moe_weights_efd.
  ShardingRulesConfig.moe_router_de: ShardingRulesConfig#moe_router_de.
  ShardingRulesConfig.moe_router_bias_e: ShardingRulesConfig#moe_router_bias_e.
  ShardingRulesConfig.emb_weight_vd: ShardingRulesConfig#emb_weight_vd.
  ShardingRulesConfig.activation_td: ShardingRulesConfig#activation_td.
  ShardingRulesConfig.prelogit_td: ShardingRulesConfig#prelogit_td.
  ShardingRulesConfig.logits_tv: ShardingRulesConfig#logits_tv.
  ShardingRulesConfig.norm_scale: ShardingRulesConfig#norm_scale.
  ShardingRulesConfig.vocab_vd: ShardingRulesConfig#vocab_vd.
  ShardingRulesConfig.vocab_dv: ShardingRulesConfig#vocab_dv.
  ShardingConfig.default_rules_cls: ShardingConfig#default_rules_cls.
  build_mesh: build_mesh().
  Sharding: Sharding#
  Sharding.validate_sharding_strategy: Sharding#validate_sharding_strategy().
  ShardingInfo: ShardingInfo#
---
# Module: [`tpu_inference/layers/common/sharding.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py)

## Classes
### `LazyShardingAxisName`
- def: [`tpu_inference/layers/common/sharding.py:99`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L99)
- doc: Lazy loading for ShardingAxisName.
- signature: `class LazyShardingAxisName:`
- members:
  - `override(self, **kwargs)` — [`L118`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L118) — Set runtime axis overrides.
  - `reset(self)` — [`L122`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L122) — Clear all overrides, restoring base-class values.
- protocol/private: `__getattr__`[`L126`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L126), `__init__`[`L102`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L102), `_cls`[`L103`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L103), `_initialize`[`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L106), `_overrides`[`L104`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L104)
- uses (calls/refs, reference-scoped): [`ShardingAxisNameBase`](sharding.md#ShardingAxisNameBase), [`NEW_MODEL_DESIGN`](../../envs.md#NEW_MODEL_DESIGN), [`ShardingAxisName2D`](sharding.md#ShardingAxisName2D)
- used by: [`ShardingAxisName`](sharding.md#ShardingAxisName), [`apply_vision_sharding`](sharding.md#ShardingConfigManager.apply_vision_sharding)

### `Sharding`
- def: [`tpu_inference/layers/common/sharding.py:532`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L532)
- doc: Generates and manages sharding configurations based on a high-level strategy.
- signature: `class Sharding:`
- members:
  - `__init__(self, prefill_rules: dict | None = None, generate_rules: dict | None = None, default_rules_cls=ShardingRulesConfig, vllm_config: VllmConfig = None)` — [`L543`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L543) — Initializes the Sharding manager.
  - `__str__(self)` — [`L581`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L581) — Succinct representation of relevant Sharding settings and overrides.
  - `_apply_overrides(self, config_obj: ShardingRulesConfig, overrides: dict | None)` — [`L603`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L603) — Applies runtime overrides to a sharding configuration object.
  - `_get_overrides(self, sharding_phase: str)` — [`L564`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L564) — Return the overrides from the vLLM config for the given sharding phase.
  - `get_sharding_cfg(self)` — [`L599`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L599) — Returns the generated sharding configuration.
  - `make_sharding_config(self, default_rules_cls: ShardingRulesConfig, prefill_overrides: dict | None = None, generate_overrides: dict | None = None)` — [`L695`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L695) — Creates the detailed `ShardingConfig` with specific partitioning rules
  - `validate_sharding_strategy(self)` — [`L589`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L589) — Validates if the sharding strategy is compatible with the environment.
  - `default_rules_cls` — [`L558`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L558) — ---
  - `sharding_cfg` — [`L559`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L559)
  - `vllm_config` — [`L557`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L557)
- protocol/private: `_make_default_sharding_config`[`L626`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L626)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](sharding.md#ShardingAxisName), [`ShardingRulesConfig`](sharding.md#ShardingRulesConfig), [`ShardingConfig`](sharding.md#ShardingConfig), [`generate_rules`](sharding.md#ShardingConfig.generate_rules), [`prefill_rules`](sharding.md#ShardingConfig.prefill_rules)

### `ShardingAxisName2D`
- def: [`tpu_inference/layers/common/sharding.py:71`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L71)
- doc: Sharding axis names for 2D data parallelism scenarios.
- signature: `class ShardingAxisName2D:`
- members:
  - `ATTN_DATA` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L78)
  - `ATTN_HEAD` — [`L80`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L80)
  - `ATTN_TENSOR` — [`L81`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L81)
  - `BATCH` — [`L87`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L87)
  - `CONTEXT` — [`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L88)
  - `EXPERT` — [`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L84)
  - `EXPERT_DATA` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L85)
  - `KV_CACHE_HEAD` — [`L89`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L89)
  - `MLP_DATA` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L79)
  - `MLP_TENSOR` — [`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L82)
  - `MODEL` — [`L90`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L90)
  - `MOE_TENSOR` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L83)
  - `SEQUENCE` — [`L77`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L77)
  - `VIT_BATCH` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L93)
  - `VIT_MODEL` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L94)
  - `VOCAB` — [`L86`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L86)
- used by: [`_initialize`](sharding.md#LazyShardingAxisName._initialize)

### `ShardingAxisNameBase`
- def: [`tpu_inference/layers/common/sharding.py:37`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L37)
- doc: Base class for sharding axis names.
- signature: `class ShardingAxisNameBase:`
- members:
  - `ATTN_DATA` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L44)
  - `ATTN_DATA_EXPERT` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L45)
  - `ATTN_HEAD` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L47)
  - `ATTN_TENSOR` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L48)
  - `BATCH` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L62)
  - `CONTEXT` — [`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L63)
  - `EXPERT` — [`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L51)
  - `EXPERT_DATA` — [`L52`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L52)
  - `KV_CACHE_HEAD` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L64)
  - `MLP_DATA` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L46)
  - `MLP_TENSOR` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L49)
  - `MODEL` — [`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L59)
  - `MODEL_1` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L55)
  - `MODEL_2` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L56)
  - `MOE_TENSOR` — [`L50`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L50)
  - `SEQUENCE` — [`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L39)
  - `VIT_BATCH` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L67)
  - `VIT_MODEL` — [`L68`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L68)
  - `VOCAB` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L54)
- used by: [`get_decoder_layer`](../../models/jax/deepseek_v3.md#DeepSeekV3.get_decoder_layer), [`shared_experts`](../../models/jax/deepseek_v3.md#DeepseekV2Moe.shared_experts), [`compute_attention`](../../models/jax/deepseek_v3.md#DeepseekV3Attention.compute_attention), [`_create_deepseek_attention`](../../models/jax/deepseek_v3.md#DeepSeekV3._create_deepseek_attention), [`gate`](../../models/jax/deepseek_v3.md#DeepseekV2Moe.gate), [`embed_tokens`](../../models/jax/deepseek_v3.md#DeepSeekV3.embed_tokens), [`lm_head`](../../models/jax/deepseek_v3.md#DeepseekV3ForCausalLM.lm_head), [`_initialize`](sharding.md#LazyShardingAxisName._initialize), [`expert_axis_name`](../../models/jax/deepseek_v3.md#expert_axis_name)

### `ShardingConfig`
- def: [`tpu_inference/layers/common/sharding.py:443`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L443)
- doc: Container for operation-specific sharding configurations.
- signature: `class ShardingConfig:`
- members:
  - `__init__(self, prefill_rules=None, generate_rules=None, default_rules_cls=ShardingRulesConfig)` — [`L473`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L473) — Initializes the ShardingConfig.
  - `default_rules_cls` — [`L487`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L487) — ---
  - `generate_rules` — [`L490`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L490)
  - `prefill_rules` — [`L488`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L488)
- uses (calls/refs, reference-scoped): [`ShardingRulesConfig`](sharding.md#ShardingRulesConfig)
- used by: [`make_sharding_config`](sharding.md#Sharding.make_sharding_config), [`get_sharding_cfg`](sharding.md#Sharding.get_sharding_cfg)

### `ShardingConfigManager`
- def: [`tpu_inference/layers/common/sharding.py:161`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L161)
- doc: Manages sharding configuration parsing and access from vLLM config.
- signature: `class ShardingConfigManager:`
- members:
  - `apply_vision_sharding(self)` — [`L353`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L353)
  - `attn_dp_expert_size(self)` — [`L330`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L330)
  - `attn_dp_size(self)` — [`L326`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L326)
  - `decode_cp_size(self)` — [`L346`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L346)
  - `expert_size(self)` — [`L338`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L338)
  - `from_vllm_config(cls, vllm_config: VllmConfig)` — [`L187`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L187)
  - `model_dp_size(self)` — [`L322`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L322)
  - `sequence_size(self)` — [`L342`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L342)
  - `total_devices(self)` — [`L350`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L350)
  - `total_dp_size(self)` — [`L318`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L318)
  - `tp_size(self)` — [`L334`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L334)
  - `validate(cls, vllm_config, sharding_strategy)` — [`L298`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L298)
  - `device_indexes` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L179)
  - `mm_encoder_tp_mode` — [`L184`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L184)
  - `sharding_strategy` — [`L178`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L178)
- protocol/private: `__init__`[`L173`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L173), `__str__`[`L362`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L362), `_total_devices`[`L180`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L180)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](sharding.md#ShardingAxisName), [`NEW_MODEL_DESIGN`](../../envs.md#NEW_MODEL_DESIGN), [`TPU_MULTIPROCESS_DP`](../../envs.md#TPU_MULTIPROCESS_DP), [`override`](sharding.md#LazyShardingAxisName.override), [`reset`](sharding.md#LazyShardingAxisName.reset), [`ShardingStrategy`](sharding.md#ShardingStrategy), [`attention_data_expert_parallelism`](sharding.md#ShardingStrategy.attention_data_expert_parallelism), [`attention_data_parallelism`](sharding.md#ShardingStrategy.attention_data_parallelism), [`data_parallelism`](sharding.md#ShardingStrategy.data_parallelism), [`decode_context_parallelism`](sharding.md#ShardingStrategy.decode_context_parallelism), [`expert_parallelism`](sharding.md#ShardingStrategy.expert_parallelism), [`sequence_parallelism`](sharding.md#ShardingStrategy.sequence_parallelism), [`tensor_parallelism`](sharding.md#ShardingStrategy.tensor_parallelism)
- used by: `_create_multi_slice_mesh`, `_create_single_slice_mesh`, `_create_2d_mesh`

### `ShardingInfo`
- def: [`tpu_inference/layers/common/sharding.py:741`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L741)
- signature: `class ShardingInfo:`

### `ShardingRulesConfig`
- def: [`tpu_inference/layers/common/sharding.py:371`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L371)
- doc: Holds detailed sharding configurations for individual tensors, namely logical rules.
- signature: `class ShardingRulesConfig:`
- members:
  - `activation_attention_out_td` — [`L387`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L387)
  - `activation_attention_td` — [`L385`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L385)
  - `activation_ffw_td` — [`L407`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L407)
  - `activation_ffw_ted` — [`L410`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L410)
  - `activation_q_td` — [`L389`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L389)
  - `activation_td` — [`L431`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L431)
  - `attn_k_weight_dkh` — [`L400`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L400)
  - `attn_o_tnh` — [`L391`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L391)
  - `attn_o_weight_nhd` — [`L404`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L404)
  - `attn_q_weight_dnh` — [`L398`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L398)
  - `attn_v_weight_dkh` — [`L402`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L402)
  - `emb_weight_vd` — [`L429`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L429)
  - `ffw_hidden_tf` — [`L413`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L413)
  - `ffw_weight_df` — [`L416`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L416)
  - `ffw_weight_fd` — [`L418`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L418)
  - `keyvalue_skh` — [`L395`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L395)
  - `logits_tv` — [`L435`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L435)
  - `moe_router_bias_e` — [`L426`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L426)
  - `moe_router_de` — [`L424`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L424)
  - `moe_weights_edf` — [`L420`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L420)
  - `moe_weights_efd` — [`L422`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L422)
  - `norm_scale` — [`L437`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L437)
  - `prelogit_td` — [`L433`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L433)
  - `query_tnh` — [`L393`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L393)
  - `vocab_dv` — [`L440`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L440)
  - `vocab_vd` — [`L439`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L439)
- used by: [`make_sharding_config`](sharding.md#Sharding.make_sharding_config), [`_apply_overrides`](sharding.md#Sharding._apply_overrides), [`__init__`](sharding.md#Sharding.__init__), [`__init__`](sharding.md#ShardingConfig.__init__)

### `ShardingStrategy`
- def: [`tpu_inference/layers/common/sharding.py:137`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L137)
- doc: Defines the high-level parallelism strategy.
- signature: `class ShardingStrategy:`
- members:
  - `attention_data_expert_parallelism` — [`L157`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L157)
  - `attention_data_parallelism` — [`L156`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L156)
  - `data_parallelism` — [`L155`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L155)
  - `decode_context_parallelism` — [`L158`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L158)
  - `expert_parallelism` — [`L153`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L153)
  - `sequence_parallelism` — [`L154`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L154)
  - `tensor_parallelism` — [`L152`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L152)
- used by: [`from_vllm_config`](sharding.md#ShardingConfigManager.from_vllm_config), [`sharding_strategy`](sharding.md#ShardingConfigManager.sharding_strategy), [`tp_size`](sharding.md#ShardingConfigManager.tp_size), [`model_dp_size`](sharding.md#ShardingConfigManager.model_dp_size), [`total_dp_size`](sharding.md#ShardingConfigManager.total_dp_size), [`attn_dp_expert_size`](sharding.md#ShardingConfigManager.attn_dp_expert_size), [`attn_dp_size`](sharding.md#ShardingConfigManager.attn_dp_size), [`decode_cp_size`](sharding.md#ShardingConfigManager.decode_cp_size), [`expert_size`](sharding.md#ShardingConfigManager.expert_size), [`sequence_size`](sharding.md#ShardingConfigManager.sequence_size), [`__init__`](sharding.md#ShardingConfigManager.__init__)

## Functions
- `build_mesh(devices, strategy: dict[str, int])` — [`L494`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L494) — Constructs a JAX device mesh from a sharding strategy.

## Module values
- `MESH_AXIS_NAMES` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L32)
- `MESH_AXIS_NAMES_2D` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L34)
- `ShardingAxisName` — [`L133`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L133) — documented in [root](../../../../concepts/root.md)
- `logger` — [`L30`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/sharding.py#L30)

