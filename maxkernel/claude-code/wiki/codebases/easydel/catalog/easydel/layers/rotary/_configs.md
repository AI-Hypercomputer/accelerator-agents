---
title: 'Module: easydel/layers/rotary/_configs.py'
type: catalog
provenance: extracted
module: easydel/layers/rotary/_configs.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.rotary._configs`/RopeConfig#
symbols:
  RopeConfig: ''
  RopeConfig.from_dict: from_dict().
  RopeConfig.to_dict: to_dict().
  RopeConfig.original_max_position_embeddings: original_max_position_embeddings.
  RopeConfig.update: update().
  RopeConfig.to_dict.rope_scaling: to_dict().rope_scaling#
  RopeConfig.rope_type: rope_type.
  RopeConfig.factor: factor.
  RopeConfig.low_freq_factor: low_freq_factor.
  RopeConfig.high_freq_factor: high_freq_factor.
  RopeConfig.long_factor: long_factor.
  RopeConfig.short_factor: short_factor.
  RopeConfig.long_mscale: long_mscale.
  RopeConfig.short_mscale: short_mscale.
  RopeConfig.extrapolation_factor: extrapolation_factor.
  RopeConfig.attn_factor: attn_factor.
  RopeConfig.beta_fast: beta_fast.
  RopeConfig.beta_slow: beta_slow.
  RopeConfig.mscale: mscale.
  RopeConfig.mscale_all_dim: mscale_all_dim.
  RopeConfig.mrope_interleaved: mrope_interleaved.
  RopeConfig.mrope_section: mrope_section.
  RopeConfig.repetition_style: repetition_style.
  RopeConfig.to_dict.rope_scaling.__hash__: to_dict().rope_scaling#__hash__.
---
# Module: [`easydel/layers/rotary/_configs.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py)

## Classes
### `RopeConfig`
- def: [`easydel/layers/rotary/_configs.py:48`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L48)
- doc: Configuration class for RoPE (Rotary Position Embedding) parameters.
- signature: `class RopeConfig:`
- members:
  - `from_dict(cls, config_dict: dict[str, tp.Any])` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L97) — Create a RopeConfig instance from a dictionary.
  - `to_dict(self)` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L198) — Convert the RopeConfig instance to a dictionary.
  - `update(self, config_dict: collections.abc.Mapping[str, tp.Any] | RopeConfig | None = None,, **kwargs: tp.Any)` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L139) — Update the RopeConfig instance in-place with new values.
  - `attn_factor` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L87)
  - `beta_fast` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L88)
  - `beta_slow` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L89)
  - `extrapolation_factor` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L86)
  - `factor` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L78)
  - `high_freq_factor` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L80)
  - `long_factor` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L82)
  - `long_mscale` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L84)
  - `low_freq_factor` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L79)
  - `mrope_interleaved` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L92)
  - `mrope_section` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L93)
  - `mscale` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L90)
  - `mscale_all_dim` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L91)
  - `original_max_position_embeddings` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L81)
  - `repetition_style` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L94)
  - `rope_type` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L77)
  - `short_factor` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L83)
  - `short_mscale` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L85)
- uses (calls/refs, reference-scoped): [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`rope_scaling`](_configs.md#RopeConfig.to_dict.rope_scaling)
- used by: [`get_basic_frequencies`](../../infra/base_config.md#EasyDeLBaseConfig.get_basic_frequencies), [`get_basic_rope`](../../infra/base_config.md#EasyDeLBaseConfig.get_basic_rope), [`_get_rope_config`](../../infra/base_config.md#EasyDeLBaseConfig._get_rope_config), [`get_basic_inv_frequencies`](../../infra/base_config.md#EasyDeLBaseConfig.get_basic_inv_frequencies), [`_get_rope_config`](../../modules/deepseek_v2/deepseek_configuration.md#DeepseekV2Config._get_rope_config), [`_get_rope_config`](../../modules/deepseek_v3/deepseek_configuration.md#DeepseekV3Config._get_rope_config), [`_get_rope_config`](../../modules/xerxes2/xerxes2_configuration.md#Xerxes2Config._get_rope_config), [`get_rope_config`](../../modules/kimi_linear/kimi_linear_configuration.md#KimiLinearConfig.get_rope_config)

### `rope_scaling`  ·  implements/extends dict
- def: [`easydel/layers/rotary/_configs.py:218`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L218)
- doc: A dictionary subclass that is hashable.
- signature: `class rope_scaling(dict):`
- protocol/private: `__hash__`[`L221`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_configs.py#L221)
- uses (calls/refs, reference-scoped): [`hash_fn`](../../utils/compiling_utils.md#hash_fn)
- used by: [`to_dict`](_configs.md#RopeConfig.to_dict)

