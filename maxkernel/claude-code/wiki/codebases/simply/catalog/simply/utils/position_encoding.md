---
title: 'Module: simply/utils/position_encoding.py'
type: catalog
provenance: extracted
module: simply/utils/position_encoding.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.position_encoding`/
symbols:
  RoPE: RoPE#
  RoPE.apply: RoPE#apply().
  PositionEncodingRegistry.namespace: PositionEncodingRegistry#namespace.
  PositionEncodingConfig: PositionEncodingConfig#
  PositionEncodingRegistry: PositionEncodingRegistry#
  PositionEncodingConfig.apply: PositionEncodingConfig#apply().
  RoPE.min_timescale: RoPE#min_timescale.
  RoPE.max_timescale: RoPE#max_timescale.
  RoPE.scale_factor: RoPE#scale_factor.
---
# Module: [`simply/utils/position_encoding.py`](../../../../../../raw/code/simply/simply/utils/position_encoding.py)

## Classes
### `PositionEncodingConfig`
- def: [`simply/utils/position_encoding.py:32`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L32)
- doc: Base class for position encoding configurations.
- signature: `class PositionEncodingConfig:`
- members:
  - `apply(self, embedding_mat: common.Array, segment_positions: common.Array | None = None)` — [`L35`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L35) — Apply position encoding to embedding matrix.
- uses (calls/refs, reference-scoped): [`Array`](common.md#Array.Array)
- used by: [`RoPE`](position_encoding.md#RoPE), [`position_encoding`](../model_lib.md#Attention.position_encoding), [`position_encoding`](../model_lib.md#TransformerBlock.position_encoding), [`position_encoding`](../config_lib.md#BaseExperimentConfig.position_encoding)

### `PositionEncodingRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/position_encoding.py:25`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L25)
- doc: Registry for position encoding configurations.
- signature: `class PositionEncodingRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L27`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L27)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`RoPE`](position_encoding.md#RoPE)

### `RoPE`
- def: [`simply/utils/position_encoding.py:55`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L55)
- doc: Standard Rotary Position Embedding configuration.
- signature: `class RoPE(PositionEncodingConfig):`
- members:
  - `apply(self, embedding_mat: common.Array, segment_positions: common.Array | None = None)` — [`L68`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L68) — Apply rotary positional embedding.
  - `max_timescale` — [`L65`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L65)
  - `min_timescale` — [`L64`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L64)
  - `scale_factor` — [`L66`](../../../../../../raw/code/simply/simply/utils/position_encoding.py#L66)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Array`](common.md#Array.Array), [`PositionEncodingConfig`](position_encoding.md#PositionEncodingConfig), [`PositionEncodingRegistry`](position_encoding.md#PositionEncodingRegistry)
- used by: [`qwen3_0p6b`](../config_lib.md#qwen3_0p6b), [`deepseek_qwen2_32b`](../config_lib.md#deepseek_qwen2_32b), [`gemma3_1b`](../config_lib.md#gemma3_1b), [`deepseek_qwen2_14b`](../config_lib.md#deepseek_qwen2_14b), [`qwen3_235b_a22b_thinking_2507`](../config_lib.md#qwen3_235b_a22b_thinking_2507), [`gemma3_12b`](../config_lib.md#gemma3_12b), [`gemma3_4b`](../config_lib.md#gemma3_4b), [`gemma3_27b`](../config_lib.md#gemma3_27b), [`qwen3_30b_a3b_thinking_2507`](../config_lib.md#qwen3_30b_a3b_thinking_2507), [`qwen3_4b_thinking_2507`](../config_lib.md#qwen3_4b_thinking_2507), [`position_encoding`](../model_lib.md#Attention.position_encoding), [`position_encoding`](../model_lib.md#TransformerBlock.position_encoding), [`position_encoding`](../config_lib.md#BaseExperimentConfig.position_encoding), [`test_custom_segment_positions`](position_encoding_test.md#RoPETest.test_custom_segment_positions), [`test_different_max_timescales`](position_encoding_test.md#RoPETest.test_different_max_timescales), [`test_different_scale_factors`](position_encoding_test.md#RoPETest.test_different_scale_factors), [`test_position_dependent_output`](position_encoding_test.md#RoPETest.test_position_dependent_output), [`test_preserves_dtype`](position_encoding_test.md#RoPETest.test_preserves_dtype), [`test_preserves_shape`](position_encoding_test.md#RoPETest.test_preserves_shape)

