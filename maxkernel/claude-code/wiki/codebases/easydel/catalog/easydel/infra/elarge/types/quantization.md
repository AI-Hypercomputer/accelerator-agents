---
title: 'Module: easydel/infra/elarge/types/quantization.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/quantization.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.quantization`/
symbols:
  QuantizationCfg.kv_cache: QuantizationCfg#kv_cache.
  QuantizationCfg.model: QuantizationCfg#model.
  EasyDeLQuantizationCfg: EasyDeLQuantizationCfg#
  QuantizationCfg: QuantizationCfg#
  QuantizationCfg.platform: QuantizationCfg#platform.
  EasyDeLQuantizationCfg.dtype: EasyDeLQuantizationCfg#dtype.
  EasyDeLQuantizationCfg.runtime_dtype: EasyDeLQuantizationCfg#runtime_dtype.
  EasyDeLQuantizationCfg.group_size: EasyDeLQuantizationCfg#group_size.
  EasyDeLQuantizationCfg.bits: EasyDeLQuantizationCfg#bits.
  EasyDeLQuantizationCfg.simulate: EasyDeLQuantizationCfg#simulate.
  EasyDeLQuantizationCfg.jax_native: EasyDeLQuantizationCfg#jax_native.
  EasyDeLQuantizationCfg.pattern: EasyDeLQuantizationCfg#pattern.
  QuantizationCfg.apply_quantization: QuantizationCfg#apply_quantization.
  QuantizationCfg.use_qmm_best_config: QuantizationCfg#use_qmm_best_config.
  QuantizationCfg.qmm_platform_override: QuantizationCfg#qmm_platform_override.
  QuantizationCfg.qmm_tpu_path_override: QuantizationCfg#qmm_tpu_path_override.
---
# Module: [`easydel/infra/elarge/types/quantization.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py)

## Classes
### `EasyDeLQuantizationCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/quantization.py:31`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L31)
- doc: Extended quantization configuration with layer selection patterns.
- signature: `class EasyDeLQuantizationCfg(TypedDict, total=False):`
- members:
  - `bits` — [`L53`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L53)
  - `dtype` — [`L50`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L50)
  - `group_size` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L52)
  - `jax_native` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L55)
  - `pattern` — [`L56`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L56)
  - `runtime_dtype` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L51)
  - `simulate` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L54)
- used by: [`kv_cache`](quantization.md#QuantizationCfg.kv_cache), [`model`](quantization.md#QuantizationCfg.model)

### `QuantizationCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/quantization.py:59`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L59)
- doc: Quantization configuration for model compression and efficiency.
- signature: `class QuantizationCfg(TypedDict, total=False):`
- members:
  - `apply_quantization` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L82)
  - `kv_cache` — [`L80`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L80)
  - `model` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L81)
  - `platform` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L79)
  - `qmm_platform_override` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L84)
  - `qmm_tpu_path_override` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L85)
  - `use_qmm_best_config` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/quantization.py#L83)
- uses (calls/refs, reference-scoped): [`QuantizationConfig`](../../../layers/quantization/_configs.md#QuantizationConfig), [`EasyDeLPlatforms`](../../etils.md#EasyDeLPlatforms), [`EasyDeLQuantizationCfg`](quantization.md#EasyDeLQuantizationCfg)
- used by: [`quantization`](root.md#eLMConfig.quantization)

