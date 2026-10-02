---
title: 'Module: easydel/infra/elarge/types/model.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/model.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.model`/
symbols:
  ModelCfg: ModelCfg#
  OperationConfigsDict: OperationConfigsDict#
  LoaderCfg: LoaderCfg#
  ModelCfg.task: ModelCfg#task.
  LoaderCfg.dtype: LoaderCfg#dtype.
  LoaderCfg.param_dtype: LoaderCfg#param_dtype.
  LoaderCfg.precision: LoaderCfg#precision.
  OperationConfigsDict.flash_attn2: OperationConfigsDict#flash_attn2.
  OperationConfigsDict.ring: OperationConfigsDict#ring.
  OperationConfigsDict.blocksparse: OperationConfigsDict#blocksparse.
  OperationConfigsDict.ragged_page_attention_v2: OperationConfigsDict#ragged_page_attention_v2.
  OperationConfigsDict.ragged_page_attention_v3: OperationConfigsDict#ragged_page_attention_v3.
  OperationConfigsDict.multi_latent_ragged_page_attention_v1: OperationConfigsDict#multi_latent_ragged_page_attention_v1.
  OperationConfigsDict.unified_attention: OperationConfigsDict#unified_attention.
  OperationConfigsDict.paged_flash_attention: OperationConfigsDict#paged_flash_attention.
  OperationConfigsDict.sdpa: OperationConfigsDict#sdpa.
  OperationConfigsDict.vanilla: OperationConfigsDict#vanilla.
  ModelCfg.name_or_path: ModelCfg#name_or_path.
  ModelCfg.tokenizer: ModelCfg#tokenizer.
  ModelCfg.extra_kwargs: ModelCfg#extra_kwargs.
  LoaderCfg.device: LoaderCfg#device.
  LoaderCfg.verbose: LoaderCfg#verbose.
  LoaderCfg.from_torch: LoaderCfg#from_torch.
  LoaderCfg.trust_remote_code: LoaderCfg#trust_remote_code.
---
# Module: [`easydel/infra/elarge/types/model.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py)

## Classes
### `LoaderCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/model.py:110`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L110)
- doc: Model loading configuration for dtype, precision, and device settings.
- signature: `class LoaderCfg(TypedDict, total=False):`
- members:
  - `device` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L127)
  - `dtype` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L128)
  - `from_torch` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L132)
  - `param_dtype` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L129)
  - `precision` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L130)
  - `trust_remote_code` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L133)
  - `verbose` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L131)
- uses (calls/refs, reference-scoped): [`DTypeLike`](aliases.md#DTypeLike), [`PrecisionLike`](aliases.md#PrecisionLike)
- used by: [`loader`](root.md#eLMConfig.loader)

### `ModelCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/model.py:67`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L67)
- doc: Model configuration section for identifying and loading models.
- signature: `class ModelCfg(TypedDict, total=False):`
- members:
  - `extra_kwargs` — [`L107`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L107)
  - `name_or_path` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L86)
  - `task` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L88)
  - `tokenizer` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L87)
- uses (calls/refs, reference-scoped): [`TaskType`](../../factory.md#TaskType)
- used by: [`model`](root.md#eLMConfig.model), [`reference_model`](root.md#eLMConfig.reference_model), [`teacher_model`](root.md#eLMConfig.teacher_model)

### `OperationConfigsDict`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/model.py:34`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L34)
- doc: Configuration dictionary for ejkernel operation overrides.
- signature: `class OperationConfigsDict(TypedDict, total=False):`
- members:
  - `blocksparse` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L57)
  - `flash_attn2` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L55)
  - `multi_latent_ragged_page_attention_v1` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L60)
  - `paged_flash_attention` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L62)
  - `ragged_page_attention_v2` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L58)
  - `ragged_page_attention_v3` — [`L59`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L59)
  - `ring` — [`L56`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L56)
  - `sdpa` — [`L63`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L63)
  - `unified_attention` — [`L61`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L61)
  - `vanilla` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/model.py#L64)
- used by: [`operation_configs`](engine.md#BaseCfg.operation_configs)

