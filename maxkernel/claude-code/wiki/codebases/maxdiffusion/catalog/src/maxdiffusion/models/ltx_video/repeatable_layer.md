---
title: 'Module: src/maxdiffusion/models/ltx_video/repeatable_layer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/repeatable_layer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.repeatable_layer`/Repeatable
symbols:
  RepeatableLayer.__call__: Layer#__call__().
  RepeatableCarryBlock.__call__: CarryBlock#__call__().
  RepeatableLayer.pspec_name: Layer#pspec_name.
  RepeatableLayer.param_scan_axis: Layer#param_scan_axis.
  RepeatableCarryBlock: CarryBlock#
  RepeatableCarryBlock.module: CarryBlock#module.
  RepeatableCarryBlock.module_init_args: CarryBlock#module_init_args.
  RepeatableCarryBlock.module_init_kwargs: CarryBlock#module_init_kwargs.
  RepeatableLayer.module: Layer#module.
  RepeatableLayer.num_layers: Layer#num_layers.
  RepeatableLayer.module_init_args: Layer#module_init_args.
  RepeatableLayer.module_init_kwargs: Layer#module_init_kwargs.
  RepeatableLayer: Layer#
---
# Module: [`src/maxdiffusion/models/ltx_video/repeatable_layer.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py)

## Classes
### `RepeatableCarryBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/repeatable_layer.py:26`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L26)
- signature: `class RepeatableCarryBlock(nn.Module):`
- members:
  - `module` — [`L27`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L27)
  - `module_init_args` — [`L28`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L28)
  - `module_init_kwargs` — [`L29`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L29)
- protocol/private: `__call__`[`L32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L32)
- used by: [`__call__`](repeatable_layer.md#RepeatableLayer.__call__)

### `RepeatableLayer`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/repeatable_layer.py:45`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L45)
- doc: RepeatableLayer will assume a similar role to torch.nn.ModuleList
- signature: `class RepeatableLayer(nn.Module):`
- members:
  - `module` — [`L53`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L53) — ---
  - `module_init_args` — [`L63`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L63) — ---
  - `module_init_kwargs` — [`L68`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L68) — ---
  - `num_layers` — [`L58`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L58) — ---
  - `param_scan_axis` — [`L78`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L78) — ---
  - `pspec_name` — [`L73`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L73) — ---
- protocol/private: `__call__`[`L87`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/repeatable_layer.py#L87)
- uses (calls/refs, reference-scoped): [`RepeatableCarryBlock`](repeatable_layer.md#RepeatableCarryBlock)

